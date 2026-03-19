"""DynamoDB-backed MLflow model registry store."""

from __future__ import annotations

import time
from typing import Any

from mlflow.entities.model_registry import RegisteredModel, RegisteredModelTag
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.registered_model_alias import RegisteredModelAlias
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.model_registry.abstract_store import AbstractStore

from mlflow_dynamodbstore.cache import ResolutionCache
from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.fts import fts_items_for_text
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_PK,
    GSI1_RUN_PREFIX,
    GSI1_SK,
    GSI2_FTS_NAMES_PREFIX,
    GSI2_MODELS_PREFIX,
    GSI2_PK,
    GSI2_SK,
    GSI3_MODEL_NAME_PREFIX,
    GSI3_PK,
    GSI3_SK,
    GSI5_MODEL_NAMES_PREFIX,
    GSI5_PK,
    GSI5_SK,
    LSI1_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    LSI5_SK,
    PK_MODEL_PREFIX,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
    SK_MODEL_ALIAS_PREFIX,
    SK_MODEL_META,
    SK_MODEL_NAME_REV,
    SK_MODEL_TAG_PREFIX,
    SK_VERSION_PREFIX,
    SK_VERSION_TAG_SUFFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri
from mlflow_dynamodbstore.ids import generate_ulid


def _rev(s: str) -> str:
    """Return reversed string."""
    return s[::-1]


def _int_or_none(value: Any) -> int | None:
    """Convert a value to int, or return None if the value is None."""
    return int(value) if value is not None else None


def _item_to_registered_model(
    item: dict[str, Any],
    tags: list[RegisteredModelTag] | None = None,
    aliases: list[RegisteredModelAlias] | None = None,
) -> RegisteredModel:
    """Convert a DynamoDB item to an MLflow RegisteredModel entity."""
    return RegisteredModel(
        name=item["name"],
        creation_timestamp=_int_or_none(item.get("creation_timestamp")),
        last_updated_timestamp=_int_or_none(item.get("last_updated_timestamp")),
        description=item.get("description") or None,
        latest_versions=[],
        tags=tags or [],
        aliases=aliases or {},
    )


def _pad_version(version: str | int) -> str:
    """Pad a version number to 8 digits."""
    return f"{int(version):08d}"


def _item_to_model_version(
    item: dict[str, Any],
    tags: list[ModelVersionTag] | None = None,
    aliases: list[str] | None = None,
) -> ModelVersion:
    """Convert a DynamoDB item to an MLflow ModelVersion entity."""
    return ModelVersion(
        name=item["name"],
        version=int(item["version"]),  # type: ignore[arg-type]  # MLflow returns int at runtime
        creation_timestamp=_int_or_none(item.get("creation_timestamp")) or 0,
        last_updated_timestamp=_int_or_none(item.get("last_updated_timestamp")),
        description=item.get("description") or None,
        source=item.get("source", ""),
        run_id=item.get("run_id") or None,
        status=item.get("status", "READY"),
        current_stage=item.get("current_stage", "None"),
        tags=tags or [],
        run_link=item.get("run_link") or None,
        aliases=aliases or [],
    )


class DynamoDBRegistryStore(AbstractStore):
    """MLflow model registry store backed by DynamoDB."""

    def __init__(self, store_uri: str) -> None:
        uri = parse_dynamodb_uri(store_uri)
        if uri.deploy:
            ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)
        self._cache = ResolutionCache(workspace=lambda: self._workspace)
        self._config = ConfigReader(self._table)
        self._config.reconcile()

    @property
    def supports_workspaces(self) -> bool:
        """DynamoDB registry store always supports workspaces."""
        return True

    @property
    def _workspace(self) -> str:
        """Return the active workspace from context, defaulting to 'default'."""
        from mlflow.utils.workspace_context import get_request_workspace

        return get_request_workspace() or "default"

    # ------------------------------------------------------------------
    # Name -> ULID resolution
    # ------------------------------------------------------------------

    def _resolve_model_ulid(self, name: str) -> str:
        """Resolve a model name to its ULID, using cache then GSI3."""
        cached = self._cache.get("model_name", name)
        if cached:
            return cached

        results = self._table.query(
            pk=f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if not results:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        model_ulid: str = results[0][GSI3_SK]
        self._cache.put("model_name", name, model_ulid)
        return model_ulid

    # ------------------------------------------------------------------
    # Registered Model CRUD
    # ------------------------------------------------------------------

    def create_registered_model(
        self,
        name: str,
        tags: list[RegisteredModelTag] | None = None,
        description: str | None = None,
        deployment_job_id: str | None = None,
    ) -> RegisteredModel:
        """Create a new registered model."""
        from mlflow.utils.validation import _validate_model_name

        _validate_model_name(name)
        # Check uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Registered Model (name={name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        now_ms = int(time.time() * 1000)
        model_ulid = generate_ulid()

        item: dict[str, Any] = {
            "PK": f"{PK_MODEL_PREFIX}{model_ulid}",
            "SK": SK_MODEL_META,
            "name": name,
            "description": description or "",
            "creation_timestamp": now_ms,
            "last_updated_timestamp": now_ms,
            "workspace": self._workspace,
            "tags": {},
            # LSI attributes
            LSI2_SK: now_ms,
            LSI3_SK: name,
            LSI4_SK: _rev(name),
            # GSI2: list models by workspace
            GSI2_PK: f"{GSI2_MODELS_PREFIX}{self._workspace}",
            GSI2_SK: f"{now_ms}#{name}",
            # GSI3: unique name lookup
            GSI3_PK: f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{name}",
            GSI3_SK: model_ulid,
            # GSI5: all model names
            GSI5_PK: f"{GSI5_MODEL_NAMES_PREFIX}{self._workspace}",
            GSI5_SK: f"{name}#{model_ulid}",
        }

        self._table.put_item(item, condition="attribute_not_exists(PK)")

        # Write NAME_REV item for suffix ILIKE support (GSI5)
        name_rev_item = {
            "PK": f"{PK_MODEL_PREFIX}{model_ulid}",
            "SK": SK_MODEL_NAME_REV,
            GSI5_PK: f"{GSI5_MODEL_NAMES_PREFIX}{self._workspace}",
            GSI5_SK: f"REV#{_rev(name.lower())}#{model_ulid}",
            "name": name,
        }
        self._table.put_item(name_rev_item)

        # Write tags if provided
        model_tags: list[RegisteredModelTag] = []
        if tags:
            for tag in tags:
                self._write_model_tag(model_ulid, tag)
                model_tags.append(tag)

        # Write FTS items for the model name
        fts_items = fts_items_for_text(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            entity_type="M",
            entity_id=model_ulid,
            field=None,
            text=name,
            workspace=self._workspace,
        )
        self._table.batch_write(fts_items)

        self._cache.put("model_name", name, model_ulid)
        return _item_to_registered_model(item, model_tags)

    def get_registered_model(self, name: str) -> RegisteredModel:
        """Fetch a registered model by name."""
        model_ulid = self._resolve_model_ulid(name)

        item = self._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        if item is None:
            raise MlflowException(
                f"Registered Model with name={name} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_model_tags(model_ulid)
        aliases = self._aliases_for_registered_model(model_ulid)
        return _item_to_registered_model(item, tags, aliases)

    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """Rename a registered model."""
        model_ulid = self._resolve_model_ulid(name)

        # Check new name uniqueness
        existing = self._table.query(
            pk=f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{new_name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Registered Model (name={new_name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
            updates={
                "name": new_name,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
                LSI3_SK: new_name,
                LSI4_SK: _rev(new_name),
                GSI2_SK: f"{now_ms}#{new_name}",
                GSI3_PK: f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{new_name}",
                GSI5_SK: f"{new_name}#{model_ulid}",
            },
        )

        # Update NAME_REV item with new reversed name
        self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
            updates={
                GSI5_SK: f"REV#{_rev(new_name.lower())}#{model_ulid}",
                "name": new_name,
            },
        )

        # Delete old FTS items by querying the reverse index for this entity
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"
        old_rev_items = self._table.query(
            pk=pk,
            sk_prefix=f"{SK_FTS_REV_PREFIX}M#{model_ulid}",
        )
        for rev_item in old_rev_items:
            self._table.delete_item(pk=pk, sk=rev_item["SK"])

        # Also delete the forward FTS items
        old_fts_items = self._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        for fts_item in old_fts_items:
            self._table.delete_item(pk=pk, sk=fts_item["SK"])

        # Write new FTS items for the new name
        new_fts_items = fts_items_for_text(
            pk=pk,
            entity_type="M",
            entity_id=model_ulid,
            field=None,
            text=new_name,
            workspace=self._workspace,
        )
        self._table.batch_write(new_fts_items)

        # Invalidate old name cache, cache new name
        self._cache.invalidate("model_name", name)
        self._cache.put("model_name", new_name, model_ulid)

        tags = self._get_model_tags(model_ulid)
        updated_item = self._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert updated_item is not None
        return _item_to_registered_model(updated_item, tags)

    def update_registered_model(
        self,
        name: str,
        description: str,
        deployment_job_id: str | None = None,
    ) -> RegisteredModel:
        """Update a registered model's description."""
        model_ulid = self._resolve_model_ulid(name)
        now_ms = int(time.time() * 1000)

        updated_item = self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
            updates={
                "description": description,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
            },
        )

        tags = self._get_model_tags(model_ulid)
        assert updated_item is not None
        return _item_to_registered_model(updated_item, tags)

    def delete_registered_model(self, name: str) -> None:
        """Delete a registered model and all its items."""
        model_ulid = self._resolve_model_ulid(name)
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"

        # Query all items in the partition and delete each
        items = self._table.query(pk=pk)
        for item in items:
            self._table.delete_item(pk=pk, sk=item["SK"])

        self._cache.invalidate("model_name", name)

    def search_registered_models(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> list[RegisteredModel]:
        """Search registered models with filter and order_by support."""
        from mlflow.utils.search_utils import SearchModelUtils

        from mlflow_dynamodbstore.dynamodb.search import (
            FilterPredicate,
            _compare,
        )

        # Use SearchModelUtils (not parse_experiment_filter) — handles
        # backtick-quoted tag names and model registry filter syntax.
        if filter_string:
            parsed = SearchModelUtils.parse_search_filter(filter_string)
            predicates = [
                FilterPredicate(
                    field_type=p["type"],
                    key=p["key"],
                    op=p["comparator"],
                    value=p.get("value"),
                )
                for p in parsed
            ]
        else:
            predicates = []

        name_pred = next(
            (p for p in predicates if p.field_type == "attribute" and p.key == "name"),
            None,
        )
        tag_preds = [p for p in predicates if p.field_type == "tag"]

        if name_pred and name_pred.op == "=":
            models = self._search_models_by_name_exact(name_pred.value)
        elif name_pred and name_pred.op in ("LIKE", "ILIKE"):
            models = self._search_models_by_name_like(name_pred.value, name_pred.op)
        else:
            models = self._search_models_by_gsi2()

        # Apply tag filters
        if tag_preds:
            models = self._filter_models_by_tags(models, tag_preds, _compare)

        # Apply order_by
        if order_by:
            models = self._sort_models(models, order_by)

        from mlflow.store.entities import PagedList

        if max_results:
            models = models[:max_results]
        return PagedList(models, token=None)

    def _search_models_by_name_exact(self, name: str) -> list[RegisteredModel]:
        """Look up a single model by exact name via GSI3."""
        results = self._table.query(
            pk=f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if not results:
            return []

        model_ulid = results[0][GSI3_SK]
        item = self._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        if item is None:
            return []

        tags = self._get_model_tags(model_ulid)
        return [_item_to_registered_model(item, tags)]

    def _search_models_by_name_like(self, pattern: str, op: str) -> list[RegisteredModel]:
        """Search models by name LIKE pattern using GSI5 or FTS."""
        import fnmatch as _fnmatch

        # Convert SQL LIKE pattern to fnmatch: % -> *, _ -> ?
        fn_pattern = pattern.replace("%", "*").replace("_", "?")
        if op == "ILIKE":
            fn_pattern = fn_pattern.lower()

        # Prefix-only: use GSI5 begins_with
        if pattern.endswith("%") and "%" not in pattern[:-1]:
            prefix = pattern[:-1]
            items = self._table.query(
                pk=f"{GSI5_MODEL_NAMES_PREFIX}{self._workspace}",
                sk_prefix=prefix,
                index_name="gsi5",
            )
            models: list[RegisteredModel] = []
            for item in items:
                model_name = item.get("name")
                if model_name:
                    model_ulid = item["PK"].replace(PK_MODEL_PREFIX, "")
                    tags = self._get_model_tags(model_ulid)
                    models.append(_item_to_registered_model(item, tags))
            return models

        # Contains pattern (%word%): use FTS via GSI2
        if pattern.startswith("%") and pattern.endswith("%"):
            search_term = pattern.strip("%")
            return self._search_models_by_fts(search_term)

        # General LIKE: fall back to GSI2 scan + Python filter
        all_models = self._search_models_by_gsi2()
        return [
            m
            for m in all_models
            if _fnmatch.fnmatch(m.name.lower() if op == "ILIKE" else m.name, fn_pattern)
        ]

    def _search_models_by_fts(self, search_term: str) -> list[RegisteredModel]:
        """Search models by FTS tokens via GSI2 cross-partition index."""
        from mlflow_dynamodbstore.dynamodb.fts import tokenize_trigrams, tokenize_words

        gsi2pk = f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}"

        # Try word tokens first
        word_tokens = tokenize_words(search_term)
        if word_tokens:
            model_ulid_sets: list[set[str]] = []
            for token in word_tokens:
                sk_prefix = f"W#{token}#M#"
                fts_items = self._table.query(
                    pk=gsi2pk,
                    sk_prefix=sk_prefix,
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    # Pattern: W#<token>#M#<model_ulid>
                    parts = gsi2sk.split("#")
                    for i, part in enumerate(parts):
                        if part == "M" and i + 1 < len(parts):
                            ids.add(parts[i + 1])
                            break
                model_ulid_sets.append(ids)

            if model_ulid_sets:
                result_ids = model_ulid_sets[0]
                for s in model_ulid_sets[1:]:
                    result_ids &= s
                if result_ids:
                    return self._fetch_models_by_ulids(list(result_ids))

        # Fallback to trigrams
        trigram_tokens = tokenize_trigrams(search_term)
        if trigram_tokens:
            model_ulid_sets = []
            for token in trigram_tokens:
                sk_prefix = f"3#{token}#M#"
                fts_items = self._table.query(
                    pk=gsi2pk,
                    sk_prefix=sk_prefix,
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    parts = gsi2sk.split("#")
                    for i, part in enumerate(parts):
                        if part == "M" and i + 1 < len(parts):
                            ids.add(parts[i + 1])
                            break
                model_ulid_sets.append(ids)

            if model_ulid_sets:
                result_ids = model_ulid_sets[0]
                for s in model_ulid_sets[1:]:
                    result_ids &= s
                if result_ids:
                    return self._fetch_models_by_ulids(list(result_ids))

        return []

    def _fetch_models_by_ulids(self, model_ulids: list[str]) -> list[RegisteredModel]:
        """Fetch registered models by their ULIDs."""
        models: list[RegisteredModel] = []
        for ulid in model_ulids:
            item = self._table.get_item(
                pk=f"{PK_MODEL_PREFIX}{ulid}",
                sk=SK_MODEL_META,
            )
            if item is not None:
                tags = self._get_model_tags(ulid)
                models.append(_item_to_registered_model(item, tags))
        return models

    def _search_models_by_gsi2(self) -> list[RegisteredModel]:
        """List all models via GSI2."""
        items = self._table.query(
            pk=f"{GSI2_MODELS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )
        models: list[RegisteredModel] = []
        for item in items:
            model_name = item.get("name")
            if model_name:
                model_ulid = item["PK"].replace(PK_MODEL_PREFIX, "")
                tags = self._get_model_tags(model_ulid)
                models.append(_item_to_registered_model(item, tags))
        return models

    def _filter_models_by_tags(
        self,
        models: list[RegisteredModel],
        tag_preds: list[Any],
        compare_fn: Any,
    ) -> list[RegisteredModel]:
        """Filter models by tag predicates."""
        filtered: list[RegisteredModel] = []
        for model in models:
            # Use _tags (includes mlflow.* prefixed tags) not .tags (which filters them out)
            tag_dict: dict[str, str] = {}
            if hasattr(model, "_tags") and isinstance(model._tags, dict):
                tag_dict = model._tags
            elif isinstance(model.tags, dict):
                tag_dict = model.tags
            else:
                for t in model.tags:
                    tag_dict[t.key] = t.value

            match = True
            for pred in tag_preds:
                actual = tag_dict.get(pred.key)
                if not compare_fn(actual, pred.op, pred.value):
                    match = False
                    break
            if match:
                filtered.append(model)
        return filtered

    def _sort_models(
        self, models: list[RegisteredModel], order_by: list[str]
    ) -> list[RegisteredModel]:
        """Sort models by order_by tokens."""
        if not order_by:
            return models

        # Parse the first order_by token
        token = order_by[0].strip()
        parts = token.rsplit(None, 1)
        if len(parts) == 2 and parts[1].upper() in ("ASC", "DESC"):
            key = parts[0].lower()
            reverse = parts[1].upper() == "DESC"
        else:
            key = token.lower()
            reverse = False

        if key == "name":
            models = sorted(models, key=lambda m: m.name, reverse=reverse)
        elif key in ("last_updated_timestamp", "timestamp"):
            models = sorted(
                models,
                key=lambda m: m.last_updated_timestamp or 0,
                reverse=reverse,
            )

        return models

    # ------------------------------------------------------------------
    # Registered Model Tags
    # ------------------------------------------------------------------

    def set_registered_model_tag(self, name: str, tag: RegisteredModelTag) -> None:
        """Set a tag on a registered model."""
        model_ulid = self._resolve_model_ulid(name)
        self._write_model_tag(model_ulid, tag)

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """Delete a tag from a registered model."""
        model_ulid = self._resolve_model_ulid(name)
        self._table.delete_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_MODEL_TAG_PREFIX}{key}",
        )
        if self._config.should_denormalize(None, key):
            self._remove_denormalized_tag(
                pk=f"{PK_MODEL_PREFIX}{model_ulid}",
                sk=SK_MODEL_META,
                tag_key=key,
            )

    def _write_model_tag(self, model_ulid: str, tag: RegisteredModelTag) -> None:
        """Write a model tag item."""
        item = {
            "PK": f"{PK_MODEL_PREFIX}{model_ulid}",
            "SK": f"{SK_MODEL_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)
        if self._config.should_denormalize(None, tag.key):
            self._denormalize_tag(
                pk=f"{PK_MODEL_PREFIX}{model_ulid}",
                sk=SK_MODEL_META,
                tag_key=tag.key,
                tag_value=tag.value,
            )

    def _get_model_tags(self, model_ulid: str) -> list[RegisteredModelTag]:
        """Read all tags for a registered model."""
        items = self._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk_prefix=SK_MODEL_TAG_PREFIX,
        )
        return [RegisteredModelTag(item["key"], item["value"]) for item in items]

    # ------------------------------------------------------------------
    # Model Version CRUD
    # ------------------------------------------------------------------

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: str | None = None,
        tags: list[Any] | None = None,
        run_link: str | None = None,
        description: str | None = None,
        local_model_path: str | None = None,
        model_id: str | None = None,
    ) -> ModelVersion:
        """Create a new model version under the given registered model."""
        model_ulid = self._resolve_model_ulid(name)
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"

        # Determine next version number
        existing = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
        # Filter out tag items (V#00000001#TAG#key)
        version_items = [it for it in existing if SK_VERSION_TAG_SUFFIX not in it["SK"]]
        next_ver = len(version_items) + 1
        padded = _pad_version(next_ver)

        now_ms = int(time.time() * 1000)

        item: dict[str, Any] = {
            "PK": pk,
            "SK": f"{SK_VERSION_PREFIX}{padded}",
            "name": name,
            "version": padded,
            "source": source or "",
            "run_id": run_id or "",
            "run_link": run_link or "",
            "description": description or "",
            "status": "READY",
            "current_stage": "None",
            "creation_timestamp": now_ms,
            "last_updated_timestamp": now_ms,
            "tags": {},
            # LSI attributes
            LSI1_SK: str(now_ms),
            LSI2_SK: now_ms,
            LSI3_SK: f"None#{padded}",
        }

        # Sparse LSI keys — DynamoDB rejects empty string index keys
        if source:
            item[LSI4_SK] = source.lower()
        if run_id:
            item[LSI5_SK] = f"{run_id}#{padded}"

        # GSI1: run linkage (only if run_id provided)
        if run_id:
            item[GSI1_PK] = f"{GSI1_RUN_PREFIX}{run_id}"
            item[GSI1_SK] = f"MV#{model_ulid}#{padded}"

        self._table.put_item(item)

        # Update model's last_updated_timestamp
        self._table.update_item(
            pk=pk,
            sk=SK_MODEL_META,
            updates={"last_updated_timestamp": now_ms, LSI2_SK: now_ms},
        )

        # Write tags if provided
        version_tags: list[ModelVersionTag] = []
        if tags:
            for tag in tags:
                self._write_version_tag(model_ulid, padded, tag)
                version_tags.append(tag)

        return _item_to_model_version(item, version_tags)

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        """Fetch a model version by name and version number."""
        try:
            model_ulid = self._resolve_model_ulid(name)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                raise MlflowException(
                    f"Model Version (name={name}, version={version}) not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from None
            raise
        padded = _pad_version(version)

        item = self._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        if item is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_version_tags(model_ulid, padded)
        aliases = self._aliases_for_model_version(model_ulid, int(version))
        return _item_to_model_version(item, tags, aliases)

    def update_model_version(self, name: str, version: str, description: str) -> ModelVersion:
        """Update a model version's description."""
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        now_ms = int(time.time() * 1000)

        updated_item = self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
            updates={
                "description": description,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
            },
        )

        if updated_item is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_version_tags(model_ulid, padded)
        return _item_to_model_version(updated_item, tags)

    def delete_model_version(self, name: str, version: str) -> None:
        """Delete a model version and its tags."""
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"

        # Delete version item
        self._table.delete_item(pk=pk, sk=f"{SK_VERSION_PREFIX}{padded}")

        # Delete version tags
        tag_prefix = f"{SK_VERSION_PREFIX}{padded}{SK_VERSION_TAG_SUFFIX}"
        tag_items = self._table.query(pk=pk, sk_prefix=tag_prefix)
        for tag_item in tag_items:
            self._table.delete_item(pk=pk, sk=tag_item["SK"])

        # Delete aliases pointing to this version
        for alias_name in self._aliases_for_model_version(model_ulid, int(version)):
            self._table.delete_item(pk=pk, sk=f"{SK_MODEL_ALIAS_PREFIX}{alias_name}")

    def search_model_versions(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> list[ModelVersion]:
        """Search model versions with filter support."""
        import re as _re

        model_name: str | None = None
        run_id_filter: str | None = None

        if filter_string:
            # Parse name = 'value'
            name_match = _re.search(r"name\s*=\s*'([^']+)'", filter_string)
            if name_match:
                model_name = name_match.group(1)

            # Parse run_id = 'value'
            run_id_match = _re.search(r"run_id\s*=\s*'([^']+)'", filter_string)
            if run_id_match:
                run_id_filter = run_id_match.group(1)

        if model_name:
            # Targeted: get versions for a specific model
            try:
                model_ulid = self._resolve_model_ulid(model_name)
            except MlflowException:
                return []
            versions = self._get_versions_for_model(model_ulid, model_name)
        elif run_id_filter:
            # Search by run_id via GSI1
            versions = self._search_versions_by_run_id(run_id_filter)
        else:
            # Default: list all versions across all models
            versions = self._list_all_versions()

        # Apply run_id post-filter if name was also specified
        if run_id_filter and model_name:
            versions = [v for v in versions if v.run_id == run_id_filter]

        if max_results:
            versions = versions[:max_results]
        from mlflow.store.entities import PagedList

        return PagedList(versions, token=None)

    def _get_versions_for_model(self, model_ulid: str, model_name: str) -> list[ModelVersion]:
        """Get all versions for a specific model."""
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"
        ver_items = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
        versions: list[ModelVersion] = []
        for vi in ver_items:
            if SK_VERSION_TAG_SUFFIX in vi["SK"]:
                continue
            padded = vi["SK"].replace(SK_VERSION_PREFIX, "")
            tags = self._get_version_tags(model_ulid, padded)
            versions.append(_item_to_model_version(vi, tags))
        return versions

    def _search_versions_by_run_id(self, run_id: str) -> list[ModelVersion]:
        """Search model versions by run_id via GSI1."""
        items = self._table.query(
            pk=f"{GSI1_RUN_PREFIX}{run_id}",
            sk_prefix="MV#",
            index_name="gsi1",
        )
        versions: list[ModelVersion] = []
        for item in items:
            # GSI1 SK pattern: MV#<model_ulid>#<padded_version>
            gsi1_sk = item.get(GSI1_SK, "")
            parts = gsi1_sk.split("#")
            if len(parts) >= 3:
                model_ulid = parts[1]
                padded = parts[2]
                vi = self._table.get_item(
                    pk=f"{PK_MODEL_PREFIX}{model_ulid}",
                    sk=f"{SK_VERSION_PREFIX}{padded}",
                )
                if vi is not None:
                    tags = self._get_version_tags(model_ulid, padded)
                    versions.append(_item_to_model_version(vi, tags))
        return versions

    def _list_all_versions(self) -> list[ModelVersion]:
        """List all model versions across all models."""
        model_items = self._table.query(
            pk=f"{GSI2_MODELS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )
        versions: list[ModelVersion] = []
        for model_item in model_items:
            model_ulid = model_item["PK"].replace(PK_MODEL_PREFIX, "")
            pk = f"{PK_MODEL_PREFIX}{model_ulid}"
            ver_items = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
            for vi in ver_items:
                if SK_VERSION_TAG_SUFFIX in vi["SK"]:
                    continue
                padded = vi["SK"].replace(SK_VERSION_PREFIX, "")
                tags = self._get_version_tags(model_ulid, padded)
                versions.append(_item_to_model_version(vi, tags))
        return versions

    def get_latest_versions(self, name: str, stages: list[str] | None = None) -> list[ModelVersion]:
        """Get the latest version for each requested stage."""
        model_ulid = self._resolve_model_ulid(name)
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"

        if not stages:
            # Get all versions and determine unique stages
            ver_items = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
            ver_items = [vi for vi in ver_items if SK_VERSION_TAG_SUFFIX not in vi["SK"]]
            # Group by stage, pick latest (highest version) per stage
            stage_latest: dict[str, dict[str, Any]] = {}
            for vi in ver_items:
                stage = vi.get("current_stage", "None")
                existing_item = stage_latest.get(stage)
                if existing_item is None or vi["version"] > existing_item["version"]:
                    stage_latest[stage] = vi
            results = []
            for vi in stage_latest.values():
                padded = vi["SK"].replace(SK_VERSION_PREFIX, "")
                tags = self._get_version_tags(model_ulid, padded)
                results.append(_item_to_model_version(vi, tags))
            return results

        results = []
        for stage in stages:
            # Query LSI3 where lsi3sk begins_with "stage#"
            stage_items = self._table.query(
                pk=pk,
                sk_prefix=f"{stage}#",
                index_name="lsi3",
                scan_forward=False,
                limit=1,
            )
            if stage_items:
                vi = stage_items[0]
                padded = vi["SK"].replace(SK_VERSION_PREFIX, "")
                tags = self._get_version_tags(model_ulid, padded)
                results.append(_item_to_model_version(vi, tags))
        return results

    def set_model_version_tag(self, name: str, version: str, tag: Any) -> None:
        """Set a tag on a model version."""
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        self._write_version_tag(model_ulid, padded, tag)

    def delete_model_version_tag(self, name: str, version: str, key: str) -> None:
        """Delete a tag from a model version."""
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        self._table.delete_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}{SK_VERSION_TAG_SUFFIX}{key}",
        )
        if self._config.should_denormalize(None, key):
            self._remove_denormalized_tag(
                pk=f"{PK_MODEL_PREFIX}{model_ulid}",
                sk=f"{SK_VERSION_PREFIX}{padded}",
                tag_key=key,
            )

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        """Return the source URI for a model version."""
        mv = self.get_model_version(name, version)
        return mv.source or ""

    def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool,
    ) -> ModelVersion:
        """Transition a model version to a new stage."""
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        now_ms = int(time.time() * 1000)

        updated_item = self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
            updates={
                "current_stage": stage,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
                LSI3_SK: f"{stage}#{padded}",
            },
        )

        if updated_item is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_version_tags(model_ulid, padded)
        return _item_to_model_version(updated_item, tags)

    def copy_model_version(self, src_mv: ModelVersion, dst_name: str) -> ModelVersion:
        """Copy a model version to another registered model."""
        return self.create_model_version(
            name=dst_name,
            source=src_mv.source or "",
            run_id=src_mv.run_id,
            description=src_mv.description,
        )

    # ------------------------------------------------------------------
    # Model Version Tags
    # ------------------------------------------------------------------

    def _write_version_tag(
        self, model_ulid: str, padded_version: str, tag: ModelVersionTag
    ) -> None:
        """Write a version tag item."""
        item = {
            "PK": f"{PK_MODEL_PREFIX}{model_ulid}",
            "SK": f"{SK_VERSION_PREFIX}{padded_version}{SK_VERSION_TAG_SUFFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)
        if self._config.should_denormalize(None, tag.key):
            self._denormalize_tag(
                pk=f"{PK_MODEL_PREFIX}{model_ulid}",
                sk=f"{SK_VERSION_PREFIX}{padded_version}",
                tag_key=tag.key,
                tag_value=tag.value,
            )

    def _denormalize_tag(self, pk: str, sk: str, tag_key: str, tag_value: str) -> None:
        """Update the tags map on an item with a new key-value entry."""
        self._table._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression="SET #tags.#k = :v",
            ExpressionAttributeNames={"#tags": "tags", "#k": tag_key},
            ExpressionAttributeValues={":v": tag_value},
        )

    def _remove_denormalized_tag(self, pk: str, sk: str, tag_key: str) -> None:
        """Remove a key from the tags map on an item."""
        self._table._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression="REMOVE #tags.#k",
            ExpressionAttributeNames={"#tags": "tags", "#k": tag_key},
        )

    def _get_version_tags(self, model_ulid: str, padded_version: str) -> list[ModelVersionTag]:
        """Read all tags for a model version."""
        prefix = f"{SK_VERSION_PREFIX}{padded_version}{SK_VERSION_TAG_SUFFIX}"
        items = self._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk_prefix=prefix,
        )
        return [ModelVersionTag(item["key"], item["value"]) for item in items]

    def _get_model_aliases(self, model_ulid: str) -> list[dict[str, Any]]:
        """Query all alias items for a model. Returns raw DynamoDB items."""
        return self._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk_prefix=SK_MODEL_ALIAS_PREFIX,
        )

    def _aliases_for_registered_model(self, model_ulid: str) -> list[RegisteredModelAlias]:
        """Return aliases as RegisteredModelAlias objects for a RegisteredModel."""
        items = self._get_model_aliases(model_ulid)
        return [RegisteredModelAlias(item["alias"], int(item["version"])) for item in items]

    def _aliases_for_model_version(self, model_ulid: str, version: int) -> list[str]:
        """Return alias names pointing to a specific version."""
        items = self._get_model_aliases(model_ulid)
        return [item["alias"] for item in items if int(item["version"]) == version]

    # ------------------------------------------------------------------
    # Alias operations (Task 19)
    # ------------------------------------------------------------------

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        """Set an alias pointing to a specific version of a registered model."""
        model_ulid = self._resolve_model_ulid(name)
        # Verify the version exists by fetching it (raises if not found)
        self.get_model_version(name, version)
        item = {
            "PK": f"{PK_MODEL_PREFIX}{model_ulid}",
            "SK": f"{SK_MODEL_ALIAS_PREFIX}{alias}",
            "alias": alias,
            "version": version,
        }
        self._table.put_item(item)

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        """Delete an alias from a registered model (no-op if alias does not exist)."""
        model_ulid = self._resolve_model_ulid(name)
        self._table.delete_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_MODEL_ALIAS_PREFIX}{alias}",
        )

    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
        """Return the model version that the given alias resolves to."""
        model_ulid = self._resolve_model_ulid(name)
        item = self._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_MODEL_ALIAS_PREFIX}{alias}",
        )
        if item is None:
            raise MlflowException(
                f"Registered model alias {alias} not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        version: str = item["version"]
        return self.get_model_version(name, version)
