"""DynamoDB-backed MLflow model registry store."""

from __future__ import annotations

import base64
import json
from typing import Any

from mlflow.entities.model_registry import RegisteredModel, RegisteredModelTag
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.registered_model_alias import RegisteredModelAlias
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.time import get_current_time_millis

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


# MLflow attribute name → DynamoDB item field name for model versions
_MV_ATTRIBUTE_MAP: dict[str, str] = {
    "source_path": "source",
    "version_number": "version",
}


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
        from mlflow.prompt.registry_utils import handle_resource_already_exist_error, has_prompt_tag
        from mlflow.utils.validation import _validate_model_name

        _validate_model_name(name)
        # Check uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            existing_ulid = existing[0][GSI3_SK]
            existing_tags = self._get_model_tags(existing_ulid)
            handle_resource_already_exist_error(
                name, has_prompt_tag(existing_tags), has_prompt_tag(tags)
            )

        now_ms = get_current_time_millis()
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
            GSI2_SK: f"{now_ms:020d}#{name}",
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
        rm = _item_to_registered_model(item, tags, aliases)
        rm.latest_versions = self.get_latest_versions(name)
        return rm

    def rename_registered_model(self, name: str, new_name: str) -> RegisteredModel:
        """Rename a registered model."""
        from mlflow.utils.validation import _validate_model_renaming

        _validate_model_renaming(new_name)
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

        now_ms = get_current_time_millis()

        self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
            updates={
                "name": new_name,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
                LSI3_SK: new_name,
                LSI4_SK: _rev(new_name),
                GSI2_SK: f"{now_ms:020d}#{new_name}",
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

        # Update name on all model version items
        ver_items = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
        for vi in ver_items:
            if SK_VERSION_TAG_SUFFIX not in vi["SK"]:
                self._table.update_item(pk=pk, sk=vi["SK"], updates={"name": new_name})

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
        now_ms = get_current_time_millis()

        updated_item = self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
            updates={
                "description": description,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
                GSI2_SK: f"{now_ms:020d}#{name}",
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
        """Search registered models with filter, ordering, and pagination."""

        from mlflow.store.entities import PagedList
        from mlflow.store.model_registry import (
            SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
            SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
        )
        from mlflow.utils.search_utils import SearchModelUtils, SearchUtils

        from mlflow_dynamodbstore.dynamodb.search import (
            FilterPredicate,
            _compare,
        )

        if max_results is None:
            max_results = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        # Validate order_by clauses
        for clause in order_by or []:
            SearchUtils.parse_order_by_for_search_registered_models(clause)

        # Parse filters
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
        tag_preds = [p for p in predicates if p.field_type == "tag" and p.key != IS_PROMPT_TAG_KEY]

        # Post-filter for tags and prompts (used by non-paginated paths)
        def _post_filter(models: list[RegisteredModel]) -> list[RegisteredModel]:
            if tag_preds:
                models = self._filter_models_by_tags(models, tag_preds, _compare)
            if self._is_querying_prompt(predicates):
                models = [m for m in models if m._is_prompt()]
            else:
                models = [m for m in models if not m._is_prompt()]
            return models

        # Filter for paginated path (tags + prompts, applied per-item)
        def _paginated_filter(model: RegisteredModel) -> bool:
            if tag_preds:
                tag_dict: dict[str, str] = {}
                if hasattr(model, "_tags") and isinstance(model._tags, dict):
                    tag_dict = model._tags
                elif isinstance(model.tags, dict):
                    tag_dict = model.tags
                else:
                    for t in model.tags:
                        tag_dict[t.key] = t.value
                for pred in tag_preds:
                    if not _compare(tag_dict.get(pred.key), pred.op, pred.value):
                        return False
            if self._is_querying_prompt(predicates):
                if not model._is_prompt():
                    return False
            elif model._is_prompt():
                return False
            return True

        if name_pred and name_pred.op == "=":
            # Exact name — no pagination needed
            models = _post_filter(self._search_models_by_name_exact(name_pred.value))
            return PagedList(models[:max_results], token=None)

        if name_pred and name_pred.op in ("LIKE", "ILIKE"):
            is_prefix = name_pred.value.endswith("%") and "%" not in name_pred.value[:-1]
            if not is_prefix:
                # Non-prefix LIKE — FTS / general fallback, no pagination
                models = _post_filter(
                    self._search_models_by_name_like(name_pred.value, name_pred.op)
                )
                models = self._sort_models(models, order_by)
                return PagedList(models[:max_results], token=None)
            # Prefix LIKE falls through to paginated path below

        # Paginated path: no name filter, or prefix LIKE
        name_prefix = name_pred.value[:-1] if name_pred else None
        index_name, scan_forward = self._resolve_registered_model_order(order_by)
        if index_name == "gsi5":
            pk = f"{GSI5_MODEL_NAMES_PREFIX}{self._workspace}"
            sk_prefix = name_prefix
        else:
            pk = f"{GSI2_MODELS_PREFIX}{self._workspace}"
            sk_prefix = None

        # When using GSI2 with a prefix LIKE, add name filtering
        if name_prefix and index_name != "gsi5":
            _inner = _paginated_filter

            def _name_and_paginated_filter(model: RegisteredModel) -> bool:
                if not model.name.startswith(name_prefix):
                    return False
                return _inner(model)

            filter_fn = _name_and_paginated_filter
        else:
            filter_fn = _paginated_filter

        models, next_token = self._search_models_paginated(
            pk=pk,
            index_name=index_name,
            sk_prefix=sk_prefix,
            scan_forward=scan_forward,
            max_results=max_results,
            page_token=page_token,
            filter_fn=filter_fn,
        )
        return PagedList(models, next_token)

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
                sk_prefix = f"W#M#{token}#"
                fts_items = self._table.query(
                    pk=gsi2pk,
                    sk_prefix=sk_prefix,
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    # Pattern: W#M#<token>#<model_ulid>
                    parts = gsi2sk.split("#")
                    if len(parts) >= 4:
                        ids.add(parts[3])
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
                sk_prefix = f"3#M#{token}#"
                fts_items = self._table.query(
                    pk=gsi2pk,
                    sk_prefix=sk_prefix,
                    index_name="gsi2",
                )
                ids = set()
                for item in fts_items:
                    gsi2sk = item.get(GSI2_SK, "")
                    # Pattern: 3#M#<token>#<model_ulid>
                    parts = gsi2sk.split("#")
                    if len(parts) >= 4:
                        ids.add(parts[3])
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

    @staticmethod
    def _is_querying_prompt(predicates: list[Any]) -> bool:
        """Check if the filter explicitly requests prompt entities."""

        for p in predicates:
            if p.field_type != "tag" or p.key != IS_PROMPT_TAG_KEY:
                continue
            return bool(
                (p.op == "=" and p.value.lower() == "true")
                or (p.op == "!=" and p.value.lower() == "false")
            )
        return False

    @staticmethod
    def _version_is_prompt(version: ModelVersion) -> bool:
        """Check if a model version is a prompt version."""

        tags = version.tags or {}
        return tags.get(IS_PROMPT_TAG_KEY, "false").lower() == "true"

    @staticmethod
    def _filter_versions_by_tags(
        versions: list[ModelVersion],
        tag_preds: list[Any],
        compare_fn: Any,
    ) -> list[ModelVersion]:
        """Filter model versions by tag predicates."""
        filtered: list[ModelVersion] = []
        for version in versions:
            tag_dict = version.tags or {}
            match = True
            for pred in tag_preds:
                actual = tag_dict.get(pred.key)
                if not compare_fn(actual, pred.op, pred.value):
                    match = False
                    break
            if match:
                filtered.append(version)
        return filtered

    @staticmethod
    def _encode_page_token(last_evaluated_key: dict[str, Any]) -> str:
        """Encode a DynamoDB LastEvaluatedKey as an opaque page token."""
        from mlflow_dynamodbstore.dynamodb.table import convert_decimals

        return base64.b64encode(
            json.dumps(convert_decimals(last_evaluated_key)).encode("utf-8")
        ).decode("ascii")

    @staticmethod
    def _decode_page_token(page_token: str | None) -> dict[str, Any] | None:
        """Decode an opaque page token back to a DynamoDB ExclusiveStartKey."""
        if not page_token:
            return None
        try:
            decoded = base64.b64decode(page_token)
        except Exception:
            raise MlflowException(
                "Invalid page token, could not base64-decode",
                error_code=INVALID_PARAMETER_VALUE,
            )
        try:
            result: dict[str, Any] = json.loads(decoded)
            return result
        except Exception:
            raise MlflowException(
                f"Invalid page token, decoded value={decoded!r}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @staticmethod
    def _build_exclusive_start_key(item: dict[str, Any], index_name: str | None) -> dict[str, Any]:
        """Build an ExclusiveStartKey from an item for the given index."""
        from mlflow_dynamodbstore.dynamodb.table import _INDEX_KEY_ATTRS

        key: dict[str, Any] = {"PK": item["PK"], "SK": item["SK"]}
        if index_name is None:
            return key
        idx_pk, idx_sk = _INDEX_KEY_ATTRS[index_name]
        if idx_pk != "PK":
            key[idx_pk] = item[idx_pk]
        key[idx_sk] = item[idx_sk]
        return key

    @staticmethod
    def _resolve_registered_model_order(
        order_by: list[str] | None,
    ) -> tuple[str, bool]:
        """Determine (index_name, scan_forward) from order_by clauses."""
        from mlflow.utils.search_utils import SearchUtils

        if not order_by:
            return "gsi5", True  # default: name ASC

        attribute, ascending = SearchUtils.parse_order_by_for_search_registered_models(order_by[0])
        if attribute == "name":
            return "gsi5", ascending
        if attribute in SearchUtils.VALID_TIMESTAMP_ORDER_BY_KEYS:
            return "gsi2", ascending
        return "gsi5", True

    def _search_models_paginated(
        self,
        pk: str,
        index_name: str,
        sk_prefix: str | None = None,
        scan_forward: bool = True,
        max_results: int = 100,
        page_token: str | None = None,
        filter_fn: Any | None = None,
    ) -> tuple[list[RegisteredModel], str | None]:
        """Query an index page-by-page, apply filter_fn, return (models, next_token)."""
        exclusive_start_key = self._decode_page_token(page_token)
        results: list[RegisteredModel] = []
        # Track the raw DynamoDB item for the last *kept* result (for cursor construction)
        kept_items: list[dict[str, Any]] = []

        while len(results) < max_results + 1:
            needed = max_results + 1 - len(results)
            batch_size = needed * 2 if filter_fn else needed
            items, lek = self._table.query_page(
                pk=pk,
                sk_prefix=sk_prefix,
                index_name=index_name,
                limit=batch_size,
                scan_forward=scan_forward,
                exclusive_start_key=exclusive_start_key,
            )
            for item in items:
                # Skip NAME_REV items in GSI5
                if item.get(GSI5_SK, "").startswith("REV#"):
                    continue
                model_name = item.get("name")
                if not model_name:
                    continue
                model_ulid = item["PK"].replace(PK_MODEL_PREFIX, "")
                tags = self._get_model_tags(model_ulid)
                aliases = self._aliases_for_registered_model(model_ulid)
                rm = _item_to_registered_model(item, tags, aliases)
                if filter_fn and not filter_fn(rm):
                    continue
                results.append(rm)
                kept_items.append(item)
                if len(results) > max_results:
                    break

            if lek is None or len(results) > max_results:
                break
            exclusive_start_key = lek

        if len(results) > max_results:
            results = results[:max_results]
            kept_items = kept_items[:max_results]
            last_kept = kept_items[-1]
            next_token = self._encode_page_token(
                self._build_exclusive_start_key(last_kept, index_name)
            )
        else:
            next_token = None

        return results, next_token

    def _sort_models(
        self, models: list[RegisteredModel], order_by: list[str] | None
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
        from mlflow.utils.validation import _validate_model_name, _validate_registered_model_tag

        _validate_model_name(name)
        _validate_registered_model_tag(tag.key, tag.value)
        model_ulid = self._resolve_model_ulid(name)
        self._write_model_tag(model_ulid, tag)

    def delete_registered_model_tag(self, name: str, key: str) -> None:
        """Delete a tag from a registered model."""
        from mlflow.utils.validation import _validate_model_name, _validate_registered_model_tag

        _validate_model_name(name)
        _validate_registered_model_tag(key, "")
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
        if not run_id and model_id:
            from mlflow import MlflowClient

            model = MlflowClient().get_logged_model(model_id)
            run_id = model.source_run_id

        model_ulid = self._resolve_model_ulid(name)
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"

        # Determine next version number
        existing = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
        # Filter out tag items (V#00000001#TAG#key)
        version_items = [it for it in existing if SK_VERSION_TAG_SUFFIX not in it["SK"]]
        next_ver = len(version_items) + 1
        padded = _pad_version(next_ver)

        now_ms = get_current_time_millis()

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
        # Verify version exists (raises on deleted versions)
        self.get_model_version(name, version)
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        now_ms = get_current_time_millis()

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
        # Verify version exists (raises on deleted/missing versions)
        self.get_model_version(name, version)
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
        from mlflow.utils.search_utils import SearchModelVersionUtils

        from mlflow_dynamodbstore.dynamodb.search import (
            FilterPredicate,
            _compare,
        )

        # Validate order_by clauses (raises on invalid columns/syntax)
        for clause in order_by or []:
            SearchModelVersionUtils.parse_order_by_for_search_model_versions(clause)

        if filter_string:
            parsed = SearchModelVersionUtils.parse_search_filter(filter_string)
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
        run_id_pred = next(
            (p for p in predicates if p.field_type == "attribute" and p.key == "run_id"),
            None,
        )

        # Separate prompt tag from other tag predicates — prompt filtering is handled separately
        tag_preds = [p for p in predicates if p.field_type == "tag" and p.key != IS_PROMPT_TAG_KEY]

        from mlflow.store.entities import PagedList
        from mlflow.store.model_registry import (
            SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
            SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD,
        )

        if max_results is None:
            max_results = SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at most "
                f"{SEARCH_MODEL_VERSION_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        # Resolve model name(s) from filter
        model_name: str | None = None
        model_names: list[str] | None = None
        if name_pred and name_pred.op == "=":
            model_name = name_pred.value
        elif name_pred and name_pred.op in ("LIKE", "ILIKE"):
            # For prefix LIKE, find all matching models via GSI5
            pattern = name_pred.value
            if pattern.endswith("%") and "%" not in pattern[:-1]:
                prefix = pattern[:-1]
                items = self._table.query(
                    pk=f"{GSI5_MODEL_NAMES_PREFIX}{self._workspace}",
                    sk_prefix=prefix,
                    index_name="gsi5",
                )
                model_names = [
                    item["name"]
                    for item in items
                    if item.get("name") and not item.get(GSI5_SK, "").startswith("REV#")
                ]
                # Optimize: single match → use single-model path
                if len(model_names) == 1:
                    model_name = model_names[0]
                    model_names = None

        run_id_filter = run_id_pred.value if run_id_pred and run_id_pred.op == "=" else None

        # Build DynamoDB-native filters from attribute predicates
        from boto3.dynamodb.conditions import Attr

        # SK constraints from version_number (used by main-table queries)
        version_pred = next(
            (p for p in predicates if p.field_type == "attribute" and p.key == "version_number"),
            None,
        )
        sk_prefix: str | None = SK_VERSION_PREFIX
        sk_gte: str | None = None
        sk_lte: str | None = None
        if version_pred:
            padded_val = _pad_version(version_pred.value)
            if version_pred.op == "=":
                sk_prefix = f"{SK_VERSION_PREFIX}{padded_val}"
            elif version_pred.op == "<=":
                sk_gte = SK_VERSION_PREFIX
                sk_lte = f"{SK_VERSION_PREFIX}{padded_val}"
                sk_prefix = None
            elif version_pred.op == ">=":
                sk_gte = f"{SK_VERSION_PREFIX}{padded_val}"
                sk_lte = f"{SK_VERSION_PREFIX}99999999"
                sk_prefix = None
            elif version_pred.op == "<":
                sk_gte = SK_VERSION_PREFIX
                sk_lte = f"{SK_VERSION_PREFIX}{_pad_version(int(version_pred.value) - 1)}"
                sk_prefix = None
            elif version_pred.op == ">":
                sk_gte = f"{SK_VERSION_PREFIX}{_pad_version(int(version_pred.value) + 1)}"
                sk_lte = f"{SK_VERSION_PREFIX}99999999"
                sk_prefix = None

        # FilterExpression for source_path, run_id, and version_number (used by LSI2 queries)
        filter_expr: Any = None
        for p in predicates:
            if p.field_type != "attribute" or p.key == "name":
                continue
            # run_id exact match handled by GSI1 path when no model_name
            if p.key == "run_id" and p.op == "=" and not model_name:
                continue
            dynamo_field = _MV_ATTRIBUTE_MAP.get(p.key, p.key)
            val = _pad_version(p.value) if p.key == "version_number" else p.value
            condition: Any = None
            if p.op == "=":
                condition = Attr(dynamo_field).eq(val)
            elif p.op == "!=":
                condition = Attr(dynamo_field).ne(val)
            elif p.op == "<=":
                condition = Attr(dynamo_field).lte(val)
            elif p.op == ">=":
                condition = Attr(dynamo_field).gte(val)
            elif p.op == "<":
                condition = Attr(dynamo_field).lt(val)
            elif p.op == ">":
                condition = Attr(dynamo_field).gt(val)
            elif p.op == "IN":
                condition = Attr(dynamo_field).is_in(val)
            elif p.op in ("LIKE", "ILIKE"):
                pattern = val if isinstance(val, str) else str(val)
                if p.op == "ILIKE":
                    pattern = pattern.lower()
                if pattern.endswith("%") and "%" not in pattern[:-1]:
                    condition = Attr(dynamo_field).begins_with(pattern[:-1])
            if condition is not None:
                filter_expr = filter_expr & condition if filter_expr else condition

        # Build filter_fn for tag and prompt filtering
        def version_filter_fn(mv: ModelVersion) -> bool:
            if tag_preds:
                tag_dict = mv.tags or {}
                for pred in tag_preds:
                    actual = tag_dict.get(pred.key)
                    if not _compare(actual, pred.op, pred.value):
                        return False
            if self._is_querying_prompt(predicates):
                if not self._version_is_prompt(mv):
                    return False
            elif self._version_is_prompt(mv):
                return False
            return True

        if model_name:
            # Single model: use LSI2 for native timestamp DESC ordering + pagination
            # LSI2 can't use SK constraints, so use FilterExpression for all attributes
            try:
                model_ulid = self._resolve_model_ulid(model_name)
            except MlflowException:
                return PagedList([], token=None)
            versions, next_token = self._get_versions_for_model(
                model_ulid,
                model_name,
                max_results=max_results,
                page_token=page_token,
                filter_fn=version_filter_fn,
                filter_expression=filter_expr,
            )
            # Apply run_id post-filter if also specified
            if run_id_filter:
                versions = [v for v in versions if v.run_id == run_id_filter]
            return PagedList(versions, next_token)

        if model_names:
            # Multiple models from LIKE prefix: query each, no pagination
            multi_versions: list[ModelVersion] = []
            for mn in model_names:
                try:
                    ulid = self._resolve_model_ulid(mn)
                except MlflowException:
                    continue
                mvs, _ = self._get_versions_for_model(
                    ulid,
                    mn,
                    filter_fn=version_filter_fn,
                    filter_expression=filter_expr,
                )
                multi_versions.extend(mvs)
            if run_id_filter:
                multi_versions = [v for v in multi_versions if v.run_id == run_id_filter]
            if max_results is not None:
                multi_versions = multi_versions[:max_results]
            return PagedList(multi_versions, token=None)

        # Non-single-model paths: use SK constraints + FilterExpression on main table
        if run_id_filter:
            versions = self._search_versions_by_run_id(run_id_filter)
        else:
            versions = self._list_all_versions(
                sk_prefix=sk_prefix,
                sk_gte=sk_gte,
                sk_lte=sk_lte,
                filter_expression=filter_expr,
            )

        # Apply run_id post-filter if name was also specified
        if run_id_filter and model_name:
            versions = [v for v in versions if v.run_id == run_id_filter]

        # Apply tag filters
        if tag_preds:
            versions = self._filter_versions_by_tags(versions, tag_preds, _compare)

        # Filter by prompt status
        if self._is_querying_prompt(predicates):
            versions = [v for v in versions if self._version_is_prompt(v)]
        else:
            versions = [v for v in versions if not self._version_is_prompt(v)]

        if max_results is not None:
            versions = versions[:max_results]

        return PagedList(versions, token=None)

    def _get_versions_for_model(
        self,
        model_ulid: str,
        model_name: str,
        max_results: int | None = None,
        page_token: str | None = None,
        filter_fn: Any | None = None,
        filter_expression: Any | None = None,
    ) -> tuple[list[ModelVersion], str | None]:
        """Get versions for a model, ordered by last_updated_timestamp DESC via LSI2."""
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"
        exclusive_start_key = self._decode_page_token(page_token)
        index_name = "lsi2"

        if max_results is None:
            max_results = 200_000  # MLflow default threshold

        results: list[ModelVersion] = []
        kept_items: list[dict[str, Any]] = []

        while len(results) < max_results + 1:
            needed = max_results + 1 - len(results)
            has_filters = filter_fn or filter_expression
            batch_size = needed * 2 if has_filters else needed
            items, lek = self._table.query_page(
                pk=pk,
                index_name=index_name,
                limit=batch_size,
                scan_forward=False,
                exclusive_start_key=exclusive_start_key,
                filter_expression=filter_expression,
            )
            for item in items:
                # Skip tag items and non-version items
                sk = item.get("SK", "")
                if not sk.startswith(SK_VERSION_PREFIX) or SK_VERSION_TAG_SUFFIX in sk:
                    continue
                padded = sk.replace(SK_VERSION_PREFIX, "")
                tags = self._get_version_tags(model_ulid, padded)
                aliases = self._aliases_for_model_version(model_ulid, int(padded))
                mv = _item_to_model_version(item, tags, aliases)
                if filter_fn and not filter_fn(mv):
                    continue
                results.append(mv)
                kept_items.append(item)
                if len(results) > max_results:
                    break

            if lek is None or len(results) > max_results:
                break
            exclusive_start_key = lek

        if len(results) > max_results:
            results = results[:max_results]
            kept_items = kept_items[:max_results]
            last_kept = kept_items[-1]
            next_token = self._encode_page_token(
                self._build_exclusive_start_key(last_kept, index_name)
            )
        else:
            next_token = None

        return results, next_token

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

    def _list_all_versions(
        self,
        sk_prefix: str | None = None,
        sk_gte: str | None = None,
        sk_lte: str | None = None,
        filter_expression: Any | None = None,
    ) -> list[ModelVersion]:
        """List all model versions across all models."""
        model_items = self._table.query(
            pk=f"{GSI2_MODELS_PREFIX}{self._workspace}",
            index_name="gsi2",
        )
        if sk_prefix is None and sk_gte is None:
            sk_prefix = SK_VERSION_PREFIX
        versions: list[ModelVersion] = []
        for model_item in model_items:
            model_ulid = model_item["PK"].replace(PK_MODEL_PREFIX, "")
            pk = f"{PK_MODEL_PREFIX}{model_ulid}"
            ver_items = self._table.query(
                pk=pk,
                sk_prefix=sk_prefix,
                sk_gte=sk_gte,
                sk_lte=sk_lte,
                filter_expression=filter_expression,
            )
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

        from mlflow.entities.model_registry.model_version_stages import (
            get_canonical_stage,
        )

        results = []
        for stage in stages:
            canonical = get_canonical_stage(stage)
            # Query LSI3 where lsi3sk begins_with "CanonicalStage#"
            stage_items = self._table.query(
                pk=pk,
                sk_prefix=f"{canonical}#",
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
        from mlflow.utils.validation import (
            _validate_model_name,
            _validate_model_version,
            _validate_model_version_tag,
        )

        _validate_model_name(name)
        _validate_model_version(version)
        _validate_model_version_tag(tag.key, tag.value)
        # Verify version exists (raises on deleted versions)
        self.get_model_version(name, version)
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        self._write_version_tag(model_ulid, padded, tag)

    def delete_model_version_tag(self, name: str, version: str, key: str) -> None:
        """Delete a tag from a model version."""
        from mlflow.utils.validation import (
            _validate_model_name,
            _validate_model_version,
            _validate_model_version_tag,
        )

        _validate_model_name(name)
        _validate_model_version(version)
        _validate_model_version_tag(key, "")
        # Verify version exists (raises on deleted versions)
        self.get_model_version(name, version)
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
        from mlflow.entities.model_registry.model_version_stages import (
            DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS,
            get_canonical_stage,
        )

        canonical_stage = get_canonical_stage(stage)

        is_active = canonical_stage in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        if archive_existing_versions and not is_active:
            raise MlflowException(
                f"Model version transition cannot archive existing model versions "
                f"because '{stage}' is not an Active stage. Valid stages are "
                f"{', '.join(DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS)}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Verify version exists (raises on deleted versions)
        self.get_model_version(name, version)
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)
        now_ms = get_current_time_millis()
        pk = f"{PK_MODEL_PREFIX}{model_ulid}"

        # Archive other versions in active stages if requested
        if archive_existing_versions:
            version_items = self._table.query(pk=pk, sk_prefix=SK_VERSION_PREFIX)
            for vi in version_items:
                vi_padded = vi["SK"].replace(SK_VERSION_PREFIX, "")
                if vi_padded == padded:
                    continue
                vi_stage = vi.get("current_stage", "None")
                if vi_stage == canonical_stage:
                    self._table.update_item(
                        pk=pk,
                        sk=vi["SK"],
                        updates={
                            "current_stage": "Archived",
                            "last_updated_timestamp": now_ms,
                            LSI2_SK: now_ms,
                            LSI3_SK: f"Archived#{vi_padded}",
                        },
                    )

        updated_item = self._table.update_item(
            pk=pk,
            sk=f"{SK_VERSION_PREFIX}{padded}",
            updates={
                "current_stage": canonical_stage,
                "last_updated_timestamp": now_ms,
                LSI2_SK: now_ms,
                LSI3_SK: f"{canonical_stage}#{padded}",
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
