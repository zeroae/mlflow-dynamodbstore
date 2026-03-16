"""DynamoDB-backed MLflow model registry store."""

from __future__ import annotations

import time
from typing import Any

from mlflow.entities.model_registry import RegisteredModel, RegisteredModelTag
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
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


def _item_to_registered_model(
    item: dict[str, Any],
    tags: list[RegisteredModelTag] | None = None,
) -> RegisteredModel:
    """Convert a DynamoDB item to an MLflow RegisteredModel entity."""
    return RegisteredModel(
        name=item["name"],
        creation_timestamp=item.get("creation_timestamp"),
        last_updated_timestamp=item.get("last_updated_timestamp"),
        description=item.get("description", ""),
        tags=tags or [],
    )


def _pad_version(version: str | int) -> str:
    """Pad a version number to 8 digits."""
    return f"{int(version):08d}"


def _item_to_model_version(
    item: dict[str, Any],
    tags: list[ModelVersionTag] | None = None,
) -> ModelVersion:
    """Convert a DynamoDB item to an MLflow ModelVersion entity."""
    return ModelVersion(
        name=item["name"],
        version=str(int(item["version"])),
        creation_timestamp=item.get("creation_timestamp", 0),
        last_updated_timestamp=item.get("last_updated_timestamp"),
        description=item.get("description", ""),
        source=item.get("source", ""),
        run_id=item.get("run_id", ""),
        status=item.get("status", "READY"),
        current_stage=item.get("current_stage", "None"),
        tags=tags or [],
        run_link=item.get("run_link", ""),
    )


class DynamoDBRegistryStore(AbstractStore):
    """MLflow model registry store backed by DynamoDB."""

    def __init__(self, store_uri: str) -> None:
        uri = parse_dynamodb_uri(store_uri)
        ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)
        self._cache = ResolutionCache()
        self._workspace = "default"
        self._config = ConfigReader(self._table)
        self._config.reconcile()

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
                f"Registered Model with name={name} not found.",
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
        # Check uniqueness via GSI3
        existing = self._table.query(
            pk=f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{name}",
            index_name="gsi3",
            limit=1,
        )
        if existing:
            raise MlflowException(
                f"Registered Model '{name}' already exists.",
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
            LSI2_SK: str(now_ms),
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
                f"Registered Model with name={name} not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_model_tags(model_ulid)
        return _item_to_registered_model(item, tags)

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
                f"Registered Model '{new_name}' already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        now_ms = int(time.time() * 1000)

        self._table.update_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
            updates={
                "name": new_name,
                "last_updated_timestamp": now_ms,
                LSI2_SK: str(now_ms),
                LSI3_SK: new_name,
                LSI4_SK: _rev(new_name),
                GSI2_SK: f"{now_ms}#{new_name}",
                GSI3_PK: f"{GSI3_MODEL_NAME_PREFIX}{self._workspace}#{new_name}",
                GSI5_SK: f"{new_name}#{model_ulid}",
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
                LSI2_SK: str(now_ms),
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
        """Search registered models using GSI2."""
        items = self._table.query(
            pk=f"{GSI2_MODELS_PREFIX}{self._workspace}",
            index_name="gsi2",
            limit=max_results,
        )

        models: list[RegisteredModel] = []
        for item in items:
            model_name = item.get("name")
            if model_name:
                model_ulid = item["PK"].replace(PK_MODEL_PREFIX, "")
                tags = self._get_model_tags(model_ulid)
                models.append(_item_to_registered_model(item, tags))

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
            LSI2_SK: str(now_ms),
            LSI3_SK: f"None#{padded}",
            LSI4_SK: (source or "").lower(),
            LSI5_SK: f"{run_id or ''}#{padded}",
        }

        # GSI1: run linkage (only if run_id provided)
        if run_id:
            item[GSI1_PK] = f"{GSI1_RUN_PREFIX}{run_id}"
            item[GSI1_SK] = f"MV#{model_ulid}#{padded}"

        self._table.put_item(item)

        # Update model's last_updated_timestamp
        self._table.update_item(
            pk=pk,
            sk=SK_MODEL_META,
            updates={"last_updated_timestamp": now_ms, LSI2_SK: str(now_ms)},
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
        model_ulid = self._resolve_model_ulid(name)
        padded = _pad_version(version)

        item = self._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        if item is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        tags = self._get_version_tags(model_ulid, padded)
        return _item_to_model_version(item, tags)

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
                LSI2_SK: str(now_ms),
            },
        )

        if updated_item is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found.",
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

    def search_model_versions(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> list[ModelVersion]:
        """Search model versions across all registered models."""
        # Get all models in the workspace
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
                # Skip tag items
                if SK_VERSION_TAG_SUFFIX in vi["SK"]:
                    continue
                padded = vi["SK"].replace(SK_VERSION_PREFIX, "")
                tags = self._get_version_tags(model_ulid, padded)
                versions.append(_item_to_model_version(vi, tags))

        if max_results:
            versions = versions[:max_results]
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
                LSI2_SK: str(now_ms),
                LSI3_SK: f"{stage}#{padded}",
            },
        )

        if updated_item is None:
            raise MlflowException(
                f"Model Version (name={name}, version={version}) not found.",
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
                f"Alias '{alias}' not found for model '{name}'.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        version: str = item["version"]
        return self.get_model_version(name, version)
