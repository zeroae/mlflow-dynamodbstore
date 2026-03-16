"""DynamoDB-backed MLflow model registry store."""

from __future__ import annotations

import time
from typing import Any

from mlflow.entities.model_registry import RegisteredModel, RegisteredModelTag
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.model_registry.abstract_store import AbstractStore

from mlflow_dynamodbstore.cache import ResolutionCache
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_MODELS_PREFIX,
    GSI2_PK,
    GSI2_SK,
    GSI3_MODEL_NAME_PREFIX,
    GSI3_PK,
    GSI3_SK,
    GSI5_MODEL_NAMES_PREFIX,
    GSI5_PK,
    GSI5_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    PK_MODEL_PREFIX,
    SK_MODEL_META,
    SK_MODEL_TAG_PREFIX,
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


class DynamoDBRegistryStore(AbstractStore):
    """MLflow model registry store backed by DynamoDB."""

    def __init__(self, store_uri: str) -> None:
        uri = parse_dynamodb_uri(store_uri)
        ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)
        self._cache = ResolutionCache()
        self._workspace = "default"

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

    def _write_model_tag(self, model_ulid: str, tag: RegisteredModelTag) -> None:
        """Write a model tag item."""
        item = {
            "PK": f"{PK_MODEL_PREFIX}{model_ulid}",
            "SK": f"{SK_MODEL_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        }
        self._table.put_item(item)

    def _get_model_tags(self, model_ulid: str) -> list[RegisteredModelTag]:
        """Read all tags for a registered model."""
        items = self._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk_prefix=SK_MODEL_TAG_PREFIX,
        )
        return [RegisteredModelTag(item["key"], item["value"]) for item in items]

    # ------------------------------------------------------------------
    # Stub methods -- Model Versions (Task 18)
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
        raise MlflowException("Deferred to Task 18")

    def get_model_version(self, name: str, version: str) -> ModelVersion:
        raise MlflowException("Deferred to Task 18")

    def update_model_version(self, name: str, version: str, description: str) -> ModelVersion:
        raise MlflowException("Deferred to Task 18")

    def delete_model_version(self, name: str, version: str) -> None:
        raise MlflowException("Deferred to Task 18")

    def search_model_versions(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> list[ModelVersion]:
        raise MlflowException("Deferred to Task 18")

    def get_latest_versions(self, name: str, stages: list[str] | None = None) -> list[ModelVersion]:
        raise MlflowException("Deferred to Task 18")

    def set_model_version_tag(self, name: str, version: str, tag: Any) -> None:
        raise MlflowException("Deferred to Task 18")

    def delete_model_version_tag(self, name: str, version: str, key: str) -> None:
        raise MlflowException("Deferred to Task 18")

    def get_model_version_download_uri(self, name: str, version: str) -> str:
        raise MlflowException("Deferred to Task 18")

    def transition_model_version_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool,
    ) -> ModelVersion:
        raise MlflowException("Deferred to Task 18")

    def copy_model_version(self, src_mv: ModelVersion, dst_name: str) -> ModelVersion:
        raise MlflowException("Deferred to Task 18")

    # ------------------------------------------------------------------
    # Stub methods -- Aliases (Task 19)
    # ------------------------------------------------------------------

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        raise MlflowException("Deferred to Task 19")

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        raise MlflowException("Deferred to Task 19")

    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
        raise MlflowException("Deferred to Task 19")
