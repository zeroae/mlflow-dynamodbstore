"""Tests for tag denormalization in DynamoDBRegistryStore."""

from __future__ import annotations

from mlflow.entities.model_registry import RegisteredModelTag
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_MODEL_PREFIX,
    SK_MODEL_META,
    SK_VERSION_PREFIX,
)


class TestRegisteredModelTagDenormalization:
    """Test that set/delete_registered_model_tag denormalizes tags into the META item."""

    def test_set_registered_model_tag_denormalizes_mlflow_tag(self, registry_store):
        """After set_registered_model_tag with mlflow.* key, META item has tags map entry."""
        registry_store.create_registered_model("my-model")
        tag = RegisteredModelTag("mlflow.note", "hello")
        registry_store.set_registered_model_tag("my-model", tag)

        # Read the META item directly from DynamoDB
        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"]["mlflow.note"] == "hello"

    def test_set_registered_model_tag_overwrites_existing(self, registry_store):
        """Setting the same tag twice updates the value in the META item."""
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag("my-model", RegisteredModelTag("mlflow.note", "v1"))
        registry_store.set_registered_model_tag("my-model", RegisteredModelTag("mlflow.note", "v2"))

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item["tags"]["mlflow.note"] == "v2"

    def test_delete_registered_model_tag_removes_denormalized_entry(self, registry_store):
        """delete_registered_model_tag removes the key from the META tags map."""
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag(
            "my-model", RegisteredModelTag("mlflow.note", "hello")
        )
        registry_store.delete_registered_model_tag("my-model", "mlflow.note")

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "mlflow.note" not in tags

    def test_non_matching_tag_not_denormalized(self, registry_store):
        """Tags that don't match mlflow.* patterns are NOT added to META tags map."""
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag(
            "my-model", RegisteredModelTag("custom.key", "value")
        )

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "custom.key" not in tags

    def test_create_registered_model_has_empty_tags_map(self, registry_store):
        """create_registered_model initializes the META item with an empty tags map."""
        registry_store.create_registered_model("my-model")

        model_ulid = registry_store._resolve_model_ulid("my-model")
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_META,
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"] == {}


class TestModelVersionTagDenormalization:
    """Test that set/delete_model_version_tag denormalizes tags into the version item."""

    def test_set_model_version_tag_denormalizes_mlflow_tag(self, registry_store):
        """After set_model_version_tag with mlflow.* key, version item has tags map entry."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        tag = ModelVersionTag("mlflow.note", "hello")
        registry_store.set_model_version_tag("my-model", version, tag)

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"]["mlflow.note"] == "hello"

    def test_set_model_version_tag_overwrites_existing(self, registry_store):
        """Setting the same version tag twice updates the value."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("mlflow.note", "v1")
        )
        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("mlflow.note", "v2")
        )

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item["tags"]["mlflow.note"] == "v2"

    def test_delete_model_version_tag_removes_denormalized_entry(self, registry_store):
        """delete_model_version_tag removes the key from the version item tags map."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("mlflow.note", "hello")
        )
        registry_store.delete_model_version_tag("my-model", version, "mlflow.note")

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "mlflow.note" not in tags

    def test_version_non_matching_tag_not_denormalized(self, registry_store):
        """Version tags that don't match mlflow.* patterns are NOT added to version tags map."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        registry_store.set_model_version_tag(
            "my-model", version, ModelVersionTag("custom.key", "value")
        )

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        tags = item.get("tags", {})
        assert "custom.key" not in tags

    def test_create_model_version_has_empty_tags_map(self, registry_store):
        """create_model_version initializes the version item with an empty tags map."""
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version("my-model", source="s3://bucket/model")
        version = mv.version

        model_ulid = registry_store._resolve_model_ulid("my-model")
        padded = f"{int(version):08d}"
        item = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=f"{SK_VERSION_PREFIX}{padded}",
        )
        assert item is not None
        assert "tags" in item
        assert item["tags"] == {}
