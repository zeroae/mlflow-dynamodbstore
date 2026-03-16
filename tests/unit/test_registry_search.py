"""Tests for tag denormalization and FTS writes in DynamoDBRegistryStore."""

from __future__ import annotations

from mlflow.entities.model_registry import RegisteredModelTag
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_MODEL_PREFIX,
    SK_FTS_PREFIX,
    SK_MODEL_META,
    SK_MODEL_NAME_REV,
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


class TestRegistryFTSWrites:
    """Test that create/rename_registered_model writes FTS items."""

    def test_create_model_writes_fts_items(self, registry_store):
        """create_registered_model should produce FTS# items for the model name."""
        registry_store.create_registered_model("my-pipeline-model")
        model_ulid = registry_store._resolve_model_ulid("my-pipeline-model")
        fts_items = registry_store._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}", sk_prefix=SK_FTS_PREFIX
        )
        assert len(fts_items) > 0
        word_items = [i for i in fts_items if i["SK"].startswith("FTS#W#")]
        trigram_items = [i for i in fts_items if i["SK"].startswith("FTS#3#")]
        assert len(word_items) > 0
        assert len(trigram_items) > 0

    def test_create_model_fts_has_gsi2(self, registry_store):
        """FTS forward items for model names must carry gsi2pk for cross-partition search."""
        registry_store.create_registered_model("test-model")
        model_ulid = registry_store._resolve_model_ulid("test-model")
        fts_items = registry_store._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}", sk_prefix=SK_FTS_PREFIX
        )
        assert len(fts_items) > 0
        for item in fts_items:
            assert "gsi2pk" in item

    def test_rename_model_updates_fts(self, registry_store):
        """rename_registered_model should replace old FTS items with new ones."""
        registry_store.create_registered_model("old-model")
        registry_store.rename_registered_model("old-model", "new-model")
        model_ulid = registry_store._resolve_model_ulid("new-model")
        fts_items = registry_store._table.query(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}", sk_prefix=SK_FTS_PREFIX
        )
        sks = [i["SK"] for i in fts_items]
        # Should not have "old" word tokens
        assert not any("old" in sk.lower() for sk in sks if sk.startswith("FTS#W#"))


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


class TestModelNameRev:
    """Test that create/rename_registered_model writes NAME_REV items for suffix ILIKE."""

    def test_create_model_writes_name_rev(self, registry_store):
        """create_registered_model should write a NAME_REV item with reversed lowercase name."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["gsi5sk"].startswith("REV#")

    def test_create_model_name_rev_has_correct_gsi5pk(self, registry_store):
        """NAME_REV item should have gsi5pk set to MODEL_NAMES#<workspace>."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["gsi5pk"] == "MODEL_NAMES#default"

    def test_create_model_name_rev_has_reversed_name(self, registry_store):
        """NAME_REV gsi5sk should contain the reversed lowercase model name."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        # "MyModel" reversed lowercase is "ledomym"
        assert "ledomym" in name_rev["gsi5sk"].lower()

    def test_rename_model_updates_name_rev(self, registry_store):
        """rename_registered_model should update the NAME_REV item with the new reversed name."""
        registry_store.create_registered_model("OldModel")
        registry_store.rename_registered_model("OldModel", "NewModel")
        model_ulid = registry_store._resolve_model_ulid("NewModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        # "NewModel" reversed lowercase is "ledomwen"
        assert "ledomwen" in name_rev["gsi5sk"].lower()
        # "OldModel" reversed lowercase is "ledomdlo" — should NOT be present
        assert "ledomdlo" not in name_rev["gsi5sk"].lower()

    def test_create_model_name_rev_contains_ulid(self, registry_store):
        """NAME_REV gsi5sk should contain the model ULID as a suffix."""
        registry_store.create_registered_model("MyModel")
        model_ulid = registry_store._resolve_model_ulid("MyModel")
        name_rev = registry_store._table.get_item(
            pk=f"{PK_MODEL_PREFIX}{model_ulid}",
            sk=SK_MODEL_NAME_REV,
        )
        assert name_rev is not None
        assert model_ulid in name_rev["gsi5sk"]
