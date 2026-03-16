"""Tests for DynamoDBRegistryStore — registered model & version CRUD."""

import pytest
from mlflow.entities.model_registry import RegisteredModelTag
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag


class TestRegisteredModelCRUD:
    def test_create_registered_model(self, registry_store):
        model = registry_store.create_registered_model("my-model")
        assert model.name == "my-model"

    def test_get_registered_model(self, registry_store):
        registry_store.create_registered_model("my-model")
        model = registry_store.get_registered_model("my-model")
        assert model.name == "my-model"

    def test_create_duplicate_name_raises(self, registry_store):
        registry_store.create_registered_model("my-model")
        with pytest.raises(Exception):
            registry_store.create_registered_model("my-model")

    def test_rename_registered_model(self, registry_store):
        registry_store.create_registered_model("old-name")
        model = registry_store.rename_registered_model("old-name", "new-name")
        assert model.name == "new-name"
        # Old name should no longer resolve
        with pytest.raises(Exception):
            registry_store.get_registered_model("old-name")

    def test_update_registered_model(self, registry_store):
        registry_store.create_registered_model("my-model")
        model = registry_store.update_registered_model("my-model", description="A great model")
        assert model.description == "A great model"

    def test_delete_registered_model(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.delete_registered_model("my-model")
        with pytest.raises(Exception):
            registry_store.get_registered_model("my-model")

    def test_search_registered_models(self, registry_store):
        registry_store.create_registered_model("model-a")
        registry_store.create_registered_model("model-b")
        results = registry_store.search_registered_models()
        names = [m.name for m in results]
        assert "model-a" in names
        assert "model-b" in names

    def test_set_registered_model_tag(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag("my-model", RegisteredModelTag("env", "prod"))
        model = registry_store.get_registered_model("my-model")
        assert model.tags.get("env") == "prod" or any(t.key == "env" for t in model.tags)

    def test_delete_registered_model_tag(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.set_registered_model_tag("my-model", RegisteredModelTag("env", "prod"))
        registry_store.delete_registered_model_tag("my-model", "env")
        model = registry_store.get_registered_model("my-model")
        # Tag should be gone
        if isinstance(model.tags, dict):
            assert "env" not in model.tags
        else:
            assert not any(t.key == "env" for t in model.tags)


class TestModelVersionCRUD:
    def test_create_model_version(self, registry_store):
        registry_store.create_registered_model("my-model")
        mv = registry_store.create_model_version(
            name="my-model",
            source="s3://bucket/model",
            run_id="01JRABC",
        )
        assert mv.version == "1"
        assert mv.name == "my-model"

    def test_create_sequential_versions(self, registry_store):
        registry_store.create_registered_model("my-model")
        mv1 = registry_store.create_model_version("my-model", source="s3://bucket/v1")
        mv2 = registry_store.create_model_version("my-model", source="s3://bucket/v2")
        assert mv1.version == "1"
        assert mv2.version == "2"

    def test_get_model_version(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        mv = registry_store.get_model_version("my-model", "1")
        assert mv.name == "my-model"
        assert mv.version == "1"
        assert mv.source == "s3://bucket/model"

    def test_update_model_version(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        mv = registry_store.update_model_version("my-model", "1", description="Best model")
        assert mv.description == "Best model"

    def test_delete_model_version(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        registry_store.delete_model_version("my-model", "1")
        with pytest.raises(Exception):
            registry_store.get_model_version("my-model", "1")

    def test_transition_model_version_stage(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        mv = registry_store.transition_model_version_stage("my-model", "1", "Production", False)
        assert mv.current_stage == "Production"

    def test_get_latest_versions(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/v1")
        registry_store.create_model_version("my-model", source="s3://bucket/v2")
        registry_store.transition_model_version_stage("my-model", "1", "Production", False)
        latest = registry_store.get_latest_versions("my-model", stages=["Production"])
        assert len(latest) == 1
        assert latest[0].version == "1"

    def test_get_latest_versions_no_stage_filter(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/v1")
        registry_store.create_model_version("my-model", source="s3://bucket/v2")
        latest = registry_store.get_latest_versions("my-model")
        # Both are in "None" stage, so only one entry for that stage (the latest)
        assert len(latest) >= 1

    def test_search_model_versions(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/v1")
        registry_store.create_model_version("my-model", source="s3://bucket/v2")
        results = registry_store.search_model_versions()
        assert len(results) == 2

    def test_set_model_version_tag(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        registry_store.set_model_version_tag("my-model", "1", ModelVersionTag("env", "prod"))
        mv = registry_store.get_model_version("my-model", "1")
        # Tags might be dict or list depending on MLflow version
        if isinstance(mv.tags, dict):
            assert mv.tags.get("env") == "prod"
        else:
            assert any(t.key == "env" for t in mv.tags)

    def test_delete_model_version_tag(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        registry_store.set_model_version_tag("my-model", "1", ModelVersionTag("env", "prod"))
        registry_store.delete_model_version_tag("my-model", "1", "env")
        mv = registry_store.get_model_version("my-model", "1")
        if isinstance(mv.tags, dict):
            assert "env" not in mv.tags
        else:
            assert not any(t.key == "env" for t in mv.tags)

    def test_get_model_version_download_uri(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        uri = registry_store.get_model_version_download_uri("my-model", "1")
        assert uri == "s3://bucket/model"


class TestModelAliases:
    def test_set_registered_model_alias(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        registry_store.set_registered_model_alias("my-model", "champion", "1")
        mv = registry_store.get_model_version_by_alias("my-model", "champion")
        assert mv.version == "1"

    def test_delete_registered_model_alias(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/model")
        registry_store.set_registered_model_alias("my-model", "champion", "1")
        registry_store.delete_registered_model_alias("my-model", "champion")
        with pytest.raises(Exception):
            registry_store.get_model_version_by_alias("my-model", "champion")

    def test_update_alias_to_new_version(self, registry_store):
        registry_store.create_registered_model("my-model")
        registry_store.create_model_version("my-model", source="s3://bucket/v1")
        registry_store.create_model_version("my-model", source="s3://bucket/v2")
        registry_store.set_registered_model_alias("my-model", "champion", "1")
        registry_store.set_registered_model_alias("my-model", "champion", "2")
        mv = registry_store.get_model_version_by_alias("my-model", "champion")
        assert mv.version == "2"

    def test_get_alias_not_found_raises(self, registry_store):
        registry_store.create_registered_model("my-model")
        with pytest.raises(Exception):
            registry_store.get_model_version_by_alias("my-model", "nonexistent")

    def test_delete_alias_not_found_does_not_raise(self, registry_store):
        registry_store.create_registered_model("my-model")
        # Deleting a non-existent alias should succeed silently
        registry_store.delete_registered_model_alias("my-model", "nonexistent")
