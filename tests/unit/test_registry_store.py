"""Tests for DynamoDBRegistryStore — registered model CRUD."""

import pytest
from mlflow.entities.model_registry import RegisteredModelTag


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
