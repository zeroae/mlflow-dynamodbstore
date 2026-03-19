"""Integration tests for DynamoDBRegistryStore via moto HTTP server."""


class TestRegistryIntegration:
    def test_full_model_lifecycle(self, registry_store):
        # Create model
        model = registry_store.create_registered_model("test-model")
        assert model.name == "test-model"

        # Create versions
        v1 = registry_store.create_model_version("test-model", source="s3://v1")
        v2 = registry_store.create_model_version("test-model", source="s3://v2")
        assert v1.version == 1
        assert v2.version == 2

        # Set alias
        registry_store.set_registered_model_alias("test-model", "champion", "1")
        mv = registry_store.get_model_version_by_alias("test-model", "champion")
        assert mv.version == 1

        # Rename model
        registry_store.rename_registered_model("test-model", "better-model")
        model = registry_store.get_registered_model("better-model")
        assert model.name == "better-model"

        # Delete
        registry_store.delete_registered_model("better-model")

    def test_search_registered_models_gsi2(self, registry_store):
        registry_store.create_registered_model("model-a")
        registry_store.create_registered_model("model-b")
        results = registry_store.search_registered_models()
        names = [m.name for m in results]
        assert "model-a" in names
        assert "model-b" in names
