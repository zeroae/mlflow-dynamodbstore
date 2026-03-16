"""Integration tests for search_registered_models and search_model_versions via moto HTTP server."""

import pytest


@pytest.mark.integration
class TestSearchModelsIntegration:
    def test_search_no_filter(self, registry_store):
        registry_store.create_registered_model("model-1")
        registry_store.create_registered_model("model-2")
        models = registry_store.search_registered_models()
        assert len(models) >= 2

    def test_search_by_name_equals(self, registry_store):
        registry_store.create_registered_model("target-model")
        registry_store.create_registered_model("other-model")
        models = registry_store.search_registered_models(filter_string="name = 'target-model'")
        assert len(models) == 1
        assert models[0].name == "target-model"

    def test_search_by_name_like(self, registry_store):
        registry_store.create_registered_model("prod-pipeline-model")
        registry_store.create_registered_model("dev-pipeline-model")
        registry_store.create_registered_model("other-model")
        models = registry_store.search_registered_models(filter_string="name LIKE 'prod%'")
        assert len(models) == 1
        assert models[0].name == "prod-pipeline-model"

    def test_search_by_name_contains(self, registry_store):
        registry_store.create_registered_model("my-pipeline-model")
        registry_store.create_registered_model("other-job")
        models = registry_store.search_registered_models(filter_string="name LIKE '%pipeline%'")
        assert len(models) == 1

    def test_search_order_by_name(self, registry_store):
        registry_store.create_registered_model("zebra")
        registry_store.create_registered_model("alpha")
        models = registry_store.search_registered_models(order_by=["name ASC"])
        names = [m.name for m in models]
        assert names == sorted(names)

    def test_search_versions_by_model_name(self, registry_store):
        registry_store.create_registered_model("model-a")
        registry_store.create_model_version("model-a", source="s3://a")
        registry_store.create_registered_model("model-b")
        registry_store.create_model_version("model-b", source="s3://b")
        versions = registry_store.search_model_versions(filter_string="name = 'model-a'")
        assert len(versions) == 1
        assert versions[0].name == "model-a"
