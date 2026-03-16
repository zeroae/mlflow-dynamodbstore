"""E2E tests for model registry operations via MLflow client SDK."""

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.e2e


class TestModels:
    def test_create_registered_model(self, client: MlflowClient):
        model = client.create_registered_model("e2e-model")
        assert model.name == "e2e-model"

    def test_search_registered_models(self, client: MlflowClient):
        """Was failing with Decimal → protobuf error."""
        client.create_registered_model("e2e-search-model")
        models = client.search_registered_models()
        names = [m.name for m in models]
        assert "e2e-search-model" in names

    def test_get_registered_model(self, client: MlflowClient):
        client.create_registered_model("e2e-get-model")
        model = client.get_registered_model("e2e-get-model")
        assert model.name == "e2e-get-model"
        assert model.creation_timestamp is not None

    def test_update_registered_model(self, client: MlflowClient):
        client.create_registered_model("e2e-update-model")
        client.update_registered_model("e2e-update-model", description="updated")
        model = client.get_registered_model("e2e-update-model")
        assert model.description == "updated"

    def test_create_model_version(self, client: MlflowClient):
        client.create_registered_model("e2e-version-model")
        mv = client.create_model_version("e2e-version-model", source="s3://bucket/model")
        assert mv.version == "1"

    def test_get_model_version(self, client: MlflowClient):
        client.create_registered_model("e2e-getver-model")
        client.create_model_version("e2e-getver-model", source="s3://bucket/model")
        mv = client.get_model_version("e2e-getver-model", "1")
        assert mv.version == "1"
        assert mv.creation_timestamp is not None

    def test_set_registered_model_alias(self, client: MlflowClient):
        client.create_registered_model("e2e-alias-model")
        client.create_model_version("e2e-alias-model", source="s3://b")
        client.set_registered_model_alias("e2e-alias-model", "champion", "1")
        mv = client.get_model_version_by_alias("e2e-alias-model", "champion")
        assert mv.version == "1"

    def test_set_registered_model_tag(self, client: MlflowClient):
        client.create_registered_model("e2e-modeltag")
        client.set_registered_model_tag("e2e-modeltag", "env", "prod")
        model = client.get_registered_model("e2e-modeltag")
        assert model.tags["env"] == "prod"

    def test_delete_registered_model(self, client: MlflowClient):
        client.create_registered_model("e2e-delete-model")
        client.delete_registered_model("e2e-delete-model")
        with pytest.raises(Exception):
            client.get_registered_model("e2e-delete-model")
