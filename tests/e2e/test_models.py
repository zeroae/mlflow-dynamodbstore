"""E2E tests for model registry operations via MLflow client SDK."""

import uuid

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestModels:
    def test_create_registered_model(self, client: MlflowClient):
        name = f"e2e-model-{_uid()}"
        model = client.create_registered_model(name)
        assert model.name == name

    def test_search_registered_models(self, client: MlflowClient):
        """Was failing with Decimal → protobuf error."""
        name = f"e2e-search-{_uid()}"
        client.create_registered_model(name)
        models = client.search_registered_models()
        names = [m.name for m in models]
        assert name in names

    def test_get_registered_model(self, client: MlflowClient):
        name = f"e2e-get-{_uid()}"
        client.create_registered_model(name)
        model = client.get_registered_model(name)
        assert model.name == name
        assert model.creation_timestamp is not None

    def test_update_registered_model(self, client: MlflowClient):
        name = f"e2e-update-{_uid()}"
        client.create_registered_model(name)
        client.update_registered_model(name, description="updated")
        model = client.get_registered_model(name)
        assert model.description == "updated"

    def test_create_model_version(self, client: MlflowClient):
        name = f"e2e-ver-{_uid()}"
        client.create_registered_model(name)
        mv = client.create_model_version(name, source="s3://bucket/model")
        assert mv.version == "1"

    def test_get_model_version(self, client: MlflowClient):
        name = f"e2e-getver-{_uid()}"
        client.create_registered_model(name)
        client.create_model_version(name, source="s3://bucket/model")
        mv = client.get_model_version(name, "1")
        assert mv.version == "1"
        assert mv.creation_timestamp is not None

    def test_set_registered_model_alias(self, client: MlflowClient):
        name = f"e2e-alias-{_uid()}"
        client.create_registered_model(name)
        client.create_model_version(name, source="s3://bucket/model")
        client.set_registered_model_alias(name, "champion", "1")
        mv = client.get_model_version_by_alias(name, "champion")
        assert mv.version == "1"

    def test_set_registered_model_tag(self, client: MlflowClient):
        name = f"e2e-tag-{_uid()}"
        client.create_registered_model(name)
        client.set_registered_model_tag(name, "env", "prod")
        model = client.get_registered_model(name)
        assert model.tags["env"] == "prod"

    def test_search_model_versions(self, client: MlflowClient):
        name = f"e2e-searchver-{_uid()}"
        client.create_registered_model(name)
        client.create_model_version(name, source="s3://bucket/model")
        client.create_model_version(name, source="s3://bucket/model-v2")
        versions = client.search_model_versions(filter_string=f"name = '{name}'")
        assert len(versions) == 2

    def test_search_model_versions_returns_all(self, client: MlflowClient):
        name = f"e2e-latest-{_uid()}"
        client.create_registered_model(name)
        client.create_model_version(name, source="s3://bucket/model")
        client.create_model_version(name, source="s3://bucket/model-v2")
        versions = client.search_model_versions(f"name='{name}'")
        assert len(versions) == 2
        version_numbers = {v.version for v in versions}
        assert version_numbers == {"1", "2"}

    def test_search_models_by_name_like(self, client: MlflowClient):
        """Search by name LIKE — uses FTS."""
        uid = _uid()
        name = f"e2e-fts-{uid}-model"
        client.create_registered_model(name)
        models = client.search_registered_models(filter_string=f"name LIKE '%{uid}%'")
        assert any(m.name == name for m in models)

    def test_delete_registered_model(self, client: MlflowClient):
        name = f"e2e-del-{_uid()}"
        client.create_registered_model(name)
        client.delete_registered_model(name)
        with pytest.raises(Exception):
            client.get_registered_model(name)
