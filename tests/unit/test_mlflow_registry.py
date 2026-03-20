"""Verify DynamoDBRegistryStore compatibility with MLflow's expected behavior.

These tests validate the contract defined by MLflow's AbstractStore for the
model registry — behaviors the MLflow server and client depend on.
"""

import pytest
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from moto import mock_aws


@pytest.fixture
def store():
    with mock_aws():
        from mlflow_dynamodbstore.registry_store import DynamoDBRegistryStore

        s = DynamoDBRegistryStore(store_uri="dynamodb://us-east-1/compat-table")
        yield s


class TestRegistryCompatibility:
    """Tests verifying compatibility with MLflow's model registry store contract."""

    # --- RegisteredModel contract ---

    def test_registered_model_returns_correct_type(self, store):
        """create_registered_model must return a RegisteredModel instance."""
        model = store.create_registered_model("test-model")
        assert isinstance(model, RegisteredModel)

    def test_get_registered_model_returns_correct_type(self, store):
        """get_registered_model must return a RegisteredModel instance."""
        store.create_registered_model("test-model")
        model = store.get_registered_model("test-model")
        assert isinstance(model, RegisteredModel)

    def test_get_nonexistent_model_raises_mlflow_exception(self, store):
        """Fetching a model that does not exist must raise MlflowException."""
        with pytest.raises(MlflowException):
            store.get_registered_model("nonexistent-model")

    def test_create_duplicate_model_raises_mlflow_exception(self, store):
        """Creating a duplicate registered model must raise MlflowException."""
        store.create_registered_model("dup-model")
        with pytest.raises(MlflowException):
            store.create_registered_model("dup-model")

    def test_registered_model_name_matches(self, store):
        """The returned RegisteredModel name must match the requested name."""
        store.create_registered_model("my-model")
        model = store.get_registered_model("my-model")
        assert model.name == "my-model"

    def test_delete_registered_model_removes_it(self, store):
        """After deletion, get_registered_model must raise MlflowException."""
        store.create_registered_model("model-to-delete")
        store.delete_registered_model("model-to-delete")
        with pytest.raises(MlflowException):
            store.get_registered_model("model-to-delete")

    def test_search_registered_models_returns_list(self, store):
        """search_registered_models must return an iterable of RegisteredModel."""
        store.create_registered_model("model-a")
        store.create_registered_model("model-b")
        results = store.search_registered_models()
        assert isinstance(results, list)
        assert all(isinstance(m, RegisteredModel) for m in results)

    def test_search_registered_models_finds_all(self, store):
        """search_registered_models must include all created models."""
        store.create_registered_model("model-x")
        store.create_registered_model("model-y")
        results = store.search_registered_models()
        names = [m.name for m in results]
        assert "model-x" in names
        assert "model-y" in names

    # --- ModelVersion contract ---

    def test_model_version_returns_correct_type(self, store):
        """create_model_version must return a ModelVersion instance."""
        store.create_registered_model("test-model")
        mv = store.create_model_version("test-model", source="s3://bucket")
        assert isinstance(mv, ModelVersion)

    def test_version_numbering_sequential(self, store):
        """Version numbers must be assigned sequentially starting at '1'."""
        store.create_registered_model("test-model")
        v1 = store.create_model_version("test-model", source="s3://v1")
        v2 = store.create_model_version("test-model", source="s3://v2")
        v3 = store.create_model_version("test-model", source="s3://v3")
        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3

    def test_model_version_default_stage(self, store):
        """MLflow expects the default stage to be 'None' (the string)."""
        store.create_registered_model("test-model")
        mv = store.create_model_version("test-model", source="s3://bucket")
        assert mv.current_stage == "None"

    def test_model_version_name_matches_model(self, store):
        """The ModelVersion name must match its parent registered model name."""
        store.create_registered_model("my-model")
        mv = store.create_model_version("my-model", source="s3://bucket")
        assert mv.name == "my-model"

    def test_get_model_version_returns_correct_type(self, store):
        """get_model_version must return a ModelVersion instance."""
        store.create_registered_model("my-model")
        store.create_model_version("my-model", source="s3://bucket")
        mv = store.get_model_version("my-model", "1")
        assert isinstance(mv, ModelVersion)

    def test_get_model_version_source_preserved(self, store):
        """The source URI must be round-tripped correctly."""
        store.create_registered_model("my-model")
        store.create_model_version("my-model", source="s3://bucket/my/path")
        mv = store.get_model_version("my-model", "1")
        assert mv.source == "s3://bucket/my/path"

    def test_delete_model_version_removes_it(self, store):
        """After deletion, get_model_version must raise MlflowException."""
        store.create_registered_model("my-model")
        store.create_model_version("my-model", source="s3://bucket")
        store.delete_model_version("my-model", "1")
        with pytest.raises(MlflowException):
            store.get_model_version("my-model", "1")

    def test_get_model_version_download_uri_matches_source(self, store):
        """get_model_version_download_uri must return the source URI."""
        store.create_registered_model("my-model")
        store.create_model_version("my-model", source="s3://bucket/model")
        uri = store.get_model_version_download_uri("my-model", "1")
        assert uri == "s3://bucket/model"

    def test_transition_model_version_stage(self, store):
        """transition_model_version_stage must update current_stage."""
        store.create_registered_model("my-model")
        store.create_model_version("my-model", source="s3://bucket")
        mv = store.transition_model_version_stage("my-model", "1", "Production", False)
        assert mv.current_stage == "Production"

    def test_search_model_versions_returns_all(self, store):
        """search_model_versions must return all created versions."""
        store.create_registered_model("my-model")
        store.create_model_version("my-model", source="s3://v1")
        store.create_model_version("my-model", source="s3://v2")
        results = store.search_model_versions()
        assert len(results) == 2

    # --- Alias contract ---

    def test_alias_resolves_to_version(self, store):
        """An alias set via set_registered_model_alias must resolve correctly."""
        store.create_registered_model("test-model")
        store.create_model_version("test-model", source="s3://bucket")
        store.set_registered_model_alias("test-model", "prod", "1")
        mv = store.get_model_version_by_alias("test-model", "prod")
        assert mv.version == 1

    def test_alias_returns_model_version_type(self, store):
        """get_model_version_by_alias must return a ModelVersion instance."""
        store.create_registered_model("test-model")
        store.create_model_version("test-model", source="s3://bucket")
        store.set_registered_model_alias("test-model", "prod", "1")
        mv = store.get_model_version_by_alias("test-model", "prod")
        assert isinstance(mv, ModelVersion)

    def test_alias_can_be_updated(self, store):
        """Re-setting an alias must point to the new version."""
        store.create_registered_model("test-model")
        store.create_model_version("test-model", source="s3://v1")
        store.create_model_version("test-model", source="s3://v2")
        store.set_registered_model_alias("test-model", "prod", "1")
        store.set_registered_model_alias("test-model", "prod", "2")
        mv = store.get_model_version_by_alias("test-model", "prod")
        assert mv.version == 2

    def test_get_nonexistent_alias_raises_mlflow_exception(self, store):
        """Resolving a non-existent alias must raise MlflowException."""
        store.create_registered_model("test-model")
        with pytest.raises(MlflowException):
            store.get_model_version_by_alias("test-model", "nonexistent-alias")

    def test_delete_alias_prevents_resolution(self, store):
        """After delete_registered_model_alias, resolution must raise MlflowException."""
        store.create_registered_model("test-model")
        store.create_model_version("test-model", source="s3://bucket")
        store.set_registered_model_alias("test-model", "prod", "1")
        store.delete_registered_model_alias("test-model", "prod")
        with pytest.raises(MlflowException):
            store.get_model_version_by_alias("test-model", "prod")

    # --- Rename registered model contract ---

    def test_rename_registered_model(self, store):
        """rename_registered_model must update the model name."""
        store.create_registered_model("original-model")
        model = store.rename_registered_model("original-model", "renamed-model")
        assert model.name == "renamed-model"

    def test_rename_registered_model_old_name_gone(self, store):
        """After rename, get_registered_model with old name must raise."""
        store.create_registered_model("old-model")
        store.rename_registered_model("old-model", "new-model")
        with pytest.raises(MlflowException):
            store.get_registered_model("old-model")

    def test_rename_registered_model_new_name_works(self, store):
        """After rename, get_registered_model with new name must succeed."""
        store.create_registered_model("before")
        store.rename_registered_model("before", "after")
        model = store.get_registered_model("after")
        assert model.name == "after"

    # --- Update registered model contract ---

    def test_update_registered_model_description(self, store):
        """update_registered_model must update the description."""
        store.create_registered_model("desc-model")
        model = store.update_registered_model("desc-model", "updated description")
        assert model.description == "updated description"

    # --- Registered model tags contract ---

    def test_set_registered_model_tag(self, store):
        """set_registered_model_tag must be visible on get_registered_model."""
        from mlflow.entities.model_registry import RegisteredModelTag

        store.create_registered_model("tagged-model")
        store.set_registered_model_tag("tagged-model", RegisteredModelTag("env", "prod"))
        model = store.get_registered_model("tagged-model")
        # MLflow RegisteredModel.tags is a dict {key: value}
        assert isinstance(model.tags, dict)
        assert model.tags.get("env") == "prod"

    def test_delete_registered_model_tag(self, store):
        """delete_registered_model_tag must remove the tag."""
        from mlflow.entities.model_registry import RegisteredModelTag

        store.create_registered_model("tagged-model")
        store.set_registered_model_tag("tagged-model", RegisteredModelTag("tmp", "val"))
        store.delete_registered_model_tag("tagged-model", "tmp")
        model = store.get_registered_model("tagged-model")
        assert isinstance(model.tags, dict)
        assert "tmp" not in model.tags

    # --- Model version tags contract ---

    def test_set_model_version_tag(self, store):
        """set_model_version_tag must be visible on get_model_version."""
        from mlflow.entities.model_registry.model_version_tag import ModelVersionTag

        store.create_registered_model("mv-tag-model")
        store.create_model_version("mv-tag-model", source="s3://bucket")
        store.set_model_version_tag("mv-tag-model", "1", ModelVersionTag("key1", "val1"))
        mv = store.get_model_version("mv-tag-model", "1")
        # MLflow ModelVersion.tags is a dict {key: value}
        assert isinstance(mv.tags, dict)
        assert mv.tags.get("key1") == "val1"

    def test_delete_model_version_tag(self, store):
        """delete_model_version_tag must remove the tag."""
        from mlflow.entities.model_registry.model_version_tag import ModelVersionTag

        store.create_registered_model("mv-tag-model")
        store.create_model_version("mv-tag-model", source="s3://bucket")
        store.set_model_version_tag("mv-tag-model", "1", ModelVersionTag("tmp", "val"))
        store.delete_model_version_tag("mv-tag-model", "1", "tmp")
        mv = store.get_model_version("mv-tag-model", "1")
        assert isinstance(mv.tags, dict)
        assert "tmp" not in mv.tags

    # --- Update model version contract ---

    def test_update_model_version_description(self, store):
        """update_model_version must update the description."""
        store.create_registered_model("upd-model")
        store.create_model_version("upd-model", source="s3://bucket")
        mv = store.update_model_version("upd-model", "1", "new description")
        assert mv.description == "new description"

    # --- get_latest_versions contract ---

    def test_get_latest_versions_returns_list(self, store):
        """get_latest_versions must return a list of ModelVersion."""
        store.create_registered_model("latest-model")
        store.create_model_version("latest-model", source="s3://v1")
        store.create_model_version("latest-model", source="s3://v2")
        results = store.get_latest_versions("latest-model")
        assert isinstance(results, list)
        assert all(isinstance(mv, ModelVersion) for mv in results)

    def test_get_latest_versions_returns_highest_version(self, store):
        """get_latest_versions must return the highest version per stage."""
        store.create_registered_model("latest-model")
        store.create_model_version("latest-model", source="s3://v1")
        store.create_model_version("latest-model", source="s3://v2")
        results = store.get_latest_versions("latest-model")
        # Both versions are in "None" stage, so we should get just the latest
        assert len(results) == 1
        assert results[0].version == 2

    # --- Get nonexistent model version contract ---

    def test_get_nonexistent_model_version_raises(self, store):
        """Fetching a version that does not exist must raise MlflowException."""
        store.create_registered_model("my-model")
        with pytest.raises(MlflowException):
            store.get_model_version("my-model", "999")

    # --- Copy model version contract ---

    def test_copy_model_version(self, store):
        """copy_model_version must create a new version in the destination model."""
        store.create_registered_model("src-model")
        store.create_registered_model("dst-model")
        v1 = store.create_model_version("src-model", source="s3://original")
        copied = store.copy_model_version(v1, "dst-model")
        assert isinstance(copied, ModelVersion)
        assert copied.name == "dst-model"
        assert copied.source == "models:/src-model/1"
        assert store.get_model_version_download_uri("dst-model", copied.version) == "s3://original"
