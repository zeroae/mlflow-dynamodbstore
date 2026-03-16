"""E2E tests for experiment operations via MLflow client SDK."""

import uuid

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestExperiments:
    def test_default_experiment_exists(self, client: MlflowClient):
        """MLflow UI loads default experiment on startup."""
        exp = client.get_experiment("0")
        assert exp.name == "Default"
        assert exp.lifecycle_stage == "active"

    def test_create_experiment(self, client: MlflowClient):
        name = f"e2e-create-{_uid()}"
        exp_id = client.create_experiment(name)
        exp = client.get_experiment(exp_id)
        assert exp.name == name
        assert exp.experiment_id == exp_id

    def test_get_experiment_by_name(self, client: MlflowClient):
        name = f"e2e-byname-{_uid()}"
        exp_id = client.create_experiment(name)
        exp = client.get_experiment_by_name(name)
        assert exp is not None
        assert exp.experiment_id == exp_id

    def test_search_experiments(self, client: MlflowClient):
        """The MLflow UI makes this call on page load."""
        experiments = client.search_experiments(
            max_results=25,
            order_by=["last_update_time DESC"],
        )
        assert len(experiments) >= 1

    def test_search_experiments_ui_homepage(self, client: MlflowClient):
        """Exact call the MLflow UI homepage makes."""
        experiments = client.search_experiments(
            max_results=25,
            order_by=["last_update_time DESC"],
            filter_string="tags.`mlflow.experiment.isGateway` IS NULL",
        )
        assert isinstance(experiments, list)

    def test_search_experiments_by_name(self, client: MlflowClient):
        name = f"e2e-search-{_uid()}"
        client.create_experiment(name)
        experiments = client.search_experiments(
            filter_string=f"name = '{name}'",
        )
        assert len(experiments) == 1
        assert experiments[0].name == name

    def test_delete_and_restore_experiment(self, client: MlflowClient):
        exp_id = client.create_experiment(f"e2e-delete-{_uid()}")
        client.delete_experiment(exp_id)
        exp = client.get_experiment(exp_id)
        assert exp.lifecycle_stage == "deleted"

        client.restore_experiment(exp_id)
        exp = client.get_experiment(exp_id)
        assert exp.lifecycle_stage == "active"

    def test_set_experiment_tag(self, client: MlflowClient):
        exp_id = client.create_experiment(f"e2e-tag-{_uid()}")
        client.set_experiment_tag(exp_id, "team", "ml-platform")
        exp = client.get_experiment(exp_id)
        assert exp.tags["team"] == "ml-platform"

    def test_rename_experiment(self, client: MlflowClient):
        exp_id = client.create_experiment(f"e2e-rename-old-{_uid()}")
        new_name = f"e2e-rename-new-{_uid()}"
        client.rename_experiment(exp_id, new_name)
        exp = client.get_experiment(exp_id)
        assert exp.name == new_name
