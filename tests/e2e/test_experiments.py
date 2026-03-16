"""E2E tests for experiment operations via MLflow client SDK."""

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.e2e


class TestExperiments:
    def test_default_experiment_exists(self, client: MlflowClient):
        """MLflow UI loads default experiment on startup."""
        exp = client.get_experiment("0")
        assert exp.name == "Default"
        assert exp.lifecycle_stage == "active"

    def test_create_experiment(self, client: MlflowClient):
        exp_id = client.create_experiment("e2e-create-exp")
        exp = client.get_experiment(exp_id)
        assert exp.name == "e2e-create-exp"
        assert exp.experiment_id == exp_id

    def test_get_experiment_by_name(self, client: MlflowClient):
        name = "e2e-by-name"
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
        """Exact call the MLflow UI homepage makes — was failing with Decimal error."""
        experiments = client.search_experiments(
            max_results=25,
            order_by=["last_update_time DESC"],
            filter_string="tags.`mlflow.experiment.isGateway` IS NULL",
        )
        assert isinstance(experiments, list)

    def test_search_experiments_by_name(self, client: MlflowClient):
        client.create_experiment("e2e-search-name")
        experiments = client.search_experiments(
            filter_string="name = 'e2e-search-name'",
        )
        assert len(experiments) == 1
        assert experiments[0].name == "e2e-search-name"

    def test_delete_and_restore_experiment(self, client: MlflowClient):
        exp_id = client.create_experiment("e2e-delete-exp")
        client.delete_experiment(exp_id)
        exp = client.get_experiment(exp_id)
        assert exp.lifecycle_stage == "deleted"

        client.restore_experiment(exp_id)
        exp = client.get_experiment(exp_id)
        assert exp.lifecycle_stage == "active"

    def test_set_experiment_tag(self, client: MlflowClient):
        exp_id = client.create_experiment("e2e-tag-exp")
        client.set_experiment_tag(exp_id, "team", "ml-platform")
        exp = client.get_experiment(exp_id)
        assert exp.tags["team"] == "ml-platform"

    def test_rename_experiment(self, client: MlflowClient):
        exp_id = client.create_experiment("e2e-rename-old")
        client.rename_experiment(exp_id, "e2e-rename-new")
        exp = client.get_experiment(exp_id)
        assert exp.name == "e2e-rename-new"
