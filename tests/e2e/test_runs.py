"""E2E tests for run operations via MLflow client SDK."""

import uuid

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
def experiment_id(client: MlflowClient):
    """Create a fresh experiment for run tests."""
    return client.create_experiment(f"e2e-runs-{_uid()}")


class TestRuns:
    def test_create_run(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        assert run.info.experiment_id == experiment_id
        assert run.info.status == "RUNNING"

    def test_log_metrics(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.log_metric(run.info.run_id, "accuracy", 0.95)
        client.log_metric(run.info.run_id, "loss", 0.05)

        fetched = client.get_run(run.info.run_id)
        # REST API returns metrics as dict {key: value}
        assert fetched.data.metrics["accuracy"] == 0.95
        assert fetched.data.metrics["loss"] == 0.05

    def test_log_params(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.log_param(run.info.run_id, "learning_rate", "0.01")
        client.log_param(run.info.run_id, "epochs", "100")

        fetched = client.get_run(run.info.run_id)
        assert fetched.data.params["learning_rate"] == "0.01"

    def test_set_tag(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.set_tag(run.info.run_id, "model_type", "xgboost")

        fetched = client.get_run(run.info.run_id)
        assert fetched.data.tags["model_type"] == "xgboost"

    def test_log_batch(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.log_batch(
            run.info.run_id,
            metrics=[Metric("m1", 1.0, 0, 0), Metric("m2", 2.0, 0, 0)],
            params=[Param("p1", "v1")],
            tags=[RunTag("t1", "v1")],
        )

        fetched = client.get_run(run.info.run_id)
        assert "m1" in fetched.data.metrics
        assert "m2" in fetched.data.metrics
        assert "p1" in fetched.data.params

    def test_update_run_status(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.set_terminated(run.info.run_id, status="FINISHED")

        fetched = client.get_run(run.info.run_id)
        assert fetched.info.status == "FINISHED"
        assert fetched.info.end_time is not None

    def test_delete_and_restore_run(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.delete_run(run.info.run_id)
        fetched = client.get_run(run.info.run_id)
        assert fetched.info.lifecycle_stage == "deleted"

        client.restore_run(run.info.run_id)
        fetched = client.get_run(run.info.run_id)
        assert fetched.info.lifecycle_stage == "active"

    def test_get_metric_history(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        for i in range(5):
            client.log_metric(run.info.run_id, "loss", 1.0 - i * 0.1, step=i)
        history = client.get_metric_history(run.info.run_id, "loss")
        assert len(history) == 5

    def test_search_runs(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.log_metric(run.info.run_id, "acc", 0.9)
        client.set_terminated(run.info.run_id, "FINISHED")

        runs = client.search_runs(experiment_ids=[experiment_id])
        assert any(r.info.run_id == run.info.run_id for r in runs)

    def test_search_runs_with_filter(self, client: MlflowClient, experiment_id):
        run = client.create_run(experiment_id)
        client.log_param(run.info.run_id, "algo", "xgb")
        client.set_terminated(run.info.run_id, "FINISHED")

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="params.algo = 'xgb'",
        )
        assert any(r.info.run_id == run.info.run_id for r in runs)

    def test_fluent_api(self, mlflow_server, client: MlflowClient):
        """Test the high-level mlflow.* API that most users use."""
        mlflow.set_tracking_uri(mlflow_server)
        mlflow.set_experiment(f"e2e-fluent-{_uid()}")

        with mlflow.start_run() as run:
            mlflow.log_param("lr", "0.01")
            mlflow.log_metric("acc", 0.99)
            mlflow.set_tag("framework", "pytorch")

        fetched = client.get_run(run.info.run_id)
        assert fetched.data.params["lr"] == "0.01"
        assert fetched.data.tags["framework"] == "pytorch"
