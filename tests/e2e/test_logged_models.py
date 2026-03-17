"""E2E tests for logged model operations via MLflow client SDK."""

import uuid

import pytest
from mlflow import MlflowClient
from mlflow.entities.logged_model_status import LoggedModelStatus

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestLoggedModelsCRUD:
    def test_create_and_get(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-crud-{_uid()}")
        model = client.create_logged_model(experiment_id=exp_id, name="e2e-model")
        assert model.model_id.startswith("m-")
        fetched = client.get_logged_model(model.model_id)
        assert fetched.model_id == model.model_id

    def test_finalize_and_delete(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-fin-{_uid()}")
        model = client.create_logged_model(experiment_id=exp_id, name="fin-model")
        finalized = client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
        assert finalized.status == LoggedModelStatus.READY
        client.delete_logged_model(model.model_id)
        with pytest.raises(Exception):
            client.get_logged_model(model.model_id)

    def test_set_and_delete_tag(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-tag-{_uid()}")
        model = client.create_logged_model(experiment_id=exp_id, name="tag-model")
        client.set_logged_model_tags(model.model_id, {"env": "prod"})
        fetched = client.get_logged_model(model.model_id)
        assert fetched.tags.get("env") == "prod"
        client.delete_logged_model_tag(model.model_id, "env")
        fetched = client.get_logged_model(model.model_id)
        assert "env" not in fetched.tags


class TestLoggedModelsSearch:
    def test_search_empty(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-empty-{_uid()}")
        result = client.search_logged_models(experiment_ids=[exp_id])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_search_returns_created_models(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-search-{_uid()}")
        client.create_logged_model(experiment_id=exp_id, name="model-a")
        client.create_logged_model(experiment_id=exp_id, name="model-b")
        result = client.search_logged_models(experiment_ids=[exp_id])
        assert len(result) == 2

    def test_search_filter_by_name(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-fname-{_uid()}")
        client.create_logged_model(experiment_id=exp_id, name="model-a")
        client.create_logged_model(experiment_id=exp_id, name="model-b")
        result = client.search_logged_models(
            experiment_ids=[exp_id], filter_string="name = 'model-a'"
        )
        assert len(result) == 1
        assert result[0].name == "model-a"

    def test_search_filter_by_status(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-fstat-{_uid()}")
        m1 = client.create_logged_model(experiment_id=exp_id, name="ready-model")
        client.finalize_logged_model(m1.model_id, LoggedModelStatus.READY)
        client.create_logged_model(experiment_id=exp_id, name="pending-model")
        result = client.search_logged_models(
            experiment_ids=[exp_id], filter_string="status = 'READY'"
        )
        assert len(result) == 1
        assert result[0].name == "ready-model"

    def test_search_order_by_creation_time_desc(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-order-{_uid()}")
        client.create_logged_model(experiment_id=exp_id, name="first")
        client.create_logged_model(experiment_id=exp_id, name="second")
        result = client.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        )
        assert len(result) == 2
        assert result[0].creation_timestamp >= result[1].creation_timestamp

    def test_search_across_experiments(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp1 = client.create_experiment(f"e2e-lm-multi1-{_uid()}")
        exp2 = client.create_experiment(f"e2e-lm-multi2-{_uid()}")
        client.create_logged_model(experiment_id=exp1, name="m1")
        client.create_logged_model(experiment_id=exp2, name="m2")
        result = client.search_logged_models(experiment_ids=[exp1, exp2])
        assert len(result) == 2
