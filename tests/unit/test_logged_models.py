"""Tests for LoggedModel CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import pytest
from mlflow.entities import LoggedModelParameter, LoggedModelTag
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.exceptions import MlflowException

from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_LM_PREFIX,
    GSI1_PK,
    GSI1_SK,
    LSI1_SK,
    LSI3_SK,
    LSI4_SK,
    PK_EXPERIMENT_PREFIX,
    SK_LM_PREFIX,
)


def _create_experiment(tracking_store) -> str:
    return tracking_store.create_experiment("test-exp", artifact_location="s3://bucket/artifacts")


class TestCreateLoggedModel:
    def test_create_logged_model(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="my-model")

        assert model.model_id.startswith("m-")
        assert model.experiment_id == exp_id
        assert model.name == "my-model"
        assert model.status == LoggedModelStatus.PENDING
        assert "models/" in model.artifact_location
        assert model.artifact_location.endswith("/artifacts/")

    def test_create_with_tags_and_params(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="tagged-model",
            tags=[LoggedModelTag("env", "prod")],
            params=[LoggedModelParameter("lr", "0.01")],
            model_type="sklearn",
            source_run_id="run-abc",
        )

        assert model.tags == {"env": "prod"}
        assert model.params == {"lr": "0.01"}
        assert model.model_type == "sklearn"
        assert model.source_run_id == "run-abc"

    def test_create_writes_gsi1_reverse_lookup(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="gsi-test")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta[GSI1_PK] == f"{GSI1_LM_PREFIX}{model.model_id}"
        assert meta[GSI1_SK] == f"{PK_EXPERIMENT_PREFIX}{exp_id}"

    def test_create_writes_lsi_projections(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="MyModel")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta[LSI1_SK].startswith("active#")
        assert meta[LSI3_SK].startswith("PENDING#")
        assert meta[LSI4_SK] == "mymodel"


class TestGetLoggedModel:
    def test_get_logged_model(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        created = tracking_store.create_logged_model(experiment_id=exp_id, name="get-test")

        fetched = tracking_store.get_logged_model(created.model_id)
        assert fetched.model_id == created.model_id
        assert fetched.name == "get-test"
        assert fetched.experiment_id == exp_id

    def test_get_with_tags_and_params(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        created = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="full-model",
            tags=[LoggedModelTag("k", "v")],
            params=[LoggedModelParameter("p", "1")],
        )

        fetched = tracking_store.get_logged_model(created.model_id)
        assert fetched.tags == {"k": "v"}
        assert fetched.params == {"p": "1"}

    def test_get_nonexistent_raises(self, tracking_store):
        _create_experiment(tracking_store)
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_logged_model("m-nonexistent")
