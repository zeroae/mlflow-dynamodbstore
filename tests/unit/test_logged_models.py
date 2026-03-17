"""Tests for LoggedModel CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import json

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
    SK_RANK_LM_PREFIX,
    SK_RANK_LMD_PREFIX,
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


class TestFinalizeLoggedModel:
    def test_finalize_ready(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="fin-test")

        result = tracking_store.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
        assert result.status == LoggedModelStatus.READY

        fetched = tracking_store.get_logged_model(model.model_id)
        assert fetched.status == LoggedModelStatus.READY

    def test_finalize_failed(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="fail-test")

        result = tracking_store.finalize_logged_model(model.model_id, LoggedModelStatus.FAILED)
        assert result.status == LoggedModelStatus.FAILED

    def test_finalize_updates_lsi3(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="lsi3-test")
        tracking_store.finalize_logged_model(model.model_id, LoggedModelStatus.READY)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta[LSI3_SK].startswith("READY#")


class TestGetDeletedLoggedModel:
    def test_get_deleted_raises_by_default(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="del-test")
        tracking_store.delete_logged_model(model.model_id)

        with pytest.raises(MlflowException):
            tracking_store.get_logged_model(model.model_id)

    def test_get_deleted_with_allow_deleted(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="del-test")
        tracking_store.delete_logged_model(model.model_id)

        fetched = tracking_store.get_logged_model(model.model_id, allow_deleted=True)
        assert fetched.model_id == model.model_id


class TestDeleteLoggedModel:
    def test_soft_delete(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="del-test")
        tracking_store.delete_logged_model(model.model_id)

        with pytest.raises(MlflowException):
            tracking_store.get_logged_model(model.model_id)

    def test_soft_delete_sets_ttl_on_children(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="ttl-test",
            tags=[LoggedModelTag("k", "v")],
            params=[LoggedModelParameter("p", "1")],
        )
        tracking_store.delete_logged_model(model.model_id)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta.get("lifecycle_stage") == "deleted"
        assert meta[LSI1_SK].startswith("deleted#")


class TestLoggedModelTags:
    def test_set_tags(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="tag-test")

        tracking_store.set_logged_model_tags(model.model_id, [LoggedModelTag("env", "prod")])

        fetched = tracking_store.get_logged_model(model.model_id)
        assert fetched.tags["env"] == "prod"

    def test_set_tags_overwrite(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="overwrite-test",
            tags=[LoggedModelTag("env", "dev")],
        )
        tracking_store.set_logged_model_tags(model.model_id, [LoggedModelTag("env", "prod")])

        fetched = tracking_store.get_logged_model(model.model_id)
        assert fetched.tags["env"] == "prod"

    def test_delete_tag(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="deltag-test",
            tags=[LoggedModelTag("env", "prod")],
        )
        tracking_store.delete_logged_model_tag(model.model_id, "env")

        fetched = tracking_store.get_logged_model(model.model_id)
        assert "env" not in fetched.tags


class TestRecordLoggedModel:
    def test_record_logged_model(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, run_name="test-run", tags=[]
        )

        tracking_store.record_logged_model(
            run.info.run_id, {"model_id": "m-test", "name": "my-model"}
        )

        fetched_run = tracking_store.get_run(run.info.run_id)
        logged_models_tag = fetched_run.data.tags.get("mlflow.loggedModels")
        assert logged_models_tag is not None
        models = json.loads(logged_models_tag)
        assert len(models) == 1
        assert models[0]["model_id"] == "m-test"


class TestLogLoggedModelMetric:
    def test_log_metric(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="metric-test")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
        )

        fetched = tracking_store.get_logged_model(model.model_id)
        assert len(fetched.metrics) == 1
        assert fetched.metrics[0].key == "accuracy"
        assert fetched.metrics[0].value == 0.95

    def test_log_metric_writes_rank_item(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="rank-test")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(pk=pk, sk_prefix=f"{SK_RANK_LM_PREFIX}accuracy#")
        assert len(rank_items) == 1
        assert rank_items[0]["model_id"] == model.model_id

    def test_log_metric_with_dataset_writes_scoped_rank(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="ds-rank")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.90,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
            dataset_name="eval_set",
            dataset_digest="abc123",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_RANK_LMD_PREFIX}accuracy#eval_set#abc123#"
        )
        assert len(rank_items) == 1

    def test_log_metric_replaces_rank_on_update(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="replace-rank")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.80,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
        )
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=2000,
            metric_step=1,
            run_id="run-abc",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(pk=pk, sk_prefix=f"{SK_RANK_LM_PREFIX}accuracy#")
        assert len(rank_items) == 1  # Old deleted, new written
