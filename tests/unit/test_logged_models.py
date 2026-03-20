"""Tests for LoggedModel CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import json

import pytest
from mlflow.entities import LoggedModelParameter, LoggedModelTag
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList

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
        with pytest.raises(MlflowException, match="not found"):
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


class TestSearchLoggedModels:
    def test_search_empty(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        result = tracking_store.search_logged_models(experiment_ids=[exp_id])
        assert isinstance(result, PagedList)
        assert len(result) == 0

    def test_search_returns_models(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-a")
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-b")
        result = tracking_store.search_logged_models(experiment_ids=[exp_id])
        assert len(result) == 2

    def test_search_excludes_deleted(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="keep")
        m2 = tracking_store.create_logged_model(experiment_id=exp_id, name="delete-me")
        tracking_store.delete_logged_model(m2.model_id)
        result = tracking_store.search_logged_models(experiment_ids=[exp_id])
        assert len(result) == 1
        assert result[0].name == "keep"

    def test_search_filter_by_name(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-a")
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-b")
        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="name = 'model-a'"
        )
        assert len(result) == 1
        assert result[0].name == "model-a"

    def test_search_filter_by_status(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        m1 = tracking_store.create_logged_model(experiment_id=exp_id, name="ready")
        tracking_store.finalize_logged_model(m1.model_id, LoggedModelStatus.READY)
        tracking_store.create_logged_model(experiment_id=exp_id, name="pending")
        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="status = 'READY'"
        )
        assert len(result) == 1
        assert result[0].name == "ready"

    def test_search_order_by_name(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="beta")
        tracking_store.create_logged_model(experiment_id=exp_id, name="alpha")
        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "name", "ascending": True}],
        )
        assert result[0].name == "alpha"
        assert result[1].name == "beta"

    def test_search_across_experiments(self, tracking_store):
        exp_id1 = _create_experiment(tracking_store)
        exp_id2 = tracking_store.create_experiment("test-exp-2", artifact_location="s3://bucket/a2")
        tracking_store.create_logged_model(experiment_id=exp_id1, name="m1")
        tracking_store.create_logged_model(experiment_id=exp_id2, name="m2")
        result = tracking_store.search_logged_models(experiment_ids=[exp_id1, exp_id2])
        assert len(result) == 2

    def test_search_pagination(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        for i in range(5):
            tracking_store.create_logged_model(experiment_id=exp_id, name=f"model-{i}")
        page1 = tracking_store.search_logged_models(experiment_ids=[exp_id], max_results=2)
        assert len(page1) == 2
        assert page1.token is not None
        page2 = tracking_store.search_logged_models(
            experiment_ids=[exp_id], max_results=2, page_token=page1.token
        )
        assert len(page2) == 2

    def test_search_order_by_metric_uses_rank(self, tracking_store):
        """ORDER BY metrics.accuracy DESC uses RANK items (covers _execute_lm_rank)."""
        exp_id = _create_experiment(tracking_store)
        m1 = tracking_store.create_logged_model(experiment_id=exp_id, name="low")
        m2 = tracking_store.create_logged_model(experiment_id=exp_id, name="high")
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=m1.model_id,
            metric_name="accuracy",
            metric_value=0.5,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r1",
        )
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=m2.model_id,
            metric_name="accuracy",
            metric_value=0.9,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r2",
        )

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        )
        assert len(result) == 2
        assert result[0].name == "high"
        assert result[1].name == "low"

    def test_search_filter_by_tag(self, tracking_store):
        """Filter by tag exercises _apply_lm_post_filter tag branch."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="tagged",
            tags=[LoggedModelTag("env", "prod")],
        )
        tracking_store.create_logged_model(experiment_id=exp_id, name="untagged")

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="tags.env = 'prod'"
        )
        assert len(result) == 1
        assert result[0].name == "tagged"

    def test_search_filter_by_param(self, tracking_store):
        """Filter by param exercises _apply_lm_post_filter param branch."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="with-param",
            params=[LoggedModelParameter("lr", "0.01")],
        )
        tracking_store.create_logged_model(experiment_id=exp_id, name="no-param")

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="params.lr = '0.01'"
        )
        assert len(result) == 1
        assert result[0].name == "with-param"

    def test_search_filter_by_metric_value(self, tracking_store):
        """Filter metrics.accuracy > 0.7 uses rank_filters (covers execute rank_filters path)."""
        exp_id = _create_experiment(tracking_store)
        m1 = tracking_store.create_logged_model(experiment_id=exp_id, name="low")
        m2 = tracking_store.create_logged_model(experiment_id=exp_id, name="high")
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=m1.model_id,
            metric_name="accuracy",
            metric_value=0.5,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r1",
        )
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=m2.model_id,
            metric_name="accuracy",
            metric_value=0.9,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r2",
        )

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="metrics.accuracy > 0.7"
        )
        assert len(result) == 1
        assert result[0].name == "high"

    def test_search_order_by_metric_with_dataset(self, tracking_store):
        """ORDER BY metric with dataset scope uses dataset-scoped RANK items."""
        exp_id = _create_experiment(tracking_store)
        m1 = tracking_store.create_logged_model(experiment_id=exp_id, name="low")
        m2 = tracking_store.create_logged_model(experiment_id=exp_id, name="high")
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=m1.model_id,
            metric_name="acc",
            metric_value=0.5,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r1",
            dataset_name="eval",
            dataset_digest="d1",
        )
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=m2.model_id,
            metric_name="acc",
            metric_value=0.9,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r2",
            dataset_name="eval",
            dataset_digest="d1",
        )

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "metrics.acc", "ascending": False}],
            datasets=[{"name": "eval", "digest": "d1"}],
        )
        assert len(result) == 2
        assert result[0].name == "high"

    def test_search_filter_by_creation_timestamp(self, tracking_store):
        """Filter by creation_timestamp exercises timestamp coercion in post_filter."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="old")
        import time

        time.sleep(0.01)
        m2 = tracking_store.create_logged_model(experiment_id=exp_id, name="new")

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id],
            filter_string=f"creation_timestamp >= {m2.creation_timestamp}",
        )
        assert len(result) == 1
        assert result[0].name == "new"


class TestDeleteLoggedModelRankTTL:
    def test_delete_sets_ttl_on_rank_items(self, tracking_store):
        """Soft delete sets TTL on RANK items built from metric sub-items."""
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="rank-ttl")
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.9,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r1",
        )

        tracking_store.delete_logged_model(model.model_id)

        # Verify RANK item got TTL
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(pk=pk, sk_prefix=f"{SK_RANK_LM_PREFIX}accuracy#")
        assert len(rank_items) == 1
        assert "ttl" in rank_items[0]

    def test_delete_sets_ttl_on_dataset_scoped_rank(self, tracking_store):
        """Soft delete also sets TTL on dataset-scoped RANK items."""
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="ds-rank-ttl")
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.9,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r1",
            dataset_name="eval_set",
            dataset_digest="abc123",
        )

        tracking_store.delete_logged_model(model.model_id)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_RANK_LMD_PREFIX}accuracy#eval_set#abc123#"
        )
        assert len(rank_items) == 1
        assert "ttl" in rank_items[0]


class TestLogMetricDatasetScopedReplace:
    def test_replace_dataset_scoped_rank_on_update(self, tracking_store):
        """Update a dataset-scoped metric deletes old RANK and writes new one."""
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="ds-replace")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.7,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="r1",
            dataset_name="eval_set",
            dataset_digest="abc123",
        )
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=2000,
            metric_step=1,
            run_id="r1",
            dataset_name="eval_set",
            dataset_digest="abc123",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_RANK_LMD_PREFIX}accuracy#eval_set#abc123#"
        )
        assert len(rank_items) == 1  # Old deleted, new written
