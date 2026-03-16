"""Tests for TTL on soft-delete (delete_run + restore_run) and trace lifecycle."""

import time

from mlflow.entities import Metric, TraceInfo, TraceLocation, TraceLocationType, TraceState
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.tracing.constant import TraceTagKey

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
    SK_TRACE_PREFIX,
)


class TestRunTTL:
    def test_delete_run_sets_ttl_on_meta(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_run(run.info.run_id)
        item = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}", f"{SK_RUN_PREFIX}{run.info.run_id}"
        )
        assert "ttl" in item
        assert item["ttl"] > time.time()

    def test_delete_run_sets_ttl_on_children(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[]
        )
        tracking_store.delete_run(run.info.run_id)
        children = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_RUN_PREFIX}{run.info.run_id}#",
        )
        assert len(children) > 0
        for child in children:
            assert "ttl" in child, f"Missing ttl on child SK={child['SK']}"

    def test_restore_run_removes_ttl(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[]
        )
        tracking_store.delete_run(run.info.run_id)
        tracking_store.restore_run(run.info.run_id)
        item = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}", f"{SK_RUN_PREFIX}{run.info.run_id}"
        )
        assert "ttl" not in item
        children = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_RUN_PREFIX}{run.info.run_id}#",
        )
        for child in children:
            assert "ttl" not in child, f"ttl still on child SK={child['SK']}"

    def test_rank_items_get_ttl_on_delete(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[]
        )
        tracking_store.delete_run(run.info.run_id)
        rank_items = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_RANK_PREFIX}m#acc#",
        )
        assert len(rank_items) == 1
        assert "ttl" in rank_items[0]

    def test_fts_items_get_ttl(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline"
        )
        tracking_store.delete_run(run.info.run_id)
        # Check FTS_REV items for this run
        fts_rev_items = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_FTS_REV_PREFIX}{SK_RUN_PREFIX}{run.info.run_id}#",
        )
        for item in fts_rev_items:
            assert "ttl" in item, f"Missing ttl on FTS_REV item SK={item['SK']}"
        # Check FTS items that contain the run_id
        fts_items = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_FTS_PREFIX}",
        )
        for item in fts_items:
            if run.info.run_id in item.get("SK", ""):
                assert "ttl" in item, f"Missing ttl on FTS item SK={item['SK']}"

    def test_restore_run_removes_ttl_from_rank_and_fts(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline"
        )
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("acc", 0.9, 0, 0)], params=[], tags=[]
        )
        tracking_store.delete_run(run.info.run_id)
        tracking_store.restore_run(run.info.run_id)
        # RANK items should not have ttl
        rank_items = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_RANK_PREFIX}m#acc#",
        )
        for item in rank_items:
            assert "ttl" not in item, f"ttl still on RANK item SK={item['SK']}"
        # FTS items should not have ttl
        fts_rev_items = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_FTS_REV_PREFIX}{SK_RUN_PREFIX}{run.info.run_id}#",
        )
        for item in fts_rev_items:
            assert "ttl" not in item, f"ttl still on FTS_REV item SK={item['SK']}"

    def test_ttl_disabled_when_zero(self, tracking_store):
        """When soft_deleted_retention_days=0, no TTL set."""
        table = tracking_store._table
        tracking_store._config.set_ttl_policy(soft_deleted_retention_days=0)
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_run(run.info.run_id)
        item = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}", f"{SK_RUN_PREFIX}{run.info.run_id}"
        )
        assert "ttl" not in item  # disabled


class TestExperimentTTL:
    def test_delete_experiment_sets_ttl_on_meta(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        tracking_store.delete_experiment(exp_id)
        item = table.get_item(f"{PK_EXPERIMENT_PREFIX}{exp_id}", "E#META")
        assert "ttl" in item
        assert item["ttl"] > time.time()

    def test_delete_experiment_does_not_set_ttl_on_children(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.delete_experiment(exp_id)
        run_item = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}", f"{SK_RUN_PREFIX}{run.info.run_id}"
        )
        assert "ttl" not in run_item  # children untouched

    def test_restore_experiment_removes_ttl(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        tracking_store.delete_experiment(exp_id)
        tracking_store.restore_experiment(exp_id)
        item = table.get_item(f"{PK_EXPERIMENT_PREFIX}{exp_id}", "E#META")
        assert "ttl" not in item

    def test_experiment_ttl_disabled_when_zero(self, tracking_store):
        """When soft_deleted_retention_days=0, no TTL set."""
        table = tracking_store._table
        tracking_store._config.set_ttl_policy(soft_deleted_retention_days=0)
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        tracking_store.delete_experiment(exp_id)
        item = table.get_item(f"{PK_EXPERIMENT_PREFIX}{exp_id}", "E#META")
        assert "ttl" not in item


class TestMetricHistoryTTL:
    def test_metric_history_has_ttl(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("loss", 0.5, 1, 100)], params=[], tags=[]
        )
        # History item should have TTL
        history = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_RUN_PREFIX}{run.info.run_id}#MHIST#",
        )
        assert len(history) > 0
        assert "ttl" in history[0]

    def test_metric_latest_no_ttl(self, tracking_store):
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("loss", 0.5, 1, 100)], params=[], tags=[]
        )
        latest = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            f"{SK_RUN_PREFIX}{run.info.run_id}#METRIC#loss",
        )
        assert "ttl" not in latest

    def test_metric_history_ttl_disabled_when_zero(self, tracking_store):
        table = tracking_store._table
        tracking_store._config.set_ttl_policy(metric_history_retention_days=0)
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id, metrics=[Metric("loss", 0.5, 1, 100)], params=[], tags=[]
        )
        history = table.query(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk_prefix=f"{SK_RUN_PREFIX}{run.info.run_id}#MHIST#",
        )
        assert len(history) > 0
        assert "ttl" not in history[0]


def _make_trace_info(experiment_id: str, trace_id: str = "tr-ttl-test") -> TraceInfo:
    """Helper to build a TraceInfo for TTL tests."""
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=1709251200000,
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={TraceTagKey.TRACE_NAME: "my-trace"},
        tags={"user_tag": "hello"},
    )


class TestTraceTTL:
    def test_trace_meta_has_ttl(self, tracking_store):
        """start_trace should set TTL from trace_retention_days on the META item."""
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)
        item = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            f"{SK_TRACE_PREFIX}{trace_info.trace_id}",
        )
        assert "ttl" in item
        assert item["ttl"] > time.time()

    def test_trace_children_inherit_ttl(self, tracking_store):
        """Tags and metadata items should have the same TTL as the trace META."""
        table = tracking_store._table
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        trace_id = trace_info.trace_id

        # Get TTL from META
        meta = table.get_item(pk, f"{SK_TRACE_PREFIX}{trace_id}")
        meta_ttl = meta["ttl"]

        # All children (tags, metadata) should have TTL
        children = table.query(pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#")
        assert len(children) > 0, "Expected trace children (tags, metadata)"
        for child in children:
            assert "ttl" in child, f"Missing ttl on child SK={child['SK']}"
            assert child["ttl"] == meta_ttl, f"TTL mismatch on child SK={child['SK']}"

    def test_trace_ttl_disabled_when_zero(self, tracking_store):
        """When trace_retention_days=0, no TTL should be set."""
        table = tracking_store._table
        tracking_store._config.set_ttl_policy(trace_retention_days=0)
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)
        item = table.get_item(
            f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            f"{SK_TRACE_PREFIX}{trace_info.trace_id}",
        )
        assert "ttl" not in item
