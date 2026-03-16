"""Tests for TTL on soft-delete (delete_run + restore_run)."""

import time

from mlflow.entities import Metric

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
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
