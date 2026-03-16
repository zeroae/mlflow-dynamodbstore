"""Tests for DynamoDBTrackingStore experiment and run CRUD operations."""

import pytest
from mlflow.entities import (
    Dataset,
    DatasetInput,
    ExperimentTag,
    InputTag,
    Metric,
    Param,
    RunStatus,
    RunTag,
    ViewType,
)


class TestExperimentCRUD:
    def test_create_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        assert exp_id is not None
        assert len(exp_id) == 26  # ULID

    def test_get_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "test-exp"
        assert exp.experiment_id == exp_id
        assert exp.lifecycle_stage == "active"
        assert exp.artifact_location == "s3://bucket"
        assert exp.creation_time is not None
        assert exp.last_update_time is not None

    def test_get_experiment_by_name(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        exp = tracking_store.get_experiment_by_name("test-exp")
        assert exp is not None
        assert exp.experiment_id == exp_id

    def test_create_duplicate_name_raises(self, tracking_store):
        tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        with pytest.raises(Exception):
            tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")

    def test_get_default_experiment(self, tracking_store):
        exp = tracking_store.get_experiment("0")
        assert exp.name == "Default"
        assert exp.experiment_id == "0"

    def test_rename_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("old-name", artifact_location="s3://bucket")
        tracking_store.rename_experiment(exp_id, "new-name")
        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "new-name"
        assert tracking_store.get_experiment_by_name("old-name") is None
        assert tracking_store.get_experiment_by_name("new-name").experiment_id == exp_id

    def test_delete_and_restore_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_id)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "deleted"
        tracking_store.restore_experiment(exp_id)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "active"

    def test_search_experiments_active_only(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)
        results = tracking_store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        names = [e.name for e in results]
        assert "exp-a" in names
        assert "exp-b" not in names

    def test_search_experiments_deleted_only(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)
        results = tracking_store.search_experiments(view_type=ViewType.DELETED_ONLY)
        names = [e.name for e in results]
        assert "exp-a" not in names
        assert "exp-b" in names

    def test_search_experiments_all(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)
        results = tracking_store.search_experiments(view_type=ViewType.ALL)
        names = [e.name for e in results]
        assert "exp-a" in names
        assert "exp-b" in names
        assert "Default" in names

    def test_set_experiment_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tag = ExperimentTag("my-key", "my-value")
        tracking_store.set_experiment_tag(exp_id, tag)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.tags["my-key"] == "my-value"

    def test_create_experiment_with_tags(self, tracking_store):
        tags = [ExperimentTag("k1", "v1"), ExperimentTag("k2", "v2")]
        exp_id = tracking_store.create_experiment(
            "test-exp", artifact_location="s3://bucket", tags=tags
        )
        exp = tracking_store.get_experiment(exp_id)
        assert exp.tags["k1"] == "v1"
        assert exp.tags["k2"] == "v2"

    def test_get_nonexistent_experiment_raises(self, tracking_store):
        with pytest.raises(Exception):
            tracking_store.get_experiment("nonexistent")

    def test_get_experiment_by_name_nonexistent(self, tracking_store):
        result = tracking_store.get_experiment_by_name("nonexistent")
        assert result is None

    def test_search_experiments_max_results(self, tracking_store):
        for i in range(5):
            tracking_store.create_experiment(f"exp-{i}", artifact_location="s3://bucket")
        results = tracking_store.search_experiments(view_type=ViewType.ACTIVE_ONLY, max_results=3)
        # Should return at most 3 (could include Default)
        assert len(results) <= 3


class TestRunCRUD:
    def test_create_run(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200000,
            tags=[],
            run_name="my-run",
        )
        assert run is not None
        assert run.info.experiment_id == exp_id
        assert run.info.user_id == "test-user"
        assert run.info.run_name == "my-run"
        assert run.info.status == "RUNNING"
        assert len(run.info.run_id) == 26  # ULID

    def test_get_run(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        created_run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200000,
            tags=[],
            run_name="my-run",
        )
        fetched = tracking_store.get_run(created_run.info.run_id)
        assert fetched.info.run_id == created_run.info.run_id
        assert fetched.info.experiment_id == exp_id

    def test_update_run_info(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200000,
            tags=[],
            run_name="my-run",
        )
        updated = tracking_store.update_run_info(
            run_id=run.info.run_id,
            run_status=RunStatus.to_string(RunStatus.FINISHED),
            end_time=1709251300000,
            run_name="updated-name",
        )
        assert updated.status == "FINISHED"
        assert updated.end_time == 1709251300000
        assert updated.run_name == "updated-name"

    def test_delete_and_restore_run(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200000,
            tags=[],
            run_name="my-run",
        )
        tracking_store.delete_run(run.info.run_id)
        deleted = tracking_store.get_run(run.info.run_id)
        assert deleted.info.lifecycle_stage == "deleted"

        tracking_store.restore_run(run.info.run_id)
        restored = tracking_store.get_run(run.info.run_id)
        assert restored.info.lifecycle_stage == "active"

    def test_create_run_with_tags(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200000,
            tags=[RunTag("key1", "val1"), RunTag("key2", "val2")],
            run_name="my-run",
        )
        fetched = tracking_store.get_run(run.info.run_id)
        assert "key1" in fetched.data.tags
        assert "key2" in fetched.data.tags
        assert fetched.data.tags["key1"] == "val1"
        assert fetched.data.tags["key2"] == "val2"

    def test_search_runs_basic(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200000,
            tags=[],
            run_name="run-1",
        )
        tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test-user",
            start_time=1709251200001,
            tags=[],
            run_name="run-2",
        )
        results = tracking_store._search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=10,
            order_by=[],
            page_token=None,
        )
        # _search_runs returns a tuple of (runs, next_page_token)
        runs, token = results
        assert len(runs) == 2


class TestMetricsParamsTags:
    def test_log_batch_metrics(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=[
                Metric("loss", 0.5, 1709251200000, 0),
                Metric("loss", 0.3, 1709251200001, 1),
                Metric("accuracy", 0.8, 1709251200000, 0),
            ],
            params=[],
            tags=[],
        )
        fetched = tracking_store.get_run(run.info.run_id)
        # RunData.metrics is a dict {key: value}
        assert "loss" in fetched.data.metrics
        assert "accuracy" in fetched.data.metrics

    def test_log_batch_params(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=[],
            params=[Param("lr", "0.01"), Param("batch_size", "32")],
            tags=[],
        )
        fetched = tracking_store.get_run(run.info.run_id)
        # RunData.params is a dict {key: value}
        assert "lr" in fetched.data.params
        assert "batch_size" in fetched.data.params

    def test_log_batch_tags(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=[],
            params=[],
            tags=[RunTag("mlflow.note", "test note")],
        )
        fetched = tracking_store.get_run(run.info.run_id)
        assert "mlflow.note" in fetched.data.tags

    def test_get_metric_history(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=[
                Metric("loss", 0.5, 1709251200000, 0),
                Metric("loss", 0.3, 1709251200001, 1),
                Metric("loss", 0.1, 1709251200002, 2),
            ],
            params=[],
            tags=[],
        )
        history = tracking_store.get_metric_history(run.info.run_id, "loss")
        assert len(history) == 3
        # Should be ordered by step
        steps = [m.step for m in history]
        assert steps == [0, 1, 2]

    def test_set_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        tracking_store.set_tag(run.info.run_id, RunTag("my_tag", "my_value"))
        fetched = tracking_store.get_run(run.info.run_id)
        assert fetched.data.tags.get("my_tag") == "my_value"

    def test_delete_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[RunTag("to_delete", "value")],
            run_name="run",
        )
        tracking_store.delete_tag(run.info.run_id, "to_delete")
        fetched = tracking_store.get_run(run.info.run_id)
        assert "to_delete" not in fetched.data.tags

    def test_log_batch_with_rank_items(self, tracking_store):
        """Verify RANK materialized items are written."""
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=[Metric("accuracy", 0.95, 1709251200000, 0)],
            params=[Param("lr", "0.01")],
            tags=[],
        )
        # Verify RANK items exist by querying directly
        rank_items = tracking_store._table.query(
            pk=f"EXP#{exp_id}",
            sk_prefix="RANK#",
        )
        assert len(rank_items) >= 2  # one for metric, one for param


class TestDatasetsInputs:
    def test_log_inputs(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        dataset = Dataset(
            name="my-dataset",
            digest="abc123",
            source_type="local",
            source="path/to/data",
        )
        dataset_input = DatasetInput(
            dataset=dataset,
            tags=[InputTag("mlflow.data.context", "training")],
        )
        tracking_store.log_inputs(run.info.run_id, datasets=[dataset_input])

        # Verify dataset item exists
        ds_items = tracking_store._table.query(
            pk=f"EXP#{exp_id}",
            sk_prefix="D#",
        )
        assert len(ds_items) >= 1

    def test_log_inputs_creates_dlink(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="run",
        )
        dataset = Dataset(
            name="my-dataset",
            digest="abc123",
            source_type="local",
            source="path/to/data",
        )
        dataset_input = DatasetInput(
            dataset=dataset,
            tags=[InputTag("mlflow.data.context", "training")],
        )
        tracking_store.log_inputs(run.info.run_id, datasets=[dataset_input])

        # Verify DLINK materialized item
        dlink_items = tracking_store._table.query(
            pk=f"EXP#{exp_id}",
            sk_prefix="DLINK#",
        )
        assert len(dlink_items) == 1
        assert "training" in str(dlink_items[0].get("context", ""))
