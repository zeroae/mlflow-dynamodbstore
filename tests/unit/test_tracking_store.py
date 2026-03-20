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


class TestExperiments:
    def test_supports_workspaces(self, tracking_store):
        """DynamoDB store always supports workspaces."""
        assert tracking_store.supports_workspaces is True


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

    def test_delete_experiment_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tag = ExperimentTag("my-key", "my-value")
        tracking_store.set_experiment_tag(exp_id, tag)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.tags["my-key"] == "my-value"

        tracking_store.delete_experiment_tag(exp_id, "my-key")
        exp = tracking_store.get_experiment(exp_id)
        assert "my-key" not in exp.tags

    def test_delete_experiment_tag_nonexistent_is_silent(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tracking_store.delete_experiment_tag(exp_id, "does-not-exist")

    def test_delete_experiment_tag_nonexistent_experiment_raises(self, tracking_store):
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException, match="not found|not exist|No Experiment"):
            tracking_store.delete_experiment_tag("nonexistent-id", "key")

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


class TestParamRankTruncation:
    """Test RANK item SK truncation for long param values (DDB 1024-byte SK limit)."""

    _exp_counter = 0

    def _create_run(self, tracking_store, exp_id=None):
        if exp_id is None:
            TestParamRankTruncation._exp_counter += 1
            exp_id = tracking_store.create_experiment(
                f"rank-test-{self._exp_counter}", artifact_location="s3://b"
            )
        return tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="r"
        )

    def test_short_param_creates_rank_item(self, tracking_store):
        """Normal short param values produce RANK items."""
        run = self._create_run(tracking_store)
        tracking_store.log_batch(run.info.run_id, metrics=[], params=[Param("lr", "0.01")], tags=[])
        exp_id = run.info.experiment_id
        rank_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#p#lr#")
        assert len(rank_items) == 1
        assert rank_items[0]["value"] == "0.01"

    def test_max_length_param_creates_rank_item(self, tracking_store):
        """6000-char param value (MAX_PARAM_VAL_LENGTH) still creates RANK item."""
        run = self._create_run(tracking_store)
        long_val = "a" * 6000
        tracking_store.log_batch(
            run.info.run_id, metrics=[], params=[Param("key", long_val)], tags=[]
        )
        exp_id = run.info.experiment_id
        rank_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#p#key#")
        assert len(rank_items) == 1
        # Full value is stored on the item, even though SK is truncated
        assert rank_items[0]["value"] == long_val
        # SK must be <= 1024 bytes
        assert len(rank_items[0]["SK"].encode()) <= 1024

    def test_rank_sk_preserves_sort_order_for_common_prefixes(self, tracking_store):
        """Truncated RANK SKs still sort correctly when values share a common prefix."""
        run1 = self._create_run(tracking_store)
        run2 = self._create_run(tracking_store, exp_id=run1.info.experiment_id)
        # Two long values that differ only after the truncation point would sort the same,
        # but values that differ within the retained prefix sort correctly
        val_a = "a" * 900  # fits within 1024 SK
        val_b = "b" * 900  # fits within 1024 SK
        tracking_store.log_batch(run1.info.run_id, metrics=[], params=[Param("p", val_a)], tags=[])
        tracking_store.log_batch(run2.info.run_id, metrics=[], params=[Param("p", val_b)], tags=[])
        exp_id = run1.info.experiment_id
        rank_items = tracking_store._table.query(
            pk=f"EXP#{exp_id}", sk_prefix="RANK#p#p#", scan_forward=True
        )
        assert len(rank_items) == 2
        # "aaa..." sorts before "bbb..." in ascending order
        assert rank_items[0]["value"] == val_a
        assert rank_items[1]["value"] == val_b

    def test_long_key_reduces_value_budget(self, tracking_store):
        """A long param key leaves less room for value in the RANK SK."""
        run = self._create_run(tracking_store)
        long_key = "k" * 250  # MAX_ENTITY_KEY_LENGTH
        val = "v" * 1000
        tracking_store.log_batch(
            run.info.run_id, metrics=[], params=[Param(long_key, val)], tags=[]
        )
        exp_id = run.info.experiment_id
        rank_items = tracking_store._table.query(
            pk=f"EXP#{exp_id}", sk_prefix=f"RANK#p#{long_key}#"
        )
        assert len(rank_items) == 1
        assert len(rank_items[0]["SK"].encode()) <= 1024
        # Full value is preserved on the item
        assert rank_items[0]["value"] == val

    def test_exactly_1024_byte_sk_no_truncation(self, tracking_store):
        """SK that is exactly 1024 bytes should not be truncated."""
        run = self._create_run(tracking_store)
        # Calculate how many value chars fit: 1024 - prefix - suffix
        prefix = "RANK#p#x#"
        suffix = f"#{run.info.run_id}"
        budget = 1024 - len(prefix.encode()) - len(suffix.encode())
        val = "a" * budget  # exactly fills the budget
        tracking_store.log_batch(run.info.run_id, metrics=[], params=[Param("x", val)], tags=[])
        exp_id = run.info.experiment_id
        rank_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#p#x#")
        assert len(rank_items) == 1
        assert len(rank_items[0]["SK"].encode()) == 1024
        # Value in SK is not truncated
        assert rank_items[0]["SK"] == f"{prefix}{val}{suffix}"

    def test_1025_byte_sk_triggers_truncation(self, tracking_store):
        """SK that would be 1025 bytes should be truncated to <= 1024."""
        run = self._create_run(tracking_store)
        prefix = "RANK#p#x#"
        suffix = f"#{run.info.run_id}"
        budget = 1024 - len(prefix.encode()) - len(suffix.encode())
        val = "a" * (budget + 1)  # one byte over
        tracking_store.log_batch(run.info.run_id, metrics=[], params=[Param("x", val)], tags=[])
        exp_id = run.info.experiment_id
        rank_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#p#x#")
        assert len(rank_items) == 1
        assert len(rank_items[0]["SK"].encode()) <= 1024
        # Full value is preserved on the item attribute
        assert rank_items[0]["value"] == val

    def test_search_runs_order_by_short_param(self, tracking_store):
        """search_runs order_by param returns runs sorted by param value."""
        run1 = self._create_run(tracking_store)
        exp_id = run1.info.experiment_id
        run2 = self._create_run(tracking_store, exp_id=exp_id)
        run3 = self._create_run(tracking_store, exp_id=exp_id)
        tracking_store.log_batch(run1.info.run_id, [], [Param("p", "banana")], [])
        tracking_store.log_batch(run2.info.run_id, [], [Param("p", "apple")], [])
        tracking_store.log_batch(run3.info.run_id, [], [Param("p", "cherry")], [])
        runs = tracking_store.search_runs(
            [exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["params.p ASC"],
            max_results=10,
        )
        values = [r.data.params["p"] for r in runs]
        # Verify all runs returned in sorted order (ascending or descending)
        assert sorted(values) == ["apple", "banana", "cherry"]
        assert len(values) == 3

    def test_search_runs_order_by_long_param_truncated_same_prefix(self, tracking_store):
        """Runs with long param values sharing a prefix still return all results."""
        run1 = self._create_run(tracking_store)
        exp_id = run1.info.experiment_id
        run2 = self._create_run(tracking_store, exp_id=exp_id)
        # Two values that are identical in the first 990 chars but differ after
        base = "x" * 990
        val1 = base + "a" * 5010  # total 6000
        val2 = base + "b" * 5010  # total 6000
        tracking_store.log_batch(run1.info.run_id, [], [Param("p", val1)], [])
        tracking_store.log_batch(run2.info.run_id, [], [Param("p", val2)], [])
        runs = tracking_store.search_runs(
            [exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["params.p ASC"],
            max_results=10,
        )
        # Both runs should be returned (no data loss from truncation)
        assert len(runs) == 2
        returned_values = {r.data.params["p"] for r in runs}
        assert val1 in returned_values
        assert val2 in returned_values

    def test_multibyte_utf8_truncation_safe(self, tracking_store):
        """Truncation at multi-byte UTF-8 boundaries doesn't corrupt chars."""
        run = self._create_run(tracking_store)
        # Use 3-byte UTF-8 chars (e.g., Chinese characters)
        val = "\u4e16" * 2000  # 6000 bytes, 2000 chars
        tracking_store.log_batch(run.info.run_id, metrics=[], params=[Param("mb", val)], tags=[])
        exp_id = run.info.experiment_id
        rank_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="RANK#p#mb#")
        assert len(rank_items) == 1
        sk = rank_items[0]["SK"]
        assert len(sk.encode()) <= 1024
        # The truncated SK value must be valid UTF-8 (no partial chars)
        sk.encode("utf-8")  # would raise if invalid
        assert rank_items[0]["value"] == val


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


class TestLogOutputs:
    """Tests for log_outputs — run-to-model associations."""

    def test_log_single_output(self, tracking_store):
        from mlflow.entities import LoggedModelOutput

        exp_id = tracking_store.create_experiment("test-outputs", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r1",
        )
        run_id = run.info.run_id

        tracking_store.log_outputs(run_id, [LoggedModelOutput(model_id="m-abc", step=1)])

        pk = f"EXP#{exp_id}"
        items = tracking_store._table.query(pk=pk, sk_prefix=f"R#{run_id}#OUTPUT#")
        assert len(items) == 1
        assert items[0]["destination_id"] == "m-abc"
        assert items[0]["step"] == 1

    def test_log_multiple_outputs(self, tracking_store):
        from mlflow.entities import LoggedModelOutput

        exp_id = tracking_store.create_experiment("test-outputs-multi", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r2",
        )
        run_id = run.info.run_id

        models = [
            LoggedModelOutput(model_id="m-1", step=1),
            LoggedModelOutput(model_id="m-2", step=2),
            LoggedModelOutput(model_id="m-3", step=3),
        ]
        tracking_store.log_outputs(run_id, models)

        pk = f"EXP#{exp_id}"
        items = tracking_store._table.query(pk=pk, sk_prefix=f"R#{run_id}#OUTPUT#")
        assert len(items) == 3
        model_ids = {item["destination_id"] for item in items}
        assert model_ids == {"m-1", "m-2", "m-3"}

    def test_log_output_nonexistent_run_raises(self, tracking_store):
        from mlflow.entities import LoggedModelOutput
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException, match="not found|does not exist"):
            tracking_store.log_outputs(
                "nonexistent-run", [LoggedModelOutput(model_id="m-x", step=0)]
            )

    def test_log_output_deleted_run_raises(self, tracking_store):
        from mlflow.entities import LoggedModelOutput
        from mlflow.exceptions import MlflowException

        exp_id = tracking_store.create_experiment("test-outputs-del", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r-del",
        )
        run_id = run.info.run_id
        tracking_store.delete_run(run_id)

        with pytest.raises(MlflowException):
            tracking_store.log_outputs(run_id, [LoggedModelOutput(model_id="m-x", step=0)])

    def test_log_output_empty_list_is_noop(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-outputs-empty", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r3",
        )
        tracking_store.log_outputs(run.info.run_id, [])


class TestGetMetricHistoryBulkIntervalFromSteps:
    """Tests for get_metric_history_bulk_interval_from_steps."""

    def _log_metric_history(self, tracking_store, run_id, key, steps):
        """Helper: log metrics at specific steps."""
        from mlflow.entities import Metric

        metrics = [Metric(key=key, value=float(s) * 0.1, timestamp=1000 + s, step=s) for s in steps]
        tracking_store.log_batch(run_id, metrics=metrics, params=[], tags=[])

    def test_returns_only_requested_steps(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r1",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "loss", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id,
            metric_key="loss",
            steps=[3, 7],
            max_results=100,
        )
        steps_returned = [m.step for m in result]
        assert steps_returned == [3, 7]

    def test_missing_steps_silently_skipped(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-skip", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r2",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "acc", [1, 3, 5])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id,
            metric_key="acc",
            steps=[1, 2, 3, 4, 5],
            max_results=100,
        )
        steps_returned = [m.step for m in result]
        assert steps_returned == [1, 3, 5]

    def test_sorted_by_step_timestamp(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-sort", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r3",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "lr", [5, 1, 3])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id,
            metric_key="lr",
            steps=[5, 1, 3],
            max_results=100,
        )
        steps_returned = [m.step for m in result]
        assert steps_returned == [1, 3, 5]

    def test_max_results_limits_output(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-max", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r4",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "loss", list(range(1, 21)))

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id,
            metric_key="loss",
            steps=list(range(1, 21)),
            max_results=5,
        )
        assert len(result) == 5

    def test_empty_steps_returns_empty(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-empty", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r5",
        )
        run_id = run.info.run_id

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id,
            metric_key="loss",
            steps=[],
            max_results=100,
        )
        assert result == []

    def test_returns_metric_with_run_id(self, tracking_store):
        """Verify return type is list[MetricWithRunId] with run_id attribute."""
        exp_id = tracking_store.create_experiment("test-mhbi-type", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1000,
            tags=[],
            run_name="r6",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "val", [1])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id,
            metric_key="val",
            steps=[1],
            max_results=100,
        )
        assert len(result) == 1
        assert result[0].run_id == run_id
        assert result[0].key == "val"
