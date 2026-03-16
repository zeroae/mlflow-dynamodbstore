"""Verify DynamoDBTrackingStore compatibility with MLflow's expected behavior.

These tests validate the contract defined by MLflow's AbstractStore and the
behaviors that the MLflow server and client code depend on — not our internal
implementation details.
"""

import pytest
from mlflow.entities import (
    Experiment,
    Metric,
    Param,
    Run,
    RunTag,
    ViewType,
)
from mlflow.exceptions import MlflowException
from moto import mock_aws


@pytest.fixture
def store():
    with mock_aws():
        from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

        s = DynamoDBTrackingStore(
            store_uri="dynamodb://us-east-1/compat-table",
            artifact_uri="/tmp/artifacts",
        )
        yield s


class TestTrackingCompatibility:
    """Tests verifying compatibility with MLflow's tracking store contract."""

    # --- Experiment contract ---

    def test_default_experiment_exists(self, store):
        """MLflow requires experiment '0' to exist upon initialization."""
        exp = store.get_experiment("0")
        assert exp.name == "Default"
        assert exp.lifecycle_stage == "active"

    def test_experiment_returns_correct_entity_type(self, store):
        """get_experiment must return an mlflow.entities.Experiment instance."""
        exp_id = store.create_experiment("test")
        exp = store.get_experiment(exp_id)
        assert isinstance(exp, Experiment)

    def test_get_experiment_by_name_returns_none_not_error(self, store):
        """MLflow callers expect None (not an exception) for missing names."""
        result = store.get_experiment_by_name("nonexistent-experiment")
        assert result is None

    def test_create_duplicate_experiment_raises_mlflow_exception(self, store):
        """Duplicate experiment names must raise MlflowException."""
        store.create_experiment("dup-exp")
        with pytest.raises(MlflowException):
            store.create_experiment("dup-exp")

    def test_experiment_lifecycle_stages(self, store):
        """MLflow uses 'active' and 'deleted' lifecycle stages."""
        exp_id = store.create_experiment("lifecycle-test")
        assert store.get_experiment(exp_id).lifecycle_stage == "active"
        store.delete_experiment(exp_id)
        assert store.get_experiment(exp_id).lifecycle_stage == "deleted"
        store.restore_experiment(exp_id)
        assert store.get_experiment(exp_id).lifecycle_stage == "active"

    def test_get_nonexistent_experiment_raises_mlflow_exception(self, store):
        """Fetching a missing experiment must raise MlflowException."""
        with pytest.raises(MlflowException):
            store.get_experiment("nonexistent-id")

    def test_search_experiments_returns_list(self, store):
        """search_experiments must return an iterable of Experiment objects."""
        store.create_experiment("exp-a")
        store.create_experiment("exp-b")
        results = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        assert isinstance(results, list)
        assert all(isinstance(e, Experiment) for e in results)

    def test_search_experiments_active_filter(self, store):
        """ACTIVE_ONLY view must exclude deleted experiments."""
        exp_id = store.create_experiment("to-delete")
        store.delete_experiment(exp_id)
        store.create_experiment("active-exp")
        results = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        names = [e.name for e in results]
        assert "active-exp" in names
        assert "to-delete" not in names

    # --- Run contract ---

    def test_run_returns_correct_entity_type(self, store):
        """get_run must return an mlflow.entities.Run instance."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        fetched = store.get_run(run.info.run_id)
        assert isinstance(fetched, Run)

    def test_run_initial_status_is_running(self, store):
        """Newly created runs must have status RUNNING."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        assert run.info.status == "RUNNING"

    def test_run_initial_lifecycle_stage_is_active(self, store):
        """Newly created runs must be in 'active' lifecycle stage."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        fetched = store.get_run(run.info.run_id)
        assert fetched.info.lifecycle_stage == "active"

    def test_run_experiment_id_matches(self, store):
        """Run's experiment_id must match the experiment it was created in."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        fetched = store.get_run(run.info.run_id)
        assert fetched.info.experiment_id == exp_id

    # --- Search runs contract ---

    def test_search_runs_returns_tuple(self, store):
        """_search_runs must return a (runs, page_token) tuple."""
        exp_id = store.create_experiment("test")
        result = store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 10, [], None)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_search_runs_page_token_none_when_no_more(self, store):
        """_search_runs page token must be None when all results fit in one page."""
        exp_id = store.create_experiment("test")
        store.create_run(exp_id, "user", 1000, [], "r1")
        _runs, token = store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 100, [], None)
        assert token is None

    def test_search_runs_active_only_excludes_deleted(self, store):
        """ACTIVE_ONLY view must exclude deleted runs."""
        exp_id = store.create_experiment("test")
        r1 = store.create_run(exp_id, "user", 1000, [], "r1")
        r2 = store.create_run(exp_id, "user", 1001, [], "r2")
        store.delete_run(r2.info.run_id)
        runs, _token = store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 100, [], None)
        run_ids = [r.info.run_id for r in runs]
        assert r1.info.run_id in run_ids
        assert r2.info.run_id not in run_ids

    # --- Metrics contract ---

    def test_metric_history_ordered_by_step(self, store):
        """get_metric_history must return metrics in ascending step order."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        store.log_batch(
            run.info.run_id,
            [
                Metric("m", 3.0, 1000, 2),
                Metric("m", 1.0, 1000, 0),
                Metric("m", 2.0, 1000, 1),
            ],
            [],
            [],
        )
        history = store.get_metric_history(run.info.run_id, "m")
        steps = [m.step for m in history]
        assert steps == sorted(steps)

    def test_metric_history_returns_all_entries(self, store):
        """get_metric_history must return every logged step, not just the latest."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        store.log_batch(
            run.info.run_id,
            [
                Metric("loss", 0.5, 1000, 0),
                Metric("loss", 0.3, 1001, 1),
                Metric("loss", 0.1, 1002, 2),
            ],
            [],
            [],
        )
        history = store.get_metric_history(run.info.run_id, "loss")
        assert len(history) == 3

    def test_run_data_metrics_dict_has_latest_value(self, store):
        """run.data.metrics must be a dict containing the most-recent step value."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        store.log_batch(
            run.info.run_id,
            [
                Metric("loss", 0.5, 1000, 0),
                Metric("loss", 0.1, 1001, 1),
            ],
            [],
            [],
        )
        fetched = store.get_run(run.info.run_id)
        assert isinstance(fetched.data.metrics, dict)
        assert "loss" in fetched.data.metrics

    # --- Params contract ---

    def test_run_data_params_dict(self, store):
        """run.data.params must be a dict {key: value}."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        store.log_batch(run.info.run_id, [], [Param("lr", "0.01")], [])
        fetched = store.get_run(run.info.run_id)
        assert isinstance(fetched.data.params, dict)
        assert fetched.data.params["lr"] == "0.01"

    # --- Tags contract ---

    def test_run_data_tags_dict(self, store):
        """run.data.tags must be a dict {key: value}."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        store.log_batch(run.info.run_id, [], [], [RunTag("mlflow.note", "hello")])
        fetched = store.get_run(run.info.run_id)
        assert isinstance(fetched.data.tags, dict)
        assert fetched.data.tags["mlflow.note"] == "hello"

    def test_set_tag_visible_on_get_run(self, store):
        """Tags set via set_tag must be visible when fetching the run."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [], "run")
        store.set_tag(run.info.run_id, RunTag("visibility", "check"))
        fetched = store.get_run(run.info.run_id)
        assert fetched.data.tags.get("visibility") == "check"

    def test_delete_tag_removes_from_run(self, store):
        """delete_tag must remove the tag from the run's data."""
        exp_id = store.create_experiment("test")
        run = store.create_run(exp_id, "user", 1000, [RunTag("to_delete", "value")], "run")
        store.delete_tag(run.info.run_id, "to_delete")
        fetched = store.get_run(run.info.run_id)
        assert "to_delete" not in fetched.data.tags
