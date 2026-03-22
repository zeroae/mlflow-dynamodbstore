"""Phase 2b: MLflow tracking workspace isolation tests run against DynamoDB.

Uses `workspace_tracking_store` fixture (LOAD-BEARING name).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from mlflow.entities import Metric, TraceInfo, TraceState, ViewType, trace_location
from mlflow.exceptions import MlflowException

from tests.store.tracking.test_sqlalchemy_workspace_store import (  # noqa: E402, F401
    WorkspaceContext,
    _create_run,
    _now_ms,
    test_entity_associations_are_workspace_scoped,
    test_experiment_lifecycle_operations_are_workspace_scoped,
    test_experiment_tags_are_workspace_scoped,
    test_experiments_are_workspace_scoped,
    test_run_data_logging_enforces_workspaces,
    test_run_lifecycle_operations_workspace_isolation,
    test_runs_are_workspace_scoped,
    test_search_datasets_is_workspace_scoped,
)

# --- Entity associations — DONE (search_entities_by_source/destination) ---


# --- DynamoDB-compatible replacements for ManagedSessionMaker tests ---
# The vendored tests use raw SQL inserts; these use the store's public API.


def _create_trace_via_api(store, trace_id, exp_id):
    """Create a trace using start_trace (no ManagedSessionMaker needed)."""
    ti = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=_now_ms(),
        execution_duration=0,
        state=TraceState.OK,
        tags={},
        trace_metadata={},
    )
    return store.start_trace(ti)


def test_search_traces_is_workspace_scoped(workspace_tracking_store):
    """DynamoDB replacement: uses start_trace instead of ManagedSessionMaker."""
    with WorkspaceContext("team-search-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-search-a")
        _create_trace_via_api(workspace_tracking_store, "trace-a", exp_a)

    with WorkspaceContext("team-search-b"):
        exp_b = workspace_tracking_store.create_experiment("exp-search-b")
        _create_trace_via_api(workspace_tracking_store, "trace-b", exp_b)

        # Cross-workspace search returns nothing
        results, _ = workspace_tracking_store.search_traces(locations=[exp_a])
        assert results == []

        # Same-workspace search works
        results, _ = workspace_tracking_store.search_traces(locations=[exp_b])
        assert len(results) == 1
        assert results[0].trace_id == "trace-b"


def test_trace_tag_operations_are_workspace_scoped(workspace_tracking_store):
    """DynamoDB replacement: uses start_trace instead of ManagedSessionMaker."""
    with WorkspaceContext("team-trace-a"):
        exp_a = workspace_tracking_store.create_experiment("exp-trace-a")
        _create_trace_via_api(workspace_tracking_store, "trace-tag-a", exp_a)

    with WorkspaceContext("team-trace-b"):
        workspace_tracking_store.create_experiment("exp-trace-b")

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.set_trace_tag("trace-tag-a", "key", "value")

        with pytest.raises(MlflowException, match="not found"):
            workspace_tracking_store.delete_trace_tag("trace-tag-a", "key")


def test_search_and_history_calls_are_workspace_scoped(workspace_tracking_store):
    """DynamoDB replacement: uses search_runs instead of _list_run_infos."""
    exp_a_id, run_a = _create_run(
        workspace_tracking_store, "team-a", "search-exp-a", "run-a", user="alice"
    )
    exp_b_id, run_b = _create_run(
        workspace_tracking_store, "team-b", "search-exp-b", "run-b", user="bob"
    )

    with WorkspaceContext("team-a"):
        workspace_tracking_store.log_metric(run_a.info.run_id, Metric("metric", 1.0, _now_ms(), 0))
        runs = workspace_tracking_store.search_runs([exp_a_id, exp_b_id], None, ViewType.ALL)
        assert {r.info.run_id for r in runs} == {run_a.info.run_id}

    with WorkspaceContext("team-b"):
        runs = workspace_tracking_store.search_runs([exp_a_id, exp_b_id], None, ViewType.ALL)
        assert {r.info.run_id for r in runs} == {run_b.info.run_id}

        with pytest.raises(MlflowException, match=f"Run with id={run_a.info.run_id} not found"):
            workspace_tracking_store.get_metric_history(run_a.info.run_id, "metric")

    with WorkspaceContext("team-a"):
        history = workspace_tracking_store.get_metric_history(run_a.info.run_id, "metric")
        assert [m.value for m in history] == [1.0]
        # Verify only team-a's run is visible via search_runs
        runs = workspace_tracking_store.search_runs([exp_a_id], None, ViewType.ALL)
        assert len(runs) == 1

    with WorkspaceContext("team-b"):
        # team-b should not see team-a's runs in exp_a
        runs = workspace_tracking_store.search_runs([exp_a_id], None, ViewType.ALL)
        assert runs == []
