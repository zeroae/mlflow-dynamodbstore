"""Phase 2b: MLflow tracking workspace isolation tests run against DynamoDB.

Uses `workspace_tracking_store` fixture (LOAD-BEARING name).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.tracking.test_sqlalchemy_workspace_store import (  # noqa: E402, F401
    test_entity_associations_are_workspace_scoped,
    test_experiment_lifecycle_operations_are_workspace_scoped,
    test_experiment_tags_are_workspace_scoped,
    test_experiments_are_workspace_scoped,
    test_run_data_logging_enforces_workspaces,
    test_run_lifecycle_operations_workspace_isolation,
    test_runs_are_workspace_scoped,
    test_search_and_history_calls_are_workspace_scoped,
    test_search_datasets_is_workspace_scoped,
    test_search_traces_is_workspace_scoped,
    test_trace_tag_operations_are_workspace_scoped,
)

# --- Tests using SqlAlchemy-specific ManagedSessionMaker ---
_xfail_sqlalchemy = pytest.mark.xfail(reason="Test uses ManagedSessionMaker (SqlAlchemy-specific)")
test_search_traces_is_workspace_scoped = _xfail_sqlalchemy(test_search_traces_is_workspace_scoped)
test_trace_tag_operations_are_workspace_scoped = _xfail_sqlalchemy(
    test_trace_tag_operations_are_workspace_scoped
)
test_search_and_history_calls_are_workspace_scoped = _xfail_sqlalchemy(
    test_search_and_history_calls_are_workspace_scoped
)

# --- Entity associations — DONE (search_entities_by_source/destination) ---
