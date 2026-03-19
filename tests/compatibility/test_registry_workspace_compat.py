"""Phase 2b: MLflow registry workspace isolation tests run against DynamoDB.

Uses `workspace_registry_store` fixture (LOAD-BEARING name — must match
MLflow's fixture name in test_sqlalchemy_workspace_store.py).

Expected: these tests will FAIL until DynamoDB stores read from
WorkspaceContext instead of hard-coding self._workspace = "default".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.model_registry.test_sqlalchemy_workspace_store import (  # noqa: E402, F401
    test_default_workspace_context_allows_operations,
    test_model_version_operations_are_workspace_scoped,
    test_model_version_read_helpers_are_workspace_scoped,
    test_registered_model_operations_are_workspace_scoped,
    test_same_model_name_allowed_in_different_workspaces,
    test_update_and_delete_registered_model_metadata_are_workspace_scoped,
    test_webhook_operations_are_workspace_scoped,
)

# Mark all as xfail until workspace support is implemented
_xfail = pytest.mark.xfail(reason="DynamoDB stores hard-code workspace='default'")

test_registered_model_operations_are_workspace_scoped = _xfail(
    test_registered_model_operations_are_workspace_scoped
)
test_model_version_operations_are_workspace_scoped = _xfail(
    test_model_version_operations_are_workspace_scoped
)
test_model_version_read_helpers_are_workspace_scoped = _xfail(
    test_model_version_read_helpers_are_workspace_scoped
)
test_same_model_name_allowed_in_different_workspaces = _xfail(
    test_same_model_name_allowed_in_different_workspaces
)
test_update_and_delete_registered_model_metadata_are_workspace_scoped = _xfail(
    test_update_and_delete_registered_model_metadata_are_workspace_scoped
)
test_webhook_operations_are_workspace_scoped = _xfail(test_webhook_operations_are_workspace_scoped)
test_default_workspace_context_allows_operations = _xfail(
    test_default_workspace_context_allows_operations
)
