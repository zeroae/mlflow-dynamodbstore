"""Phase 2a: MLflow workspace store tests run against DynamoDB.

MLflow's workspace tests use a `workspace_store` fixture.
Our conftest provides this backed by DynamoDB (moto).

Tests that use `_workspace_rows()` (which calls `ManagedSessionMaker`) are
supported by monkey-patching the helper to use `list_workspaces()` instead.

test_delete_workspace_restrict_blocks_when_resources_exist uses
ManagedSessionMaker directly to INSERT raw SQL rows — this requires a
DynamoDB-native equivalent via the tracking store.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

# Monkey-patch _workspace_rows before importing the tests so they use
# our DynamoDB-compatible implementation instead of ManagedSessionMaker.
import tests.store.workspace.test_sqlalchemy_store as _ws_test_mod


def _ddb_workspace_rows(store):
    """DynamoDB-compatible replacement for _workspace_rows.

    Queries the DynamoDB table directly via GSI2, bypassing the store API,
    to match the intent of the original which reads raw DB rows.
    """
    from mlflow_dynamodbstore.dynamodb.schema import GSI2_WORKSPACES

    items = store._table.query(pk=GSI2_WORKSPACES, index_name="gsi2")
    return {(item["name"], item.get("description") or None) for item in items}


_ws_test_mod._workspace_rows = _ddb_workspace_rows

from tests.store.workspace.test_sqlalchemy_store import (  # noqa: E402, F401
    test_create_workspace_duplicate_raises,
    test_create_workspace_invalid_name_raises,
    test_create_workspace_persists_record,
    test_delete_default_workspace_rejected,
    test_delete_workspace_removes_empty_workspace,
    test_delete_workspace_restrict_allows_empty_workspace,
    test_delete_workspace_restrict_blocks_when_resources_exist,
    test_get_workspace_not_found,
    test_get_workspace_success,
    test_list_workspaces_returns_all,
    test_update_workspace_can_clear_default_artifact_root,
    test_update_workspace_changes_description,
    test_update_workspace_sets_default_artifact_root,
)

# --- test_delete_workspace_restrict_blocks_when_resources_exist uses
# ManagedSessionMaker directly to INSERT experiments — needs DynamoDB-native fix ---
_xfail_sql_internal = pytest.mark.xfail(
    reason="Test uses ManagedSessionMaker to INSERT raw SQL rows"
)
test_delete_workspace_restrict_blocks_when_resources_exist = _xfail_sql_internal(
    test_delete_workspace_restrict_blocks_when_resources_exist
)
