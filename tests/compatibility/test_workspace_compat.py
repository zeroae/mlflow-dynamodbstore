"""Phase 2a: MLflow workspace store tests run against DynamoDB.

MLflow's workspace tests use a `workspace_store` fixture.
Our conftest provides this backed by DynamoDB (moto).

Tests that use `_workspace_rows()` (which calls `ManagedSessionMaker`) are
supported by monkey-patching the helper to query the DynamoDB table directly.

test_delete_workspace_restrict_blocks_when_resources_exist uses
ManagedSessionMaker to INSERT/SELECT experiments — supported by adding a
ManagedSessionMaker property to the DynamoDB workspace store that delegates
to the tracking store's DynamoDB table.
"""

import sys
from contextlib import contextmanager
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

# ---------------------------------------------------------------------------
# DynamoDB-backed ManagedSessionMaker for workspace_store
# ---------------------------------------------------------------------------
# The vendored test_delete_workspace_restrict_blocks_when_resources_exist uses
# workspace_store.ManagedSessionMaker() to INSERT and SELECT experiment rows.
# We provide a lightweight adapter that translates these two known SQL patterns
# into DynamoDB operations on the shared table.


class _FakeRow:
    """Mimics a SQLAlchemy row result for `fetchone()[0]`."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return self._values[idx]


class _FakeSession:
    """Handles the two SQL statements used in the workspace restrict test."""

    def __init__(self, tracking_store):
        self._ts = tracking_store

    def execute(self, statement, params=None):
        sql = statement.text if hasattr(statement, "text") else str(statement)
        if "INSERT INTO experiments" in sql:
            from mlflow.utils.workspace_context import WorkspaceContext

            with WorkspaceContext(params["ws"]):
                self._ts.create_experiment(params["name"])
            return None
        if "SELECT workspace FROM experiments WHERE name" in sql:
            # Search all workspaces for the experiment by name
            from mlflow_dynamodbstore.dynamodb.schema import GSI3_EXP_NAME_PREFIX

            for ws in ("team-a", "default"):
                results = self._ts._table.query(
                    pk=f"{GSI3_EXP_NAME_PREFIX}{ws}#{params['name']}",
                    index_name="gsi3",
                    limit=1,
                )
                if results:
                    self._result = _FakeRow([results[0].get("workspace", ws)])
                    return self
            self._result = None
            return self
        raise NotImplementedError(f"Unsupported SQL in DynamoDB adapter: {sql}")

    def fetchone(self):
        return self._result


@contextmanager
def _ddb_managed_session(tracking_store):
    yield _FakeSession(tracking_store)


@pytest.fixture(autouse=True)
def _patch_managed_session_maker(workspace_store, tracking_store):
    """Add ManagedSessionMaker to the DynamoDB workspace store."""
    workspace_store.ManagedSessionMaker = lambda: _ddb_managed_session(tracking_store)
