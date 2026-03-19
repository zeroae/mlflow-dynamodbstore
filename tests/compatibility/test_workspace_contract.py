"""Phase 1: Contract fidelity tests for DynamoDB vs SqlAlchemy workspace stores."""

import pytest
from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException

from tests.compatibility.comparison import assert_entities_match
from tests.compatibility.field_policy import WORKSPACE


def test_list_workspaces(workspace_stores):
    """Default workspace should be present and match."""
    sql_workspaces = workspace_stores.sql.list_workspaces()
    ddb_workspaces = workspace_stores.ddb.list_workspaces()

    assert len(sql_workspaces) == len(ddb_workspaces)
    for sql_w, ddb_w in zip(
        sorted(sql_workspaces, key=lambda w: w.name),
        sorted(ddb_workspaces, key=lambda w: w.name),
    ):
        assert_entities_match(sql_w, ddb_w, WORKSPACE)


def test_create_and_get_workspace(workspace_stores):
    """Created workspace fields must match."""
    sql_ws = workspace_stores.sql.create_workspace(Workspace("test-ws", description="desc"))
    ddb_ws = workspace_stores.ddb.create_workspace(Workspace("test-ws", description="desc"))
    assert_entities_match(sql_ws, ddb_ws, WORKSPACE)

    sql_get = workspace_stores.sql.get_workspace("test-ws")
    ddb_get = workspace_stores.ddb.get_workspace("test-ws")
    assert_entities_match(sql_get, ddb_get, WORKSPACE)


def test_delete_workspace(workspace_stores):
    """Deleting a workspace should behave the same."""
    workspace_stores.sql.create_workspace(Workspace("del-ws"))
    workspace_stores.ddb.create_workspace(Workspace("del-ws"))

    workspace_stores.sql.delete_workspace("del-ws")
    workspace_stores.ddb.delete_workspace("del-ws")

    with pytest.raises(MlflowException):
        workspace_stores.sql.get_workspace("del-ws")
    with pytest.raises(MlflowException):
        workspace_stores.ddb.get_workspace("del-ws")
