"""Integration tests for DynamoDBWorkspaceStore via moto HTTP server."""

import pytest
from mlflow.entities import Workspace
from mlflow.exceptions import MlflowException


class TestWorkspaceIntegration:
    def test_workspace_lifecycle(self, workspace_store):
        # Default exists
        ws = workspace_store.get_workspace("default")
        assert ws is not None
        assert ws.name == "default"

        # Create
        workspace_store.create_workspace(Workspace("dev", description="Development"))
        ws = workspace_store.get_workspace("dev")
        assert ws.name == "dev"

        # List
        workspaces = workspace_store.list_workspaces()
        names = [w.name for w in workspaces]
        assert "default" in names
        assert "dev" in names

        # Delete
        workspace_store.delete_workspace("dev")
        with pytest.raises(MlflowException):
            workspace_store.get_workspace("dev")
