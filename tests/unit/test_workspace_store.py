"""Tests for DynamoDBWorkspaceStore CRUD operations."""

import pytest


class TestWorkspaceCRUD:
    def test_default_workspace_exists(self, workspace_store):
        ws = workspace_store.get_workspace("default")
        assert ws["name"] == "default"

    def test_create_workspace(self, workspace_store):
        workspace_store.create_workspace("test-ws", description="Test workspace")
        ws = workspace_store.get_workspace("test-ws")
        assert ws["name"] == "test-ws"
        assert ws["description"] == "Test workspace"

    def test_list_workspaces(self, workspace_store):
        workspace_store.create_workspace("ws-1")
        workspace_store.create_workspace("ws-2")
        workspaces = workspace_store.list_workspaces()
        names = [w["name"] for w in workspaces]
        assert "default" in names
        assert "ws-1" in names
        assert "ws-2" in names

    def test_update_workspace(self, workspace_store):
        workspace_store.create_workspace("test-ws")
        workspace_store.update_workspace("test-ws", description="Updated")
        ws = workspace_store.get_workspace("test-ws")
        assert ws["description"] == "Updated"

    def test_delete_workspace(self, workspace_store):
        workspace_store.create_workspace("test-ws")
        workspace_store.delete_workspace("test-ws")
        ws = workspace_store.get_workspace("test-ws")
        assert ws is None

    def test_delete_default_workspace_raises(self, workspace_store):
        with pytest.raises(Exception):
            workspace_store.delete_workspace("default")
