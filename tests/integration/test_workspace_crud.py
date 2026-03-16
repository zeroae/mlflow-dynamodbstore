"""Integration tests for DynamoDBWorkspaceStore via moto HTTP server."""


class TestWorkspaceIntegration:
    def test_workspace_lifecycle(self, workspace_store):
        # Default exists
        ws = workspace_store.get_workspace("default")
        assert ws is not None

        # Create
        workspace_store.create_workspace("dev", description="Development")
        ws = workspace_store.get_workspace("dev")
        assert ws["name"] == "dev"

        # List
        workspaces = workspace_store.list_workspaces()
        names = [w["name"] for w in workspaces]
        assert "default" in names
        assert "dev" in names

        # Delete
        workspace_store.delete_workspace("dev")
        assert workspace_store.get_workspace("dev") is None
