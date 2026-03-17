"""E2E tests for workspace CRUD via REST API."""

import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestWorkspaces:
    def test_list_workspaces_has_default(self, mlflow_server):
        """Default workspace exists on startup."""
        resp = requests.get(
            f"{mlflow_server}/ajax-api/3.0/mlflow/workspaces",
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        names = [ws["name"] for ws in data.get("workspaces", [])]
        assert "default" in names

    def test_create_and_delete_workspace(self, mlflow_server):
        """Create a workspace, verify it exists, then delete it."""
        ws_name = f"e2e-ws-{_uid()}"

        # Create
        resp = requests.post(
            f"{mlflow_server}/ajax-api/3.0/mlflow/workspaces",
            json={"name": ws_name, "description": "test workspace"},
            timeout=10,
        )
        assert resp.status_code == 201, f"Create failed: {resp.status_code} {resp.text}"

        # Get
        resp = requests.get(
            f"{mlflow_server}/ajax-api/3.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["workspace"]["name"] == ws_name

        # Delete
        resp = requests.delete(
            f"{mlflow_server}/ajax-api/3.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code in (200, 204)

        # Verify deleted
        resp = requests.get(
            f"{mlflow_server}/ajax-api/3.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 404
