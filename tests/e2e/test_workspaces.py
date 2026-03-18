"""E2E tests for workspace CRUD via REST API."""

import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestWorkspaces:
    def test_list_workspaces_has_default(self, http_session: requests.Session):
        """Default workspace exists on startup."""
        resp = http_session.get(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/workspaces",
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        names = [ws["name"] for ws in data.get("workspaces", [])]
        assert "default" in names

    def test_create_and_delete_workspace(self, http_session: requests.Session):
        """Create a workspace, verify it exists, then delete it."""
        ws_name = f"e2e-ws-{_uid()}"

        # Create
        resp = http_session.post(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/workspaces",
            json={"name": ws_name, "description": "test workspace"},
            timeout=10,
        )
        assert resp.status_code == 201, f"Create failed: {resp.status_code} {resp.text}"

        # Get
        resp = http_session.get(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["workspace"]["name"] == ws_name

        # Delete
        resp = http_session.delete(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code in (200, 204)

        # Verify deleted
        resp = http_session.get(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 404
