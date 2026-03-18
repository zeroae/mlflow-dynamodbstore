"""Smoke test that runs first to validate server infrastructure is ready."""

import pytest
import requests

pytestmark = pytest.mark.e2e


class TestSmoke:
    def test_server_health(self, http_session: requests.Session):
        """Server is up and responding to health checks."""
        resp = http_session.get(f"{http_session.base_url}/health", timeout=10)
        assert resp.status_code == 200

    def test_server_version(self, http_session: requests.Session):
        """Server returns version info."""
        resp = http_session.get(f"{http_session.base_url}/version", timeout=10)
        assert resp.status_code == 200
