"""E2E tests for the MLflow demo data generation endpoint."""

import pytest
import requests

pytestmark = pytest.mark.e2e


class TestDemoGeneration:
    """Test the MLflow demo data generation endpoint."""

    def test_generate_demo_prompts(self, http_session: requests.Session):
        """Demo prompts generation should succeed."""
        resp = http_session.post(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/demo/generate",
            json={"features": ["prompts"]},
            timeout=120,
        )
        assert resp.status_code == 200, f"Demo prompts failed: {resp.text}"

    def test_generate_demo_traces(self, http_session: requests.Session):
        """Demo traces generation should succeed (exercises artifact location)."""
        resp = http_session.post(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/demo/generate",
            json={"features": ["traces"]},
            timeout=120,
        )
        assert resp.status_code == 200, f"Demo traces failed: {resp.text}"

    def test_generate_demo_evaluation(self, http_session: requests.Session):
        """Demo evaluation requires dataset support."""
        resp = http_session.post(
            f"{http_session.base_url}/ajax-api/3.0/mlflow/demo/generate",
            json={"features": ["evaluation"]},
            timeout=300,
        )
        assert resp.status_code == 200, f"Demo evaluation failed: {resp.text}"
