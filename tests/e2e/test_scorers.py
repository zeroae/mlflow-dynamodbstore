"""E2E tests for scorer operations via MLflow REST API."""

import json
import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e

_V3_BASE = "/ajax-api/3.0/mlflow"


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestScorers:
    def test_list_scorers_empty(self, http_session: requests.Session, client):
        """list_scorers returns empty list, not 500."""
        exp_id = client.create_experiment(f"e2e-scorers-empty-{_uid()}")
        resp = http_session.get(
            f"{http_session.base_url}{_V3_BASE}/scorers/list",
            params={"experiment_id": exp_id},
            timeout=10,
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        assert resp.json().get("scorers", []) == []

    def test_register_and_get_scorer(self, http_session: requests.Session, client):
        """Register a scorer, then get it back."""
        exp_id = client.create_experiment(f"e2e-scorers-reg-{_uid()}")
        scorer_def = json.dumps({"name": "accuracy", "call_source": "test"})

        # Register
        resp = http_session.post(
            f"{http_session.base_url}{_V3_BASE}/scorers/register",
            json={
                "experiment_id": exp_id,
                "name": "accuracy",
                "serialized_scorer": scorer_def,
            },
            timeout=10,
        )
        assert resp.status_code == 200, f"Register failed: {resp.text}"
        data = resp.json()
        assert data["version"] == 1
        assert data["name"] == "accuracy"
        assert "scorer_id" in data

        # Get latest
        resp = http_session.get(
            f"{http_session.base_url}{_V3_BASE}/scorers/get",
            params={
                "experiment_id": exp_id,
                "name": "accuracy",
            },
            timeout=10,
        )
        assert resp.status_code == 200, f"Get failed: {resp.text}"
        assert resp.json()["scorer"]["scorer_version"] == 1

    def test_register_multiple_versions_and_list(self, http_session: requests.Session, client):
        """Register multiple versions, list returns latest per name."""
        exp_id = client.create_experiment(f"e2e-scorers-ver-{_uid()}")

        for i in range(1, 4):
            resp = http_session.post(
                f"{http_session.base_url}{_V3_BASE}/scorers/register",
                json={
                    "experiment_id": exp_id,
                    "name": "accuracy",
                    "serialized_scorer": json.dumps({"v": i}),
                },
                timeout=10,
            )
            assert resp.status_code == 200

        # List — should return one scorer at version 3
        resp = http_session.get(
            f"{http_session.base_url}{_V3_BASE}/scorers/list",
            params={"experiment_id": exp_id},
            timeout=10,
        )
        assert resp.status_code == 200
        scorers = resp.json().get("scorers", [])
        assert len(scorers) == 1
        assert scorers[0]["scorer_version"] == 3

    def test_list_scorer_versions(self, http_session: requests.Session, client):
        """list_scorer_versions returns all versions."""
        exp_id = client.create_experiment(f"e2e-scorers-lv-{_uid()}")
        for i in range(1, 4):
            http_session.post(
                f"{http_session.base_url}{_V3_BASE}/scorers/register",
                json={
                    "experiment_id": exp_id,
                    "name": "accuracy",
                    "serialized_scorer": json.dumps({"v": i}),
                },
                timeout=10,
            )

        resp = http_session.get(
            f"{http_session.base_url}{_V3_BASE}/scorers/versions",
            params={
                "experiment_id": exp_id,
                "name": "accuracy",
            },
            timeout=10,
        )
        assert resp.status_code == 200
        versions = [s["scorer_version"] for s in resp.json().get("scorers", [])]
        assert versions == [1, 2, 3]

    def test_delete_scorer(self, http_session: requests.Session, client):
        """Delete all versions of a scorer."""
        exp_id = client.create_experiment(f"e2e-scorers-del-{_uid()}")
        http_session.post(
            f"{http_session.base_url}{_V3_BASE}/scorers/register",
            json={
                "experiment_id": exp_id,
                "name": "accuracy",
                "serialized_scorer": "{}",
            },
            timeout=10,
        )

        resp = http_session.delete(
            f"{http_session.base_url}{_V3_BASE}/scorers/delete",
            json={
                "experiment_id": exp_id,
                "name": "accuracy",
            },
            timeout=10,
        )
        assert resp.status_code == 200

        # Verify gone
        resp = http_session.get(
            f"{http_session.base_url}{_V3_BASE}/scorers/list",
            params={"experiment_id": exp_id},
            timeout=10,
        )
        assert resp.json().get("scorers", []) == []

    def test_online_scoring_config_lifecycle(self, http_session: requests.Session, client):
        """Upsert and get online scoring config."""
        exp_id = client.create_experiment(f"e2e-scorers-osc-{_uid()}")
        reg_resp = http_session.post(
            f"{http_session.base_url}{_V3_BASE}/scorers/register",
            json={
                "experiment_id": exp_id,
                "name": "accuracy",
                "serialized_scorer": "{}",
            },
            timeout=10,
        )
        scorer_id = reg_resp.json()["scorer_id"]

        # Upsert config
        resp = http_session.put(
            f"{http_session.base_url}{_V3_BASE}/scorers/online-config",
            json={
                "experiment_id": exp_id,
                "name": "accuracy",
                "sample_rate": 0.5,
            },
            timeout=10,
        )
        assert resp.status_code == 200, f"Upsert failed: {resp.text}"

        # Get config
        resp = http_session.get(
            f"{http_session.base_url}{_V3_BASE}/scorers/online-configs",
            params={
                "scorer_ids": scorer_id,
            },
            timeout=10,
        )
        assert resp.status_code == 200, f"Get configs failed: {resp.text}"
