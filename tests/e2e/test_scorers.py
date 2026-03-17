"""E2E tests for scorer operations via MLflow REST API."""

import json
import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e

_V3_BASE = "/ajax-api/3.0/mlflow"


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _post(server, path, data=None):
    return requests.post(f"{server}{_V3_BASE}{path}", json=data or {}, timeout=10)


def _get(server, path, params=None):
    return requests.get(f"{server}{_V3_BASE}{path}", params=params or {}, timeout=10)


def _delete(server, path, data=None):
    return requests.delete(f"{server}{_V3_BASE}{path}", json=data or {}, timeout=10)


def _put(server, path, data=None):
    return requests.put(f"{server}{_V3_BASE}{path}", json=data or {}, timeout=10)


class TestScorers:
    def test_list_scorers_empty(self, mlflow_server, client):
        """list_scorers returns empty list, not 500."""
        exp_id = client.create_experiment(f"e2e-scorers-empty-{_uid()}")
        resp = _get(mlflow_server, "/scorers/list", {"experiment_id": exp_id})
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        assert resp.json().get("scorers", []) == []

    def test_register_and_get_scorer(self, mlflow_server, client):
        """Register a scorer, then get it back."""
        exp_id = client.create_experiment(f"e2e-scorers-reg-{_uid()}")
        scorer_def = json.dumps({"name": "accuracy", "call_source": "test"})

        # Register
        resp = _post(
            mlflow_server,
            "/scorers/register",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
                "serialized_scorer": scorer_def,
            },
        )
        assert resp.status_code == 200, f"Register failed: {resp.text}"
        data = resp.json()
        assert data["version"] == 1
        assert data["name"] == "accuracy"
        assert "scorer_id" in data

        # Get latest
        resp = _get(
            mlflow_server,
            "/scorers/get",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
            },
        )
        assert resp.status_code == 200, f"Get failed: {resp.text}"
        assert resp.json()["scorer"]["scorer_version"] == 1

    def test_register_multiple_versions_and_list(self, mlflow_server, client):
        """Register multiple versions, list returns latest per name."""
        exp_id = client.create_experiment(f"e2e-scorers-ver-{_uid()}")

        for i in range(1, 4):
            resp = _post(
                mlflow_server,
                "/scorers/register",
                {
                    "experiment_id": exp_id,
                    "name": "accuracy",
                    "serialized_scorer": json.dumps({"v": i}),
                },
            )
            assert resp.status_code == 200

        # List — should return one scorer at version 3
        resp = _get(mlflow_server, "/scorers/list", {"experiment_id": exp_id})
        assert resp.status_code == 200
        scorers = resp.json().get("scorers", [])
        assert len(scorers) == 1
        assert scorers[0]["scorer_version"] == 3

    def test_list_scorer_versions(self, mlflow_server, client):
        """list_scorer_versions returns all versions."""
        exp_id = client.create_experiment(f"e2e-scorers-lv-{_uid()}")
        for i in range(1, 4):
            _post(
                mlflow_server,
                "/scorers/register",
                {
                    "experiment_id": exp_id,
                    "name": "accuracy",
                    "serialized_scorer": json.dumps({"v": i}),
                },
            )

        resp = _get(
            mlflow_server,
            "/scorers/versions",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
            },
        )
        assert resp.status_code == 200
        versions = [s["scorer_version"] for s in resp.json().get("scorers", [])]
        assert versions == [1, 2, 3]

    def test_delete_scorer(self, mlflow_server, client):
        """Delete all versions of a scorer."""
        exp_id = client.create_experiment(f"e2e-scorers-del-{_uid()}")
        _post(
            mlflow_server,
            "/scorers/register",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
                "serialized_scorer": "{}",
            },
        )

        resp = _delete(
            mlflow_server,
            "/scorers/delete",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
            },
        )
        assert resp.status_code == 200

        # Verify gone
        resp = _get(mlflow_server, "/scorers/list", {"experiment_id": exp_id})
        assert resp.json().get("scorers", []) == []

    def test_online_scoring_config_lifecycle(self, mlflow_server, client):
        """Upsert and get online scoring config."""
        exp_id = client.create_experiment(f"e2e-scorers-osc-{_uid()}")
        reg_resp = _post(
            mlflow_server,
            "/scorers/register",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
                "serialized_scorer": "{}",
            },
        )
        scorer_id = reg_resp.json()["scorer_id"]

        # Upsert config
        resp = _put(
            mlflow_server,
            "/scorers/online-config",
            {
                "experiment_id": exp_id,
                "name": "accuracy",
                "sample_rate": 0.5,
            },
        )
        assert resp.status_code == 200, f"Upsert failed: {resp.text}"

        # Get config
        resp = _get(
            mlflow_server,
            "/scorers/online-configs",
            {
                "scorer_ids": scorer_id,
            },
        )
        assert resp.status_code == 200, f"Get configs failed: {resp.text}"
