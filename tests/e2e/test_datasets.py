"""E2E tests for Evaluation Dataset operations via MLflow client SDK."""

import uuid

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestDatasets:
    def test_create_and_get_dataset(self, client: MlflowClient):
        """Create and retrieve an evaluation dataset."""
        ds = client.create_dataset(name=f"e2e-ds-{_uid()}")
        assert ds.dataset_id.startswith("eval_")

        fetched = client.get_dataset(ds.dataset_id)
        assert fetched.name == ds.name

    def test_search_datasets_empty(self, client: MlflowClient):
        """search_datasets returns empty for experiment with no datasets."""
        exp_id = client.create_experiment(f"e2e-ds-empty-{_uid()}")
        results = client.search_datasets(experiment_ids=[exp_id])
        assert len(results) == 0

    def test_delete_dataset(self, client: MlflowClient):
        """delete_dataset removes the dataset."""
        ds = client.create_dataset(name=f"e2e-ds-del-{_uid()}")
        client.delete_dataset(ds.dataset_id)
        with pytest.raises(Exception):
            client.get_dataset(ds.dataset_id)
