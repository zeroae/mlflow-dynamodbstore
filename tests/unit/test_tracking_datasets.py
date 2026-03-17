"""Tests for Evaluation Dataset CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import pytest
from mlflow.entities import EvaluationDataset
from mlflow.exceptions import MlflowException


class TestCreateDataset:
    def test_create_returns_dataset(self, tracking_store):
        ds = tracking_store.create_dataset(name="my-dataset")
        assert isinstance(ds, EvaluationDataset)
        assert ds.name == "my-dataset"
        assert ds.dataset_id.startswith("eval_")
        assert ds.digest != ""
        assert ds.created_time > 0
        assert ds.last_update_time > 0

    def test_create_with_tags(self, tracking_store):
        ds = tracking_store.create_dataset(name="tagged-ds", tags={"env": "prod", "team": "ml"})
        assert ds.tags == {"env": "prod", "team": "ml"}

    def test_create_with_experiment_ids(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp")
        ds = tracking_store.create_dataset(name="linked-ds", experiment_ids=[exp_id])
        exp_ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert exp_id in exp_ids

    def test_create_duplicate_name_raises(self, tracking_store):
        tracking_store.create_dataset(name="unique-name")
        with pytest.raises(MlflowException, match="already exists"):
            tracking_store.create_dataset(name="unique-name")


class TestGetDataset:
    def test_get_existing(self, tracking_store):
        created = tracking_store.create_dataset(name="get-test")
        fetched = tracking_store.get_dataset(created.dataset_id)
        assert fetched.dataset_id == created.dataset_id
        assert fetched.name == "get-test"

    def test_get_nonexistent_raises(self, tracking_store):
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_dataset("eval_nonexistent")

    def test_get_includes_tags(self, tracking_store):
        created = tracking_store.create_dataset(name="tags-test", tags={"key": "val"})
        fetched = tracking_store.get_dataset(created.dataset_id)
        assert fetched.tags == {"key": "val"}

    def test_get_lazy_loads_experiment_ids(self, tracking_store):
        exp_id = tracking_store.create_experiment("lazy-exp")
        ds = tracking_store.create_dataset(name="lazy-test", experiment_ids=[exp_id])
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert exp_id in fetched.experiment_ids


class TestDeleteDataset:
    def test_delete_removes_dataset(self, tracking_store):
        ds = tracking_store.create_dataset(name="delete-me")
        tracking_store.delete_dataset(ds.dataset_id)
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_dataset(ds.dataset_id)

    def test_delete_nonexistent_is_idempotent(self, tracking_store):
        tracking_store.delete_dataset("eval_nonexistent")  # no error
