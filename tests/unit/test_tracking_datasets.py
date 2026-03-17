"""Tests for Evaluation Dataset CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import pytest
from mlflow.entities import EvaluationDataset
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList


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


class TestSearchDatasets:
    def test_search_all_in_workspace(self, tracking_store):
        tracking_store.create_dataset(name="ds-alpha")
        tracking_store.create_dataset(name="ds-beta")
        results = tracking_store.search_datasets()
        assert len(results) == 2

    def test_search_by_experiment_id(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp-search")
        ds1 = tracking_store.create_dataset(name="linked", experiment_ids=[exp_id])
        tracking_store.create_dataset(name="unlinked")
        results = tracking_store.search_datasets(experiment_ids=[exp_id])
        assert len(results) == 1
        assert results[0].dataset_id == ds1.dataset_id

    def test_search_by_name_prefix_filter(self, tracking_store):
        tracking_store.create_dataset(name="prod-dataset")
        tracking_store.create_dataset(name="dev-dataset")
        results = tracking_store.search_datasets(filter_string="name LIKE 'prod%'")
        assert len(results) == 1
        assert results[0].name == "prod-dataset"

    def test_search_by_name_substring_filter(self, tracking_store):
        tracking_store.create_dataset(name="my-prod-dataset")
        tracking_store.create_dataset(name="my-dev-dataset")
        results = tracking_store.search_datasets(filter_string="name LIKE '%prod%'")
        assert len(results) == 1
        assert results[0].name == "my-prod-dataset"

    def test_search_order_by_name(self, tracking_store):
        tracking_store.create_dataset(name="charlie")
        tracking_store.create_dataset(name="alpha")
        tracking_store.create_dataset(name="bravo")
        results = tracking_store.search_datasets(order_by=["name ASC"])
        names = [r.name for r in results]
        assert names == ["alpha", "bravo", "charlie"]

    def test_search_pagination(self, tracking_store):
        for i in range(5):
            tracking_store.create_dataset(name=f"page-ds-{i}")
        page1 = tracking_store.search_datasets(max_results=2)
        assert len(page1) == 2
        assert page1.token is not None
        page2 = tracking_store.search_datasets(max_results=2, page_token=page1.token)
        assert len(page2) == 2

    def test_search_returns_paged_list(self, tracking_store):
        tracking_store.create_dataset(name="paged")
        results = tracking_store.search_datasets()
        assert isinstance(results, PagedList)

    def test_search_empty(self, tracking_store):
        results = tracking_store.search_datasets()
        assert len(results) == 0
