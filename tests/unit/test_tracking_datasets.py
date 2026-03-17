"""Tests for Evaluation Dataset CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import json
from decimal import Decimal

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

    def test_search_by_name_suffix_filter(self, tracking_store):
        """Test suffix LIKE filter (lines 2847-2849)."""
        tracking_store.create_dataset(name="prefix-test")
        tracking_store.create_dataset(name="prefix-other")
        results = tracking_store.search_datasets(filter_string="name LIKE '%test'")
        assert len(results) == 1
        assert results[0].name == "prefix-test"

    def test_search_by_name_exact_match(self, tracking_store):
        """Test exact match filter without wildcards (lines 2850-2851)."""
        tracking_store.create_dataset(name="exact-name")
        tracking_store.create_dataset(name="exact-name-2")
        results = tracking_store.search_datasets(filter_string="name LIKE 'exact-name'")
        assert len(results) == 1
        assert results[0].name == "exact-name"

    def test_search_order_by_name(self, tracking_store):
        tracking_store.create_dataset(name="charlie")
        tracking_store.create_dataset(name="alpha")
        tracking_store.create_dataset(name="bravo")
        results = tracking_store.search_datasets(order_by=["name ASC"])
        names = [r.name for r in results]
        assert names == ["alpha", "bravo", "charlie"]

    def test_search_order_by_created_time(self, tracking_store):
        """Test order_by created_time (lines 2862-2863)."""
        tracking_store.create_dataset(name="ds-first")
        tracking_store.create_dataset(name="ds-second")
        tracking_store.create_dataset(name="ds-third")

        results = tracking_store.search_datasets(order_by=["created_time ASC"])
        names = [r.name for r in results]
        assert names == ["ds-first", "ds-second", "ds-third"]

    def test_search_order_by_last_update_time(self, tracking_store):
        """Test order_by last_update_time (lines 2864-2865)."""
        ds1 = tracking_store.create_dataset(name="ds-early")
        tracking_store.create_dataset(name="ds-late")

        # Manually update ds1 to have a newer last_update_time
        tracking_store.set_dataset_tags(ds1.dataset_id, {"tag": "updated"})

        results = tracking_store.search_datasets(order_by=["last_update_time ASC"])
        names = [r.name for r in results]
        assert names == ["ds-late", "ds-early"]

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

    def test_search_with_orphaned_gsi_entry(self, tracking_store):
        """Test that orphaned GSI entries (no META) are skipped (lines 2800, 2821).

        This simulates a dataset where the META item was deleted but GSI entries remain.
        """
        # Create a dataset
        ds = tracking_store.create_dataset(name="orphaned-test")

        # Manually delete the META item to simulate orphaned GSI entry
        from mlflow_dynamodbstore.dynamodb.schema import (
            PK_DATASET_PREFIX,
            SK_DATASET_META,
        )

        pk = f"{PK_DATASET_PREFIX}{ds.dataset_id}"
        tracking_store._table.delete_item(pk=pk, sk=SK_DATASET_META)

        # Search should return empty (META was missing, so dataset is skipped)
        results = tracking_store.search_datasets()
        assert len(results) == 0


class TestDatasetTags:
    def test_set_tags(self, tracking_store):
        ds = tracking_store.create_dataset(name="tag-crud")
        tracking_store.set_dataset_tags(ds.dataset_id, {"k1": "v1", "k2": "v2"})
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert fetched.tags["k1"] == "v1"
        assert fetched.tags["k2"] == "v2"

    def test_set_tags_overwrites(self, tracking_store):
        ds = tracking_store.create_dataset(name="overwrite", tags={"k": "old"})
        tracking_store.set_dataset_tags(ds.dataset_id, {"k": "new"})
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert fetched.tags["k"] == "new"

    def test_delete_tag(self, tracking_store):
        ds = tracking_store.create_dataset(name="del-tag", tags={"k": "v"})
        tracking_store.delete_dataset_tag(ds.dataset_id, "k")
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert "k" not in fetched.tags


class TestDatasetExperimentAssociation:
    def test_get_experiment_ids(self, tracking_store):
        exp1 = tracking_store.create_experiment("exp-1")
        exp2 = tracking_store.create_experiment("exp-2")
        ds = tracking_store.create_dataset(name="multi-exp", experiment_ids=[exp1, exp2])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert set(ids) == {exp1, exp2}

    def test_add_dataset_to_experiments(self, tracking_store):
        exp1 = tracking_store.create_experiment("add-1")
        exp2 = tracking_store.create_experiment("add-2")
        ds = tracking_store.create_dataset(name="add-test")
        tracking_store.add_dataset_to_experiments(ds.dataset_id, [exp1, exp2])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert set(ids) == {exp1, exp2}

    def test_remove_dataset_from_experiments(self, tracking_store):
        exp1 = tracking_store.create_experiment("rm-1")
        exp2 = tracking_store.create_experiment("rm-2")
        ds = tracking_store.create_dataset(name="rm-test", experiment_ids=[exp1, exp2])
        tracking_store.remove_dataset_from_experiments(ds.dataset_id, [exp1])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert set(ids) == {exp2}

    def test_add_idempotent(self, tracking_store):
        exp1 = tracking_store.create_experiment("idem-1")
        ds = tracking_store.create_dataset(name="idem-test")
        tracking_store.add_dataset_to_experiments(ds.dataset_id, [exp1])
        tracking_store.add_dataset_to_experiments(ds.dataset_id, [exp1])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert ids.count(exp1) == 1


class TestDatasetRecords:
    def test_upsert_inserts_new_records(self, tracking_store):
        ds = tracking_store.create_dataset(name="records-test")
        result = tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [
                {"inputs": {"q": "hello"}, "outputs": {"a": "world"}},
                {"inputs": {"q": "foo"}, "outputs": {"a": "bar"}},
            ],
        )
        assert result["inserted"] == 2
        assert result["updated"] == 0

    def test_upsert_updates_existing_by_input_hash(self, tracking_store):
        ds = tracking_store.create_dataset(name="dedup-test")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": "hello"}, "outputs": {"a": "old"}}],
        )
        result = tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": "hello"}, "outputs": {"a": "new"}}],
        )
        assert result["inserted"] == 0
        assert result["updated"] == 1

    def test_upsert_updates_profile(self, tracking_store):
        ds = tracking_store.create_dataset(name="profile-test")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": "a"}}, {"inputs": {"q": "b"}}],
        )
        fetched = tracking_store.get_dataset(ds.dataset_id)
        profile = json.loads(fetched.profile) if fetched.profile else {}
        assert profile.get("num_records") == 2

    def test_upsert_nonexistent_dataset_raises(self, tracking_store):
        """Test upsert on non-existent dataset raises MlflowException (line 2997)."""
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.upsert_dataset_records(
                "eval_nope",
                [{"inputs": {"q": "test"}}],
            )

    def test_upsert_with_all_optional_fields(self, tracking_store):
        """Test record upsert with optional fields (lines 3037-3041, 3068-3072).

        Verify that optional fields are saved in the record item.
        """
        ds = tracking_store.create_dataset(name="optional-fields")
        result = tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [
                {
                    "inputs": {"q": "test"},
                    "outputs": {"a": "answer"},
                    "expectations": {"score": Decimal("0.9")},
                    "tags": {"validation": "passed"},
                    "source": "manual",
                }
            ],
        )
        assert result["inserted"] == 1
        assert result["updated"] == 0

        # Load and verify fields (source not passed to DatasetRecord)
        records, _ = tracking_store._load_dataset_records(ds.dataset_id)
        assert len(records) == 1
        rec = records[0]
        assert rec.inputs == {"q": "test"}
        assert rec.outputs == {"a": "answer"}
        assert rec.expectations == {"score": Decimal("0.9")}
        assert rec.tags == {"validation": "passed"}

    def test_upsert_update_with_optional_fields(self, tracking_store):
        """Test that updating a record can add/modify optional fields (lines 3037-3041)."""
        ds = tracking_store.create_dataset(name="update-optional")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": "hello"}, "outputs": {"a": "old"}}],
        )
        result = tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [
                {
                    "inputs": {"q": "hello"},
                    "outputs": {"a": "new"},
                    "expectations": {"accuracy": Decimal("0.95")},
                    "tags": {"status": "validated"},
                    "source": "system",
                }
            ],
        )
        assert result["updated"] == 1

        records, _ = tracking_store._load_dataset_records(ds.dataset_id)
        rec = records[0]
        assert rec.outputs == {"a": "new"}
        assert rec.expectations == {"accuracy": Decimal("0.95")}
        assert rec.tags == {"status": "validated"}

    def test_load_records_paginated(self, tracking_store):
        ds = tracking_store.create_dataset(name="paginate-test")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": f"item-{i}"}} for i in range(5)],
        )
        records, token = tracking_store._load_dataset_records(ds.dataset_id, max_results=2)
        assert len(records) == 2
        assert token is not None
        records2, token2 = tracking_store._load_dataset_records(
            ds.dataset_id, max_results=2, page_token=token
        )
        assert len(records2) == 2

    def test_delete_records(self, tracking_store):
        ds = tracking_store.create_dataset(name="del-records")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": "keep"}}, {"inputs": {"q": "remove"}}],
        )
        records, _ = tracking_store._load_dataset_records(ds.dataset_id)
        to_delete = [r.dataset_record_id for r in records if r.inputs["q"] == "remove"]
        deleted = tracking_store.delete_dataset_records(ds.dataset_id, to_delete)
        assert deleted == 1
        remaining, _ = tracking_store._load_dataset_records(ds.dataset_id)
        assert len(remaining) == 1
        assert remaining[0].inputs["q"] == "keep"


class TestSearchDatasetsLegacy:
    def test_search_datasets_v2_empty(self, tracking_store):
        """_search_datasets returns empty list when no datasets logged."""
        exp_id = tracking_store.create_experiment("v2-empty")
        results = tracking_store._search_datasets([exp_id])
        assert results == []

    def test_search_datasets_v2_with_data(self, tracking_store):
        """_search_datasets returns DatasetSummary from D# and DLINK# items."""
        from mlflow.entities import Dataset, DatasetInput, InputTag

        exp_id = tracking_store.create_experiment("v2-data")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="test",
            start_time=0,
            tags=[],
            run_name="test-run",
        )
        dataset = Dataset(
            name="my-ds",
            digest="abc123",
            source_type="local",
            source="file:///data",
        )
        dataset_input = DatasetInput(
            dataset=dataset,
            tags=[InputTag(key="mlflow.data.context", value="training")],
        )
        tracking_store.log_inputs(run.info.run_id, [dataset_input])

        results = tracking_store._search_datasets([exp_id])
        assert len(results) >= 1
        summary = results[0]
        assert summary.name == "my-ds"
        assert summary.digest == "abc123"
