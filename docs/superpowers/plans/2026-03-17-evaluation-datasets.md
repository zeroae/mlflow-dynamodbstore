# Evaluation Datasets Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MLflow V3 Evaluation Datasets in the DynamoDB tracking store (12 store methods + 1 legacy method), eliminating `search_datasets` 500 errors on the UI.

**Architecture:** Datasets live in their own DynamoDB partition family (`DS#<dataset_id>`) with META, Tag, Record, and Experiment Link sub-items. GSI1/GSI2/GSI3 provide cross-experiment lookups, workspace listing, and name uniqueness. LSI3 enables O(1) record deduplication by `input_hash`.

**Tech Stack:** Python 3.11+, boto3 (DynamoDB), moto (testing), MLflow 3.10.1 entities (`EvaluationDataset`, `DatasetRecord`, `PagedList`)

**Spec:** `docs/superpowers/specs/2026-03-16-evaluation-datasets-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/mlflow_dynamodbstore/dynamodb/schema.py` | Modify | Add dataset PK/SK/GSI constants |
| `src/mlflow_dynamodbstore/dynamodb/table.py` | Modify | Add `batch_delete` and `query_page` methods |
| `src/mlflow_dynamodbstore/tracking_store.py` | Modify | Add 13 dataset store methods |
| `tests/unit/test_tracking_datasets.py` | Create | Unit tests for all dataset operations |
| `tests/e2e/test_datasets.py` | Create | E2E tests for dataset REST endpoints |
| `tests/e2e/test_traces.py` | Modify | Remove xfail from demo evaluation test |

**Note on integration tests:** Integration tests (`tests/integration/`) run the store in-process with moto (no REST layer). The E2E tests cover REST round-trips. The dataset REST handlers in MLflow are standard protobuf-based handlers that require no custom serialization, so integration-tier tests add limited value beyond unit + E2E. If REST serialization issues arise during E2E, an integration test file can be added then.

---

## Task 1: Schema Constants

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py`
- Test: `tests/unit/dynamodb/test_schema.py` (existing, verify import)

- [ ] **Step 1: Add dataset schema constants**

Add to `src/mlflow_dynamodbstore/dynamodb/schema.py` after the existing constants:

```python
# ---------------------------------------------------------------------------
# Evaluation Dataset partition (DS#<dataset_id>)
# ---------------------------------------------------------------------------
PK_DATASET_PREFIX = "DS#"
SK_DATASET_META = "DS#META"
SK_DATASET_TAG_PREFIX = "DS#TAG#"
SK_DATASET_RECORD_PREFIX = "DS#REC#"
SK_DATASET_EXP_PREFIX = "DS#EXP#"

# GSI prefixes for evaluation datasets
# Note: GSI1_DS_PREFIX = "DS#" already exists for legacy V2 dataset items.
# The new DS_EXP# prefix is for experiment-dataset associations (distinct purpose).
GSI1_DS_EXP_PREFIX = "DS_EXP#"
GSI2_DS_LIST_PREFIX = "DS_LIST#"
GSI3_DS_NAME_PREFIX = "DS_NAME#"
```

- [ ] **Step 2: Verify constants import cleanly**

Run: `python -c "from mlflow_dynamodbstore.dynamodb.schema import PK_DATASET_PREFIX, SK_DATASET_META, GSI1_DS_EXP_PREFIX; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat(datasets): add evaluation dataset schema constants"
```

---

## Task 2: DynamoDBTable Infrastructure

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/table.py`
- Modify: `tests/unit/dynamodb/test_table.py` (existing file — uses `@mock_aws` decorator pattern, NOT fixtures)

**Note:** `tests/unit/dynamodb/test_table.py` already exists with 11 tests using the `@mock_aws` decorator + `_create_test_table()` helper pattern. Add new test classes following the same pattern. The existing `_create_test_table()` helper already creates a table with `lsi1` and `gsi1` — extend it to add `lsi3` for the `input_hash` dedup pattern.

- [ ] **Step 1: Extend `_create_test_table` and write failing test for `batch_delete`**

Add `lsi3sk` to the existing `_create_test_table()` helper's `AttributeDefinitions` and `LocalSecondaryIndexes`. Then append these test classes to `tests/unit/dynamodb/test_table.py`:

```python
class TestBatchDelete:
    @mock_aws
    def test_batch_delete_removes_items(self):
        """batch_delete removes items by PK+SK keys."""
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "A", "SK": "1", "val": "a"})
        table.put_item({"PK": "A", "SK": "2", "val": "b"})
        table.put_item({"PK": "A", "SK": "3", "val": "c"})

        table.batch_delete([{"PK": "A", "SK": "1"}, {"PK": "A", "SK": "2"}])

        assert table.get_item("A", "1") is None
        assert table.get_item("A", "2") is None
        assert table.get_item("A", "3") is not None

    @mock_aws
    def test_batch_delete_empty_list(self):
        """batch_delete with empty list is a no-op."""
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.batch_delete([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/dynamodb/test_table.py::TestBatchDelete -v`
Expected: FAIL with `AttributeError: 'DynamoDBTable' object has no attribute 'batch_delete'`

- [ ] **Step 3: Implement `batch_delete`**

Add to `src/mlflow_dynamodbstore/dynamodb/table.py` after `batch_write`:

```python
def batch_delete(self, keys: list[dict[str, Any]]) -> None:
    """Batch delete items by PK+SK key dicts."""
    with self._table.batch_writer() as batch:
        for key in keys:
            batch.delete_item(Key=key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/dynamodb/test_table.py::TestBatchDelete -v`
Expected: PASS

- [ ] **Step 5: Write failing test for `query_page`**

Add to `tests/unit/dynamodb/test_table.py`:

```python
class TestQueryPage:
    @mock_aws
    def test_query_page_returns_items_and_token(self):
        """query_page returns (items, last_evaluated_key)."""
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        for i in range(5):
            table.put_item({"PK": "B", "SK": f"R#{i:03d}", "val": str(i)})

        items, lek = table.query_page(pk="B", sk_prefix="R#", limit=2)
        assert len(items) == 2
        assert lek is not None

        # Second page using lek
        items2, lek2 = table.query_page(
            pk="B", sk_prefix="R#", limit=2, exclusive_start_key=lek
        )
        assert len(items2) == 2

    @mock_aws
    def test_query_page_no_more_results(self):
        """query_page returns None token when no more results."""
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "C", "SK": "X#1"})

        items, lek = table.query_page(pk="C", sk_prefix="X#", limit=10)
        assert len(items) == 1
        assert lek is None

    @mock_aws
    def test_query_page_with_lsi3(self):
        """query_page works with LSI3 index for input_hash dedup."""
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "DS#1", "SK": "DS#REC#a", "lsi3sk": "abc123"})
        table.put_item({"PK": "DS#1", "SK": "DS#REC#b", "lsi3sk": "def456"})
        table.put_item({"PK": "DS#1", "SK": "DS#META", "lsi3sk": "myname"})

        # Query LSI3 for a specific input_hash — should find only the matching record
        from boto3.dynamodb.conditions import Attr
        items, lek = table.query_page(
            pk="DS#1",
            sk_prefix="abc123",
            index_name="lsi3",
            filter_expression=Attr("SK").begins_with("DS#REC#"),
        )
        assert len(items) == 1
        assert items[0]["SK"] == "DS#REC#a"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/unit/dynamodb/test_table.py::TestQueryPage -v`
Expected: FAIL with `AttributeError: 'DynamoDBTable' object has no attribute 'query_page'`

- [ ] **Step 7: Implement `query_page`**

Add to `src/mlflow_dynamodbstore/dynamodb/table.py` after `query`:

```python
def query_page(
    self,
    pk: str,
    sk_prefix: str | None = None,
    index_name: str | None = None,
    limit: int | None = None,
    scan_forward: bool = True,
    consistent: bool = False,
    exclusive_start_key: dict[str, Any] | None = None,
    filter_expression: ConditionBase | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Query a single page, returning (items, last_evaluated_key).

    Unlike query(), this does NOT auto-exhaust pagination.
    Returns the raw LastEvaluatedKey for caller-managed cursors.
    """
    if index_name:
        pk_attr, sk_attr = _INDEX_KEY_ATTRS[index_name]
    else:
        pk_attr, sk_attr = "PK", "SK"

    key_cond: ConditionBase = Key(pk_attr).eq(pk)
    if sk_prefix:
        key_cond = key_cond & Key(sk_attr).begins_with(sk_prefix)

    kwargs: dict[str, Any] = {
        "KeyConditionExpression": key_cond,
        "ScanIndexForward": scan_forward,
        "ConsistentRead": consistent,
    }
    if index_name:
        kwargs["IndexName"] = index_name
    if limit is not None:
        kwargs["Limit"] = limit
    if exclusive_start_key is not None:
        kwargs["ExclusiveStartKey"] = exclusive_start_key
    if filter_expression is not None:
        kwargs["FilterExpression"] = filter_expression

    response = self._table.query(**kwargs)
    items: list[dict[str, Any]] = response.get("Items", [])
    lek: dict[str, Any] | None = response.get("LastEvaluatedKey")
    return items, lek
```

- [ ] **Step 8: Run all table tests**

Run: `uv run pytest tests/unit/dynamodb/test_table.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/table.py tests/unit/dynamodb/test_table.py
git commit -m "feat(table): add batch_delete and query_page to DynamoDBTable"
```

---

## Task 3: Dataset Lifecycle — create, get, delete

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Create: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Write failing tests for create_dataset, get_dataset, delete_dataset**

Create `tests/unit/test_tracking_datasets.py`:

```python
"""Tests for Evaluation Dataset CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import hashlib
import json
import time

import pytest
from mlflow.entities import EvaluationDataset
from mlflow.exceptions import MlflowException

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_DATASET_PREFIX,
    SK_DATASET_META,
)


class TestCreateDataset:
    def test_create_returns_dataset(self, tracking_store):
        """create_dataset returns an EvaluationDataset with correct fields."""
        ds = tracking_store.create_dataset(name="my-dataset")
        assert isinstance(ds, EvaluationDataset)
        assert ds.name == "my-dataset"
        assert ds.dataset_id.startswith("eval_")
        assert ds.digest != ""
        assert ds.created_time > 0
        assert ds.last_update_time > 0

    def test_create_with_tags(self, tracking_store):
        """create_dataset stores tags."""
        ds = tracking_store.create_dataset(
            name="tagged-ds", tags={"env": "prod", "team": "ml"}
        )
        assert ds.tags == {"env": "prod", "team": "ml"}

    def test_create_with_experiment_ids(self, tracking_store):
        """create_dataset associates with experiments."""
        exp_id = tracking_store.create_experiment("test-exp")
        ds = tracking_store.create_dataset(
            name="linked-ds", experiment_ids=[exp_id]
        )
        exp_ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert exp_id in exp_ids

    def test_create_duplicate_name_raises(self, tracking_store):
        """create_dataset rejects duplicate names."""
        tracking_store.create_dataset(name="unique-name")
        with pytest.raises(MlflowException, match="already exists"):
            tracking_store.create_dataset(name="unique-name")


class TestGetDataset:
    def test_get_existing(self, tracking_store):
        """get_dataset retrieves a created dataset."""
        created = tracking_store.create_dataset(name="get-test")
        fetched = tracking_store.get_dataset(created.dataset_id)
        assert fetched.dataset_id == created.dataset_id
        assert fetched.name == "get-test"

    def test_get_nonexistent_raises(self, tracking_store):
        """get_dataset raises for unknown ID."""
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_dataset("eval_nonexistent")

    def test_get_includes_tags(self, tracking_store):
        """get_dataset returns tags."""
        created = tracking_store.create_dataset(
            name="tags-test", tags={"key": "val"}
        )
        fetched = tracking_store.get_dataset(created.dataset_id)
        assert fetched.tags == {"key": "val"}

    def test_get_lazy_loads_experiment_ids(self, tracking_store):
        """get_dataset returns entity whose experiment_ids property works."""
        exp_id = tracking_store.create_experiment("lazy-exp")
        ds = tracking_store.create_dataset(
            name="lazy-test", experiment_ids=[exp_id]
        )
        fetched = tracking_store.get_dataset(ds.dataset_id)
        # Access the lazy property — triggers get_dataset_experiment_ids
        assert exp_id in fetched.experiment_ids


class TestDeleteDataset:
    def test_delete_removes_dataset(self, tracking_store):
        """delete_dataset makes dataset unfetchable."""
        ds = tracking_store.create_dataset(name="delete-me")
        tracking_store.delete_dataset(ds.dataset_id)
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_dataset(ds.dataset_id)

    def test_delete_nonexistent_is_idempotent(self, tracking_store):
        """delete_dataset does not raise for unknown ID."""
        tracking_store.delete_dataset("eval_nonexistent")  # no error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: FAIL with `AttributeError: 'DynamoDBTrackingStore' object has no attribute 'create_dataset'`

- [ ] **Step 3: Implement create_dataset, get_dataset, delete_dataset**

Add imports and methods to `src/mlflow_dynamodbstore/tracking_store.py`. The implementation should follow the spec's DynamoDB item design. Key imports to add:

```python
from mlflow.entities import EvaluationDataset
from mlflow.entities.dataset_record import DatasetRecord
from mlflow.store.entities.paged_list import PagedList
```

New schema imports:

```python
from mlflow_dynamodbstore.dynamodb.schema import (
    PK_DATASET_PREFIX,
    SK_DATASET_META,
    SK_DATASET_TAG_PREFIX,
    SK_DATASET_RECORD_PREFIX,
    SK_DATASET_EXP_PREFIX,
    GSI1_DS_EXP_PREFIX,
    GSI2_DS_LIST_PREFIX,
    GSI3_DS_NAME_PREFIX,
)
```

Implement `_compute_dataset_digest(name, last_update_time)` helper, then `create_dataset`, `get_dataset`, `delete_dataset` following the spec's Store Methods section.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_datasets.py
git commit -m "feat(datasets): implement create_dataset, get_dataset, delete_dataset"
```

---

## Task 4: Dataset Search

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Write failing tests for search_datasets**

Add to `tests/unit/test_tracking_datasets.py`:

```python
class TestSearchDatasets:
    def test_search_all_in_workspace(self, tracking_store):
        """search_datasets with no filter returns all datasets."""
        tracking_store.create_dataset(name="ds-alpha")
        tracking_store.create_dataset(name="ds-beta")
        results = tracking_store.search_datasets()
        assert len(results) == 2

    def test_search_by_experiment_id(self, tracking_store):
        """search_datasets filters by experiment_ids."""
        exp_id = tracking_store.create_experiment("exp-search")
        ds1 = tracking_store.create_dataset(
            name="linked", experiment_ids=[exp_id]
        )
        tracking_store.create_dataset(name="unlinked")
        results = tracking_store.search_datasets(experiment_ids=[exp_id])
        assert len(results) == 1
        assert results[0].dataset_id == ds1.dataset_id

    def test_search_by_name_prefix_filter(self, tracking_store):
        """search_datasets supports name LIKE 'prefix%' filter."""
        tracking_store.create_dataset(name="prod-dataset")
        tracking_store.create_dataset(name="dev-dataset")
        results = tracking_store.search_datasets(
            filter_string="name LIKE 'prod%'"
        )
        assert len(results) == 1
        assert results[0].name == "prod-dataset"

    def test_search_by_name_substring_filter(self, tracking_store):
        """search_datasets supports name LIKE '%pattern%' (substring)."""
        tracking_store.create_dataset(name="my-prod-dataset")
        tracking_store.create_dataset(name="my-dev-dataset")
        results = tracking_store.search_datasets(
            filter_string="name LIKE '%prod%'"
        )
        assert len(results) == 1
        assert results[0].name == "my-prod-dataset"

    def test_search_order_by_name(self, tracking_store):
        """search_datasets supports order_by name."""
        tracking_store.create_dataset(name="charlie")
        tracking_store.create_dataset(name="alpha")
        tracking_store.create_dataset(name="bravo")
        results = tracking_store.search_datasets(order_by=["name ASC"])
        names = [r.name for r in results]
        assert names == ["alpha", "bravo", "charlie"]

    def test_search_pagination(self, tracking_store):
        """search_datasets supports max_results and page_token."""
        for i in range(5):
            tracking_store.create_dataset(name=f"page-ds-{i}")
        page1 = tracking_store.search_datasets(max_results=2)
        assert len(page1) == 2
        assert page1.token is not None

        page2 = tracking_store.search_datasets(
            max_results=2, page_token=page1.token
        )
        assert len(page2) == 2

    def test_search_returns_paged_list(self, tracking_store):
        """search_datasets returns PagedList."""
        tracking_store.create_dataset(name="paged")
        results = tracking_store.search_datasets()
        assert isinstance(results, PagedList)

    def test_search_empty(self, tracking_store):
        """search_datasets with no datasets returns empty list."""
        results = tracking_store.search_datasets()
        assert len(results) == 0
```

Add import at top of file: `from mlflow.store.entities.paged_list import PagedList`

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_tracking_datasets.py::TestSearchDatasets -v`
Expected: FAIL

- [ ] **Step 3: Implement search_datasets**

Follow spec: query GSI1 (for experiment_ids) or GSI2 (all in workspace), batch-get META items, apply filter_string and order_by in-memory, return `PagedList`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_datasets.py
git commit -m "feat(datasets): implement search_datasets with GSI1/GSI2 queries"
```

---

## Task 5: Dataset Tags

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Write failing tests for tag operations**

```python
class TestDatasetTags:
    def test_set_tags(self, tracking_store):
        """set_dataset_tags upserts tags."""
        ds = tracking_store.create_dataset(name="tag-crud")
        tracking_store.set_dataset_tags(ds.dataset_id, {"k1": "v1", "k2": "v2"})
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert fetched.tags["k1"] == "v1"
        assert fetched.tags["k2"] == "v2"

    def test_set_tags_overwrites(self, tracking_store):
        """set_dataset_tags overwrites existing tags."""
        ds = tracking_store.create_dataset(name="overwrite", tags={"k": "old"})
        tracking_store.set_dataset_tags(ds.dataset_id, {"k": "new"})
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert fetched.tags["k"] == "new"

    def test_delete_tag(self, tracking_store):
        """delete_dataset_tag removes a tag."""
        ds = tracking_store.create_dataset(name="del-tag", tags={"k": "v"})
        tracking_store.delete_dataset_tag(ds.dataset_id, "k")
        fetched = tracking_store.get_dataset(ds.dataset_id)
        assert "k" not in fetched.tags
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_tracking_datasets.py::TestDatasetTags -v`

- [ ] **Step 3: Implement set_dataset_tags and delete_dataset_tag**

Follow spec: write/delete tag items, update denormalized tags on META, update digest.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_datasets.py
git commit -m "feat(datasets): implement set_dataset_tags, delete_dataset_tag"
```

---

## Task 6: Experiment Association

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Write failing tests**

```python
class TestDatasetExperimentAssociation:
    def test_get_experiment_ids(self, tracking_store):
        """get_dataset_experiment_ids returns linked experiments."""
        exp1 = tracking_store.create_experiment("exp-1")
        exp2 = tracking_store.create_experiment("exp-2")
        ds = tracking_store.create_dataset(
            name="multi-exp", experiment_ids=[exp1, exp2]
        )
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert set(ids) == {exp1, exp2}

    def test_add_dataset_to_experiments(self, tracking_store):
        """add_dataset_to_experiments links dataset to new experiments."""
        exp1 = tracking_store.create_experiment("add-1")
        exp2 = tracking_store.create_experiment("add-2")
        ds = tracking_store.create_dataset(name="add-test")
        tracking_store.add_dataset_to_experiments(ds.dataset_id, [exp1, exp2])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert set(ids) == {exp1, exp2}

    def test_remove_dataset_from_experiments(self, tracking_store):
        """remove_dataset_from_experiments unlinks experiments."""
        exp1 = tracking_store.create_experiment("rm-1")
        exp2 = tracking_store.create_experiment("rm-2")
        ds = tracking_store.create_dataset(
            name="rm-test", experiment_ids=[exp1, exp2]
        )
        tracking_store.remove_dataset_from_experiments(ds.dataset_id, [exp1])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert set(ids) == {exp2}

    def test_add_idempotent(self, tracking_store):
        """add_dataset_to_experiments twice does not create duplicates."""
        exp1 = tracking_store.create_experiment("idem-1")
        ds = tracking_store.create_dataset(name="idem-test")
        tracking_store.add_dataset_to_experiments(ds.dataset_id, [exp1])
        tracking_store.add_dataset_to_experiments(ds.dataset_id, [exp1])
        ids = tracking_store.get_dataset_experiment_ids(ds.dataset_id)
        assert ids.count(exp1) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_tracking_datasets.py::TestDatasetExperimentAssociation -v`

- [ ] **Step 3: Implement get_dataset_experiment_ids, add_dataset_to_experiments, remove_dataset_from_experiments**

Follow spec: write/delete `DS#EXP#<exp_id>` items with GSI1 projections.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_datasets.py
git commit -m "feat(datasets): implement experiment association (add/remove/get)"
```

---

## Task 7: Record Upsert and Load

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Write failing tests for upsert and load**

```python
class TestDatasetRecords:
    def test_upsert_inserts_new_records(self, tracking_store):
        """upsert_dataset_records inserts new records."""
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
        """upsert_dataset_records updates records with same inputs."""
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
        """upsert_dataset_records updates profile num_records."""
        ds = tracking_store.create_dataset(name="profile-test")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": "a"}}, {"inputs": {"q": "b"}}],
        )
        fetched = tracking_store.get_dataset(ds.dataset_id)
        profile = json.loads(fetched.profile) if fetched.profile else {}
        assert profile.get("num_records") == 2

    def test_load_records_paginated(self, tracking_store):
        """_load_dataset_records supports pagination."""
        ds = tracking_store.create_dataset(name="paginate-test")
        tracking_store.upsert_dataset_records(
            ds.dataset_id,
            [{"inputs": {"q": f"item-{i}"}} for i in range(5)],
        )
        records, token = tracking_store._load_dataset_records(
            ds.dataset_id, max_results=2
        )
        assert len(records) == 2
        assert token is not None

        records2, token2 = tracking_store._load_dataset_records(
            ds.dataset_id, max_results=2, page_token=token
        )
        assert len(records2) == 2

    def test_delete_records(self, tracking_store):
        """delete_dataset_records removes specific records."""
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
```

Add `import json` at top of test file.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_tracking_datasets.py::TestDatasetRecords -v`

- [ ] **Step 3: Implement upsert_dataset_records, _load_dataset_records, delete_dataset_records**

Key implementation details:
- `input_hash`: `hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",",":")).encode()).hexdigest()[:8]` — uses `sort_keys=True` for deterministic key order, compact separators for canonical form. Produces a JSON object (not list-of-lists).
- Dedup: query LSI3 with `lsi3sk=<input_hash>`, `FilterExpression: SK begins_with DS#REC#`
- Record items: populate `lsi1sk=created_time`, `lsi2sk=last_update_time`, `lsi3sk=input_hash`
- `_load_dataset_records`: use `query_page` with native LEK pagination
- `delete_dataset_records`: use `batch_delete`
- After upsert/delete: recompute profile `{num_records: N}` and update META

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_datasets.py
git commit -m "feat(datasets): implement record upsert, load, delete with LSI3 dedup"
```

---

## Task 8: Legacy V2 _search_datasets

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Write failing test for _search_datasets**

```python
class TestSearchDatasetsLegacy:
    def test_search_datasets_v2_empty(self, tracking_store):
        """_search_datasets returns empty list when no datasets logged."""
        exp_id = tracking_store.create_experiment("v2-empty")
        results = tracking_store._search_datasets([exp_id])
        assert results == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_tracking_datasets.py::TestSearchDatasetsLegacy -v`

- [ ] **Step 3: Verify DatasetSummary import and implement _search_datasets**

Verify import path first: `python -c "from mlflow.protos.service_pb2 import DatasetSummary; print('OK')"`

Query existing `D#` and `DLINK#` items under `EXP#<exp_id>`. Return `DatasetSummary` objects. Import: `from mlflow.protos.service_pb2 import DatasetSummary`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_tracking_datasets.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_datasets.py
git commit -m "feat(datasets): implement legacy _search_datasets for V2 compatibility"
```

---

## Task 9: Full Unit Test Suite

**Files:**
- Modify: `tests/unit/test_tracking_datasets.py`

- [ ] **Step 1: Run full unit test suite to check for regressions**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: ALL PASS (559+ existing + new dataset tests)

- [ ] **Step 2: Run coverage check on patch**

Run: `uv run pytest tests/unit/ --cov=mlflow_dynamodbstore --cov-report=term-missing | grep tracking_store`
Expected: No new uncovered lines from the dataset implementation.

- [ ] **Step 3: Fix any coverage gaps**

Add targeted tests for any uncovered branches.

- [ ] **Step 4: Commit if any changes**

```bash
git add tests/unit/test_tracking_datasets.py
git commit -m "test(datasets): achieve 100% patch coverage for dataset methods"
```

---

## Task 10: E2E Tests

**Files:**
- Create: `tests/e2e/test_datasets.py`
- Modify: `tests/e2e/test_traces.py`

- [ ] **Step 1: Create E2E test file**

Create `tests/e2e/test_datasets.py`:

```python
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
```

- [ ] **Step 2: Remove xfail from demo evaluation test**

In `tests/e2e/test_traces.py`, remove the `@pytest.mark.xfail` decorator from `test_generate_demo_evaluation`.

- [ ] **Step 3: Run E2E tests**

Run: `uv run pytest tests/e2e/test_datasets.py tests/e2e/test_traces.py::TestDemoGeneration -v`
Expected: ALL PASS (demo evaluation may still xfail if logged_models/scorers are needed by the demo — verify)

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/unit/ tests/integration/ tests/e2e/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_datasets.py tests/e2e/test_traces.py
git commit -m "test(datasets): add e2e tests for dataset CRUD and demo evaluation"
```

---

## Task 11: Final Verification and PR

- [ ] **Step 1: Run linting**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: PASS

- [ ] **Step 2: Run type checking**

Run: `uv run mypy src/mlflow_dynamodbstore/tracking_store.py`
Expected: PASS

- [ ] **Step 3: Run full test suite one final time**

Run: `uv run pytest tests/unit/ tests/integration/ tests/e2e/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 4: Create PR**

```bash
git push -u origin feat/evaluation-datasets
gh pr create --title "feat: implement evaluation datasets in DynamoDB store" --body "..."
```
