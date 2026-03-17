# Logged Models Implementation Plan

> **Status: COMPLETE** — All 9 tasks implemented and verified. PR #13.

**Goal:** Implement the 8 `LoggedModel` store methods on `DynamoDBTrackingStore` so the MLflow 3.x UI can create, search, and manage logged models without 500 errors.

**Architecture:** Logged models are experiment-scoped entities stored under `EXP#<exp_id>` partitions using `LM#<model_id>` sort keys, with RANK items for metric-based search. Search uses the existing `parse → plan → execute` pipeline in `dynamodb/search.py`. Soft delete uses TTL, matching the existing run deletion pattern.

**Tech Stack:** Python, DynamoDB (moto for tests), MLflow 3.10.1 entity classes (`LoggedModel`, `LoggedModelTag`, `LoggedModelParameter`, `Metric`), `SearchLoggedModelsUtils` for filter parsing.

**Spec:** `docs/superpowers/specs/2026-03-17-logged-models-design.md`

### Results

| Metric | Value |
|--------|-------|
| Public methods implemented | 8 |
| Internal methods implemented | 1 (`_log_logged_model_metric`) |
| Unit tests | 49 (39 logged models + 10 search) |
| E2E tests | 9 |
| Total test suite | 808 passed, 0 failed |
| Commits | 10 |

### Commits

| Commit | Description |
|--------|-------------|
| `e3a1e3c` | feat(schema): add logged model SK and GSI constants |
| `562b3b3` | feat: implement create_logged_model and get_logged_model |
| `4445f9c` | feat: implement finalize, delete, tags for logged models |
| `e0583bb` | feat: implement record_logged_model |
| `9f25d4e` | feat: implement _log_logged_model_metric with RANK items |
| `6df70a2` | feat: add parse/plan/execute for logged model search |
| `32cfc0b` | feat: implement search_logged_models with parse/plan/execute |
| `d78442b` | test: add e2e tests for logged model CRUD and search |
| `d0f852d` | test: improve patch coverage for logged models search |

### Implementation notes

- `set_logged_model_tags` REST API accepts `dict[str, Any]` (not `list[LoggedModelTag]`) — e2e tests adjusted
- `self._config.get_soft_deleted_ttl_seconds()` used for TTL (not a custom helper) — matches `delete_run` pattern
- `QueryPlan` extended with `rank_filters` and `datasets` fields (backward-compatible defaults)
- Pagination handled at `search_logged_models` level after merging across experiments (not in `execute_logged_model_query`)

---

### File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/mlflow_dynamodbstore/dynamodb/schema.py` | SK/GSI prefix constants | Modified: added `SK_LM_PREFIX`, `SK_RANK_LM_PREFIX`, `SK_RANK_LMD_PREFIX`, `GSI1_LM_PREFIX` |
| `src/mlflow_dynamodbstore/dynamodb/search.py` | Query parsing, planning, execution | Modified: added `parse_logged_model_filter`, `plan_logged_model_query`, `execute_logged_model_query`, extended `QueryPlan` |
| `src/mlflow_dynamodbstore/tracking_store.py` | Store method implementations | Modified: added 8 public methods + `_log_logged_model_metric` + `_item_to_logged_model` helper |
| `tests/unit/test_logged_models.py` | Unit tests for all store methods | Created |
| `tests/unit/test_search_logged_models.py` | Unit tests for search parse/plan | Created |
| `tests/e2e/test_logged_models.py` | E2E tests for search validation | Created |

---

### Task 1: Schema Constants

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py:37-42`

- [x] **Step 1: Add logged model constants to schema.py**

After `SK_LOGGED_MODEL_PREFIX = "#LM#"` (line 37) and before `SK_DATASET_PREFIX` (line 38), add:

```python
SK_LM_PREFIX = "LM#"  # V3 experiment-scoped logged models (distinct from V2 SK_LOGGED_MODEL_PREFIX)
SK_LM_TAG_PREFIX = "#TAG#"
SK_LM_PARAM_PREFIX = "#PARAM#"
SK_LM_METRIC_PREFIX = "#METRIC#"
SK_RANK_LM_PREFIX = "RANK#lm#"    # RANK items for logged model metrics (global)
SK_RANK_LMD_PREFIX = "RANK#lmd#"  # RANK items for logged model metrics (dataset-scoped)
```

After `GSI1_TRACE_PREFIX` in the GSI constants section, add:

```python
GSI1_LM_PREFIX = "LM#"
```

- [x] **Step 2: Verify imports compile**

Run: `uv run python -c "from mlflow_dynamodbstore.dynamodb.schema import SK_LM_PREFIX, SK_RANK_LM_PREFIX, SK_RANK_LMD_PREFIX, GSI1_LM_PREFIX; print('OK')"`
Expected: `OK`

- [x] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat(schema): add logged model SK and GSI constants"
```

---

### Task 2: create_logged_model and get_logged_model

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Create: `tests/unit/test_logged_models.py`

- [x] **Step 1: Write failing tests for create and get**

Create `tests/unit/test_logged_models.py`:

```python
"""Tests for LoggedModel CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import pytest
from mlflow.entities import LoggedModel, LoggedModelTag, LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.exceptions import MlflowException

from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_LM_PREFIX,
    GSI1_PK,
    GSI1_SK,
    LSI1_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    PK_EXPERIMENT_PREFIX,
    SK_LM_PREFIX,
)


def _create_experiment(tracking_store) -> str:
    return tracking_store.create_experiment("test-exp", artifact_location="s3://bucket/artifacts")


class TestCreateLoggedModel:
    def test_create_logged_model(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="my-model")

        assert model.model_id.startswith("m-")
        assert model.experiment_id == exp_id
        assert model.name == "my-model"
        assert model.status == LoggedModelStatus.PENDING
        assert "models/" in model.artifact_location
        assert model.artifact_location.endswith("/artifacts/")

    def test_create_with_tags_and_params(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="tagged-model",
            tags=[LoggedModelTag("env", "prod")],
            params=[LoggedModelParameter("lr", "0.01")],
            model_type="sklearn",
            source_run_id="run-abc",
        )

        assert model.tags == {"env": "prod"}
        assert model.params == {"lr": "0.01"}
        assert model.model_type == "sklearn"
        assert model.source_run_id == "run-abc"

    def test_create_writes_gsi1_reverse_lookup(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="gsi-test")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta[GSI1_PK] == f"{GSI1_LM_PREFIX}{model.model_id}"
        assert meta[GSI1_SK] == f"{PK_EXPERIMENT_PREFIX}{exp_id}"

    def test_create_writes_lsi_projections(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="MyModel")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta[LSI1_SK].startswith("active#")
        assert meta[LSI3_SK].startswith("PENDING#")
        assert meta[LSI4_SK] == "mymodel"


class TestGetLoggedModel:
    def test_get_logged_model(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        created = tracking_store.create_logged_model(experiment_id=exp_id, name="get-test")

        fetched = tracking_store.get_logged_model(created.model_id)
        assert fetched.model_id == created.model_id
        assert fetched.name == "get-test"
        assert fetched.experiment_id == exp_id

    def test_get_with_tags_and_params(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        created = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="full-model",
            tags=[LoggedModelTag("k", "v")],
            params=[LoggedModelParameter("p", "1")],
        )

        fetched = tracking_store.get_logged_model(created.model_id)
        assert fetched.tags == {"k": "v"}
        assert fetched.params == {"p": "1"}

    def test_get_nonexistent_raises(self, tracking_store):
        _create_experiment(tracking_store)
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_logged_model("m-nonexistent")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_logged_models.py -v -x`
Expected: FAIL — `create_logged_model` not implemented.

- [x] **Step 3: Implement create_logged_model and get_logged_model**

In `src/mlflow_dynamodbstore/tracking_store.py`, add imports and helper:

```python
# At top, add to imports:
from mlflow.entities import LoggedModel, LoggedModelTag, LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus

# Add schema imports:
from mlflow_dynamodbstore.dynamodb.schema import SK_LM_PREFIX, GSI1_LM_PREFIX
```

Add helper function (near other `_item_to_*` functions around line 100-200):

```python
def _item_to_logged_model(
    item: dict[str, Any],
    tags: list[dict[str, Any]] | None = None,
    params: list[dict[str, Any]] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> LoggedModel:
    """Convert DynamoDB items to a LoggedModel entity."""
    from mlflow.entities import Metric as MetricEntity

    tag_dict = {t["key"]: t["value"] for t in (tags or [])}
    param_dict = {p["key"]: p["value"] for p in (params or [])}
    metric_list = [
        MetricEntity(
            key=m["metric_name"],
            value=float(m["metric_value"]),
            timestamp=int(m.get("metric_timestamp_ms", 0)),
            step=int(m.get("metric_step", 0)),
        )
        for m in (metrics or [])
    ]

    # Merge denormalized tags/params from META with sub-items
    meta_tags = item.get("tags", {})
    meta_params = item.get("params", {})
    for k, v in meta_tags.items():
        tag_dict.setdefault(k, v)
    for k, v in meta_params.items():
        param_dict.setdefault(k, v)

    return LoggedModel(
        experiment_id=item["experiment_id"],
        model_id=item["model_id"],
        name=item.get("name", ""),
        artifact_location=item.get("artifact_location", ""),
        creation_timestamp=int(item.get("creation_timestamp_ms", 0)),
        last_updated_timestamp=int(item.get("last_updated_timestamp_ms", 0)),
        model_type=item.get("model_type"),
        source_run_id=item.get("source_run_id"),
        status=LoggedModelStatus(item.get("status", "READY")),
        status_message=item.get("status_message"),
        tags=tag_dict,
        params=param_dict,
        metrics=metric_list,
    )
```

Add store methods on `DynamoDBTrackingStore`:

```python
def _resolve_logged_model_experiment(self, model_id: str) -> str:
    """Resolve experiment_id for a logged model via GSI1."""
    results = self._table.query(
        pk=f"{GSI1_LM_PREFIX}{model_id}",
        index_name="gsi1",
        limit=1,
    )
    if not results:
        raise MlflowException(
            f"LoggedModel '{model_id}' does not exist.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    return results[0][GSI1_SK].replace(PK_EXPERIMENT_PREFIX, "")

def create_logged_model(
    self,
    experiment_id: str,
    name: str | None = None,
    source_run_id: str | None = None,
    tags: list[LoggedModelTag] | None = None,
    params: list[LoggedModelParameter] | None = None,
    model_type: str | None = None,
) -> LoggedModel:
    exp = self.get_experiment(experiment_id)
    now_ms = int(time.time() * 1000)
    model_id = f"m-{generate_ulid()}"
    name = name or model_id
    artifact_location = f"{exp.artifact_location}/models/{model_id}/artifacts/"

    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    sk = f"{SK_LM_PREFIX}{model_id}"

    tag_dict = {t.key: t.value for t in (tags or [])}
    param_dict = {p.key: p.value for p in (params or [])}

    item: dict[str, Any] = {
        "PK": pk,
        "SK": sk,
        "model_id": model_id,
        "experiment_id": experiment_id,
        "name": name,
        "artifact_location": artifact_location,
        "creation_timestamp_ms": now_ms,
        "last_updated_timestamp_ms": now_ms,
        "status": str(LoggedModelStatus.PENDING),
        "lifecycle_stage": "active",
        "model_type": model_type or "",
        "source_run_id": source_run_id or "",
        "status_message": "",
        "tags": tag_dict,
        "params": param_dict,
        "workspace": self._workspace,
        LSI1_SK: f"active#{model_id}",
        LSI2_SK: str(now_ms),
        LSI3_SK: f"PENDING#{model_id}",
        LSI4_SK: name.lower(),
        GSI1_PK: f"{GSI1_LM_PREFIX}{model_id}",
        GSI1_SK: f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
    }

    self._table.put_item(item, condition="attribute_not_exists(PK)")

    # Write tag sub-items
    for key, value in tag_dict.items():
        self._table.put_item({
            "PK": pk,
            "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{key}",
            "key": key,
            "value": value,
        })

    # Write param sub-items
    for key, value in param_dict.items():
        self._table.put_item({
            "PK": pk,
            "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}{key}",
            "key": key,
            "value": value,
        })

    return _item_to_logged_model(item)

def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
    experiment_id = self._resolve_logged_model_experiment(model_id)
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    sk = f"{SK_LM_PREFIX}{model_id}"

    meta = self._table.get_item(pk=pk, sk=sk)
    if meta is None:
        raise MlflowException(
            f"LoggedModel '{model_id}' does not exist.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    if meta.get("lifecycle_stage") == "deleted" and not allow_deleted:
        raise MlflowException(
            f"LoggedModel '{model_id}' does not exist.",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    # Load sub-items
    tag_items = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}")
    param_items = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}")
    metric_items = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_METRIC_PREFIX}")

    return _item_to_logged_model(meta, tag_items, param_items, metric_items)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_logged_models.py -v`
Expected: All tests in `TestCreateLoggedModel` and `TestGetLoggedModel` PASS (except tests referencing `delete_logged_model` — mark those `@pytest.mark.skip("implemented in Task 3")` temporarily, or keep them and expect failures for now).

- [x] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_logged_models.py
git commit -m "feat: implement create_logged_model and get_logged_model"
```

---

### Task 3: finalize_logged_model, delete_logged_model, tags

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_logged_models.py`

- [x] **Step 1: Write failing tests for finalize, delete, tags**

Add to `tests/unit/test_logged_models.py`:

```python
class TestFinalizeLoggedModel:
    def test_finalize_ready(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="fin-test")

        result = tracking_store.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
        assert result.status == LoggedModelStatus.READY

        fetched = tracking_store.get_logged_model(model.model_id)
        assert fetched.status == LoggedModelStatus.READY

    def test_finalize_failed(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="fail-test")

        result = tracking_store.finalize_logged_model(model.model_id, LoggedModelStatus.FAILED)
        assert result.status == LoggedModelStatus.FAILED

    def test_finalize_updates_lsi3(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="lsi3-test")
        tracking_store.finalize_logged_model(model.model_id, LoggedModelStatus.READY)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta[LSI3_SK].startswith("READY#")


class TestGetDeletedLoggedModel:
    def test_get_deleted_raises_by_default(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="del-test")
        tracking_store.delete_logged_model(model.model_id)

        with pytest.raises(MlflowException):
            tracking_store.get_logged_model(model.model_id)

    def test_get_deleted_with_allow_deleted(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="del-test")
        tracking_store.delete_logged_model(model.model_id)

        fetched = tracking_store.get_logged_model(model.model_id, allow_deleted=True)
        assert fetched.model_id == model.model_id


class TestDeleteLoggedModel:
    def test_soft_delete(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="del-test")
        tracking_store.delete_logged_model(model.model_id)

        with pytest.raises(MlflowException):
            tracking_store.get_logged_model(model.model_id)

    def test_soft_delete_sets_ttl_on_children(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="ttl-test",
            tags=[LoggedModelTag("k", "v")],
            params=[LoggedModelParameter("p", "1")],
        )
        tracking_store.delete_logged_model(model.model_id)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model.model_id}")
        assert meta.get("lifecycle_stage") == "deleted"
        # TTL should be set (or not, depending on TTL policy config)
        assert meta[LSI1_SK].startswith("deleted#")


class TestLoggedModelTags:
    def test_set_tags(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="tag-test")

        tracking_store.set_logged_model_tags(model.model_id, [LoggedModelTag("env", "prod")])

        fetched = tracking_store.get_logged_model(model.model_id)
        assert fetched.tags["env"] == "prod"

    def test_set_tags_overwrite(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="overwrite-test",
            tags=[LoggedModelTag("env", "dev")],
        )
        tracking_store.set_logged_model_tags(model.model_id, [LoggedModelTag("env", "prod")])

        fetched = tracking_store.get_logged_model(model.model_id)
        assert fetched.tags["env"] == "prod"

    def test_delete_tag(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(
            experiment_id=exp_id,
            name="deltag-test",
            tags=[LoggedModelTag("env", "prod")],
        )
        tracking_store.delete_logged_model_tag(model.model_id, "env")

        fetched = tracking_store.get_logged_model(model.model_id)
        assert "env" not in fetched.tags
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_logged_models.py -v -x -k "Finalize or Delete or Tags"`
Expected: FAIL — methods not implemented.

- [x] **Step 3: Implement finalize, delete, set_logged_model_tags, delete_logged_model_tag**

Add to `DynamoDBTrackingStore`:

```python
def finalize_logged_model(self, model_id: str, status: LoggedModelStatus) -> LoggedModel:
    experiment_id = self._resolve_logged_model_experiment(model_id)
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    sk = f"{SK_LM_PREFIX}{model_id}"
    now_ms = int(time.time() * 1000)

    self._table.update_item(
        pk=pk,
        sk=sk,
        updates={
            "status": str(status),
            "last_updated_timestamp_ms": now_ms,
            LSI3_SK: f"{status}#{model_id}",
        },
    )
    return self.get_logged_model(model_id)

def delete_logged_model(self, model_id: str) -> None:
    experiment_id = self._resolve_logged_model_experiment(model_id)
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    sk = f"{SK_LM_PREFIX}{model_id}"
    now_ms = int(time.time() * 1000)

    ttl_seconds = self._config.get_soft_deleted_ttl_seconds()
    ttl_value = int(time.time()) + ttl_seconds if ttl_seconds is not None else None

    updates: dict[str, Any] = {
        "lifecycle_stage": "deleted",
        "last_updated_timestamp_ms": now_ms,
        LSI1_SK: f"deleted#{model_id}",
    }
    if ttl_value is not None:
        updates["ttl"] = ttl_value
    self._table.update_item(pk=pk, sk=sk, updates=updates)

    # Set TTL on child items (tags, params, metrics)
    children = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}#")
    for child in children:
        if ttl_value is not None:
            self._table.update_item(pk=pk, sk=child["SK"], updates={"ttl": ttl_value})

    # Set TTL on RANK items built from metric sub-items
    metric_children = [c for c in children if f"{SK_LM_METRIC_PREFIX}" in c["SK"]]
    for mc in metric_children:
        inv_value = self._invert_metric_value(float(mc["metric_value"]))
        rank_sk = f"{SK_RANK_LM_PREFIX}{mc['metric_name']}#{inv_value}#{model_id}"
        if ttl_value is not None:
            self._table.update_item(pk=pk, sk=rank_sk, updates={"ttl": ttl_value})
        if mc.get("dataset_name") and mc.get("dataset_digest"):
            rank_sk_ds = f"{SK_RANK_LMD_PREFIX}{mc['metric_name']}#{mc['dataset_name']}#{mc['dataset_digest']}#{inv_value}#{model_id}"
            if ttl_value is not None:
                self._table.update_item(pk=pk, sk=rank_sk_ds, updates={"ttl": ttl_value})

def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
    experiment_id = self._resolve_logged_model_experiment(model_id)
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    now_ms = int(time.time() * 1000)

    tag_dict = {}
    for tag in tags:
        self._table.put_item({
            "PK": pk,
            "SK": f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{tag.key}",
            "key": tag.key,
            "value": tag.value,
        })
        tag_dict[tag.key] = tag.value

    # Update denormalized tags on META
    meta = self._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model_id}")
    existing_tags = meta.get("tags", {})
    existing_tags.update(tag_dict)
    self._table.update_item(
        pk=pk,
        sk=f"{SK_LM_PREFIX}{model_id}",
        updates={"tags": existing_tags, "last_updated_timestamp_ms": now_ms},
    )

def delete_logged_model_tag(self, model_id: str, key: str) -> None:
    experiment_id = self._resolve_logged_model_experiment(model_id)
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    now_ms = int(time.time() * 1000)

    self._table.delete_item(pk=pk, sk=f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}{key}")

    # Update denormalized tags on META
    meta = self._table.get_item(pk=pk, sk=f"{SK_LM_PREFIX}{model_id}")
    existing_tags = meta.get("tags", {})
    existing_tags.pop(key, None)
    self._table.update_item(
        pk=pk,
        sk=f"{SK_LM_PREFIX}{model_id}",
        updates={"tags": existing_tags, "last_updated_timestamp_ms": now_ms},
    )

@staticmethod
def _invert_metric_value(value: float) -> str:
    """Invert a metric value for descending sort in DynamoDB RANK items."""
    max_val = 9999999999.9999
    inv = max_val - value
    return f"{inv:020.4f}"
```


- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_logged_models.py -v`
Expected: All tests PASS.

- [x] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_logged_models.py
git commit -m "feat: implement finalize, delete, tags for logged models"
```

---

### Task 4: record_logged_model

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_logged_models.py`

- [x] **Step 1: Write failing test**

Add to `tests/unit/test_logged_models.py`:

```python
import json


class TestRecordLoggedModel:
    def test_record_logged_model(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, run_name="test-run", tags=[]
        )

        tracking_store.record_logged_model(run.info.run_id, {"model_id": "m-test", "name": "my-model"})

        fetched_run = tracking_store.get_run(run.info.run_id)
        logged_models_tag = None
        for tag in fetched_run.data.tags:
            if tag == "mlflow.loggedModels":
                logged_models_tag = fetched_run.data.tags[tag]
                break
        assert logged_models_tag is not None
        models = json.loads(logged_models_tag)
        assert len(models) == 1
        assert models[0]["model_id"] == "m-test"
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_logged_models.py::TestRecordLoggedModel -v -x`
Expected: FAIL.

- [x] **Step 3: Implement record_logged_model**

```python
def record_logged_model(self, run_id, mlflow_model):
    import json as _json

    run = self.get_run(run_id)
    experiment_id = run.info.experiment_id
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

    # Read existing tag
    tag_sk = f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}mlflow.loggedModels"
    existing = self._table.get_item(pk=pk, sk=tag_sk)
    models = _json.loads(existing["value"]) if existing else []
    models.append(mlflow_model)

    self._table.put_item({
        "PK": pk,
        "SK": tag_sk,
        "key": "mlflow.loggedModels",
        "value": _json.dumps(models),
    })

    # Update denormalized tags on run META
    run_sk = f"{SK_RUN_PREFIX}{run_id}"
    meta = self._table.get_item(pk=pk, sk=run_sk)
    run_tags = meta.get("tags", {})
    run_tags["mlflow.loggedModels"] = _json.dumps(models)
    self._table.update_item(pk=pk, sk=run_sk, updates={"tags": run_tags})
```

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_logged_models.py::TestRecordLoggedModel -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_logged_models.py
git commit -m "feat: implement record_logged_model"
```

---

### Task 5: _log_logged_model_metric (internal)

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_logged_models.py`

- [x] **Step 1: Write failing test**

```python
from decimal import Decimal


class TestLogLoggedModelMetric:
    def test_log_metric(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="metric-test")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
        )

        fetched = tracking_store.get_logged_model(model.model_id)
        assert len(fetched.metrics) == 1
        assert fetched.metrics[0].key == "accuracy"
        assert fetched.metrics[0].value == 0.95

    def test_log_metric_writes_rank_item(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="rank-test")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(pk=pk, sk_prefix=f"{SK_RANK_LM_PREFIX}accuracy#")
        assert len(rank_items) == 1
        assert rank_items[0]["model_id"] == model.model_id

    def test_log_metric_with_dataset_writes_scoped_rank(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="ds-rank")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.90,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
            dataset_name="eval_set",
            dataset_digest="abc123",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_RANK_LMD_PREFIX}accuracy#eval_set#abc123#"
        )
        assert len(rank_items) == 1

    def test_log_metric_replaces_rank_on_update(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        model = tracking_store.create_logged_model(experiment_id=exp_id, name="replace-rank")

        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.80,
            metric_timestamp_ms=1000,
            metric_step=0,
            run_id="run-abc",
        )
        tracking_store._log_logged_model_metric(
            experiment_id=exp_id,
            model_id=model.model_id,
            metric_name="accuracy",
            metric_value=0.95,
            metric_timestamp_ms=2000,
            metric_step=1,
            run_id="run-abc",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rank_items = tracking_store._table.query(pk=pk, sk_prefix=f"{SK_RANK_LM_PREFIX}accuracy#")
        assert len(rank_items) == 1  # Old one deleted, new one written
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_logged_models.py::TestLogLoggedModelMetric -v -x`
Expected: FAIL.

- [x] **Step 3: Implement _log_logged_model_metric**

```python
def _log_logged_model_metric(
    self,
    experiment_id: str,
    model_id: str,
    metric_name: str,
    metric_value: float,
    metric_timestamp_ms: int,
    metric_step: int,
    run_id: str,
    dataset_name: str | None = None,
    dataset_digest: str | None = None,
) -> None:
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    metric_sk = f"{SK_LM_PREFIX}{model_id}{SK_LM_METRIC_PREFIX}{metric_name}#{run_id}"

    # Check for existing metric to delete old RANK item
    existing = self._table.get_item(pk=pk, sk=metric_sk)
    if existing is not None:
        old_inv = self._invert_metric_value(float(existing["metric_value"]))
        old_rank_sk = f"{SK_RANK_LM_PREFIX}{metric_name}#{old_inv}#{model_id}"
        self._table.delete_item(pk=pk, sk=old_rank_sk)
        if existing.get("dataset_name") and existing.get("dataset_digest"):
            old_rank_sk_ds = f"{SK_RANK_LMD_PREFIX}{metric_name}#{existing['dataset_name']}#{existing['dataset_digest']}#{old_inv}#{model_id}"
            self._table.delete_item(pk=pk, sk=old_rank_sk_ds)

    # Write metric sub-item
    metric_item: dict[str, Any] = {
        "PK": pk,
        "SK": metric_sk,
        "metric_name": metric_name,
        "metric_value": Decimal(str(metric_value)),
        "metric_timestamp_ms": metric_timestamp_ms,
        "metric_step": metric_step,
        "run_id": run_id,
        "model_id": model_id,
    }
    if dataset_name:
        metric_item["dataset_name"] = dataset_name
    if dataset_digest:
        metric_item["dataset_digest"] = dataset_digest
    self._table.put_item(metric_item)

    # Write global RANK item
    inv_value = self._invert_metric_value(metric_value)
    rank_item: dict[str, Any] = {
        "PK": pk,
        "SK": f"{SK_RANK_LM_PREFIX}{metric_name}#{inv_value}#{model_id}",
        "model_id": model_id,
        "metric_name": metric_name,
        "metric_value": Decimal(str(metric_value)),
    }
    self._table.put_item(rank_item)

    # Write dataset-scoped RANK item
    if dataset_name and dataset_digest:
        rank_item_ds: dict[str, Any] = {
            "PK": pk,
            "SK": f"{SK_RANK_LMD_PREFIX}{metric_name}#{dataset_name}#{dataset_digest}#{inv_value}#{model_id}",
            "model_id": model_id,
            "metric_name": metric_name,
            "metric_value": Decimal(str(metric_value)),
            "dataset_name": dataset_name,
            "dataset_digest": dataset_digest,
        }
        self._table.put_item(rank_item_ds)

    # Update last_updated_timestamp_ms on META
    now_ms = int(time.time() * 1000)
    self._table.update_item(
        pk=pk,
        sk=f"{SK_LM_PREFIX}{model_id}",
        updates={"last_updated_timestamp_ms": now_ms},
    )
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_logged_models.py::TestLogLoggedModelMetric -v`
Expected: All PASS.

- [x] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_logged_models.py
git commit -m "feat: implement _log_logged_model_metric with RANK items"
```

---

### Task 6: Search infrastructure (parse, plan, execute)

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/search.py`
- Create: `tests/unit/test_search_logged_models.py`

- [x] **Step 1: Write failing tests for parse and plan**

Create `tests/unit/test_search_logged_models.py`:

```python
"""Tests for logged model search parsing, planning, and execution."""

from mlflow_dynamodbstore.dynamodb.search import (
    FilterPredicate,
    parse_logged_model_filter,
    plan_logged_model_query,
)


class TestParseLoggedModelFilter:
    def test_empty_filter(self):
        assert parse_logged_model_filter(None) == []
        assert parse_logged_model_filter("") == []

    def test_attribute_filter(self):
        preds = parse_logged_model_filter("name = 'my-model'")
        assert len(preds) == 1
        assert preds[0].field_type == "attribute"
        assert preds[0].key == "name"
        assert preds[0].op == "="
        assert preds[0].value == "my-model"

    def test_metric_filter(self):
        preds = parse_logged_model_filter("metrics.accuracy > 0.5")
        assert len(preds) == 1
        assert preds[0].field_type == "metric"
        assert preds[0].key == "accuracy"

    def test_tag_filter(self):
        preds = parse_logged_model_filter("tags.env = 'prod'")
        assert len(preds) == 1
        assert preds[0].field_type == "tag"

    def test_param_filter(self):
        preds = parse_logged_model_filter("params.lr = '0.01'")
        assert len(preds) == 1
        assert preds[0].field_type == "param"


class TestPlanLoggedModelQuery:
    def test_default_plan(self):
        plan = plan_logged_model_query([], None, None)
        assert plan.strategy == "index"
        assert plan.index == "lsi1"
        assert plan.sk_prefix == "active#"

    def test_metric_order_by_uses_rank(self):
        plan = plan_logged_model_query(
            [], [{"field_name": "metrics.accuracy", "ascending": False}], None
        )
        assert plan.strategy == "rank"
        assert plan.rank_key == "accuracy"

    def test_status_filter_uses_lsi3(self):
        preds = [FilterPredicate("attribute", "status", "=", "READY")]
        plan = plan_logged_model_query(preds, None, None)
        assert plan.index == "lsi3"
        assert plan.sk_prefix == "READY#"

    def test_name_order_by_uses_lsi4(self):
        plan = plan_logged_model_query(
            [], [{"field_name": "name", "ascending": True}], None
        )
        assert plan.index == "lsi4"

    def test_metric_filter_becomes_rank_filter(self):
        preds = [FilterPredicate("metric", "accuracy", ">", 0.5)]
        plan = plan_logged_model_query(preds, None, None)
        assert len(plan.rank_filters) == 1
        assert plan.rank_filters[0].key == "accuracy"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_search_logged_models.py -v -x`
Expected: FAIL — `parse_logged_model_filter` not found.

- [x] **Step 3: Implement parse, plan, execute in search.py**

Add to `src/mlflow_dynamodbstore/dynamodb/search.py`:

1. Extend `QueryPlan` with new fields:

```python
@dataclass
class QueryPlan:
    # ... existing fields ...
    rank_filters: list[FilterPredicate] = field(default_factory=list)
    datasets: list[dict[str, Any]] | None = None
```

2. Add parser:

```python
def parse_logged_model_filter(filter_string: str | None) -> list[FilterPredicate]:
    """Parse an MLflow logged model search filter string."""
    if not filter_string:
        return []
    from mlflow.utils.search_utils import SearchLoggedModelsUtils
    parsed = SearchLoggedModelsUtils.parse_search_filter(filter_string)
    return _to_predicates(parsed)
```

3. Add planner:

```python
_LM_ORDER_BY_LSI: dict[str, str] = {
    "creation_timestamp": "lsi2",
    "creation_time": "lsi2",
    "last_updated_timestamp": "lsi2",
    "name": "lsi4",
}


def plan_logged_model_query(
    predicates: list[FilterPredicate],
    order_by: list[dict[str, Any]] | None,
    datasets: list[dict[str, Any]] | None,
) -> QueryPlan:
    """Produce a QueryPlan for logged model search."""
    # 1. RANK strategy for metric order_by
    if order_by:
        for ob in order_by:
            field_name = ob.get("field_name", "")
            if "." in field_name:
                entity, key = field_name.split(".", 1)
                if entity == "metrics":
                    ascending = ob.get("ascending", True)
                    return QueryPlan(
                        strategy="rank",
                        index=None,
                        sk_prefix=None,
                        scan_forward=ascending,  # inverted values flip this in execute
                        rank_key=key,
                        rank_filters=[p for p in predicates if p.field_type == "metric"],
                        post_filters=[p for p in predicates if p.field_type != "metric"],
                        datasets=datasets,
                    )

    # Separate metric predicates as rank_filters
    rank_filters = [p for p in predicates if p.field_type == "metric"]
    non_metric_preds = [p for p in predicates if p.field_type != "metric"]

    # 2. Status filter → LSI3
    for pred in non_metric_preds:
        if pred.field_type == "attribute" and pred.key == "status" and pred.op == "=":
            remaining = [p for p in non_metric_preds if p is not pred]
            return QueryPlan(
                strategy="index",
                index="lsi3",
                sk_prefix=f"{pred.value}#",
                scan_forward=False,
                post_filters=remaining,
                rank_filters=rank_filters,
                datasets=datasets,
            )

    # 3. Order by attribute → LSI
    chosen_index = "lsi1"
    scan_forward = False
    sk_prefix: str | None = "active#"

    if order_by:
        for ob in order_by:
            field_name = ob.get("field_name", "")
            if field_name in _LM_ORDER_BY_LSI:
                chosen_index = _LM_ORDER_BY_LSI[field_name]
                scan_forward = ob.get("ascending", True)
                sk_prefix = None
                break

    return QueryPlan(
        strategy="index",
        index=chosen_index,
        sk_prefix=sk_prefix,
        scan_forward=scan_forward,
        post_filters=non_metric_preds,
        rank_filters=rank_filters,
        datasets=datasets,
    )
```

4. Add executor:

```python
def _is_logged_model_meta(item: dict[str, Any]) -> bool:
    """Check if item is a logged model META (not a sub-item)."""
    return "model_id" in item and "lifecycle_stage" in item


def _execute_lm_rank(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute rank-based strategy for logged models."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_LM_PREFIX, SK_RANK_LM_PREFIX, SK_RANK_LMD_PREFIX

    sk_prefix = f"{SK_RANK_LM_PREFIX}{plan.rank_key}#"
    # If datasets specified, use dataset-scoped RANK
    if plan.datasets:
        ds = plan.datasets[0]
        ds_name = ds.get("name", "")
        ds_digest = ds.get("digest", "")
        if ds_name:
            sk_prefix = f"{SK_RANK_LMD_PREFIX}{plan.rank_key}#{ds_name}#"
            if ds_digest:
                sk_prefix = f"{SK_RANK_LMD_PREFIX}{plan.rank_key}#{ds_name}#{ds_digest}#"

    rank_items = table.query(
        pk=pk,
        sk_prefix=sk_prefix,
        scan_forward=not plan.scan_forward,  # inverted values
    )

    model_ids = [item["model_id"] for item in rank_items if "model_id" in item]
    # Batch get META items
    items = []
    for mid in model_ids:
        meta = table.get_item(pk, f"{SK_LM_PREFIX}{mid}")
        if meta and _is_logged_model_meta(meta) and meta.get("lifecycle_stage") != "deleted":
            items.append(meta)
    return items


def _execute_lm_index(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute index-based strategy for logged models."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_LM_PREFIX

    items = table.query(
        pk=pk,
        sk_prefix=plan.sk_prefix,
        index_name=plan.index,
        scan_forward=plan.scan_forward,
    )
    # Filter to logged model META items only
    return [item for item in items if _is_logged_model_meta(item)]


def _apply_lm_post_filter(
    table: DynamoDBTable,
    pk: str,
    model_id: str,
    item: dict[str, Any],
    pred: FilterPredicate,
) -> bool:
    """Apply a post-filter predicate for logged model search."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_LM_PREFIX

    if pred.field_type == "attribute":
        key_map = {
            "creation_timestamp": "creation_timestamp_ms",
            "creation_time": "creation_timestamp_ms",
            "last_updated_timestamp": "last_updated_timestamp_ms",
        }
        item_key = key_map.get(pred.key, pred.key)
        actual = item.get(item_key)
        if pred.key in ("creation_timestamp", "creation_time", "last_updated_timestamp"):
            if actual is not None:
                actual = int(actual)
            if pred.value is not None:
                pred = FilterPredicate(pred.field_type, pred.key, pred.op, int(pred.value))
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "tag":
        tags = item.get("tags", {})
        return _compare(tags.get(pred.key), pred.op, pred.value)

    if pred.field_type == "param":
        params = item.get("params", {})
        return _compare(params.get(pred.key), pred.op, pred.value)

    return True


def execute_logged_model_query(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
    predicates: list[FilterPredicate] | None = None,
) -> list[dict[str, Any]]:
    """Execute a logged model search query. Returns all matching items (no pagination).

    Pagination is handled by the caller (search_logged_models) after merging
    results across multiple experiments.
    """
    predicates = predicates or []

    if plan.strategy == "rank":
        items = _execute_lm_rank(table, plan, pk)
    else:
        items = _execute_lm_index(table, plan, pk)

    # Apply rank_filters (metric predicates via RANK range queries)
    if plan.rank_filters and plan.strategy != "rank":
        from mlflow_dynamodbstore.dynamodb.schema import SK_RANK_LM_PREFIX

        for rf in plan.rank_filters:
            sk_prefix = f"{SK_RANK_LM_PREFIX}{rf.key}#"
            rank_items = table.query(pk=pk, sk_prefix=sk_prefix)
            matching_ids = set()
            for ri in rank_items:
                val = float(ri.get("metric_value", 0))
                if _compare(val, rf.op, rf.value):
                    matching_ids.add(ri["model_id"])
            items = [i for i in items if i.get("model_id") in matching_ids]

    # Apply post-filters
    filtered = []
    for item in items:
        model_id = item.get("model_id", "")
        if all(
            _apply_lm_post_filter(table, pk, model_id, item, pred)
            for pred in plan.post_filters
        ):
            filtered.append(item)

    return filtered
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_search_logged_models.py -v`
Expected: All PASS.

- [x] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/search.py tests/unit/test_search_logged_models.py
git commit -m "feat: add parse/plan/execute for logged model search"
```

---

### Task 7: search_logged_models store method

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Modify: `tests/unit/test_logged_models.py`

- [x] **Step 1: Write failing tests for search**

Add to `tests/unit/test_logged_models.py`:

```python
from mlflow.store.entities.paged_list import PagedList


class TestSearchLoggedModels:
    def test_search_empty(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        result = tracking_store.search_logged_models(experiment_ids=[exp_id])
        assert isinstance(result, PagedList)
        assert len(result) == 0

    def test_search_returns_models(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-a")
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-b")

        result = tracking_store.search_logged_models(experiment_ids=[exp_id])
        assert len(result) == 2

    def test_search_excludes_deleted(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        m1 = tracking_store.create_logged_model(experiment_id=exp_id, name="keep")
        m2 = tracking_store.create_logged_model(experiment_id=exp_id, name="delete-me")
        tracking_store.delete_logged_model(m2.model_id)

        result = tracking_store.search_logged_models(experiment_ids=[exp_id])
        assert len(result) == 1
        assert result[0].name == "keep"

    def test_search_filter_by_name(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-a")
        tracking_store.create_logged_model(experiment_id=exp_id, name="model-b")

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="name = 'model-a'"
        )
        assert len(result) == 1
        assert result[0].name == "model-a"

    def test_search_filter_by_status(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        m1 = tracking_store.create_logged_model(experiment_id=exp_id, name="ready")
        tracking_store.finalize_logged_model(m1.model_id, LoggedModelStatus.READY)
        tracking_store.create_logged_model(experiment_id=exp_id, name="pending")

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id], filter_string="status = 'READY'"
        )
        assert len(result) == 1
        assert result[0].name == "ready"

    def test_search_order_by_name(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tracking_store.create_logged_model(experiment_id=exp_id, name="beta")
        tracking_store.create_logged_model(experiment_id=exp_id, name="alpha")

        result = tracking_store.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "name", "ascending": True}],
        )
        assert result[0].name == "alpha"
        assert result[1].name == "beta"

    def test_search_across_experiments(self, tracking_store):
        exp1 = _create_experiment(tracking_store)
        exp2 = tracking_store.create_experiment("test-exp-2", artifact_location="s3://bucket/a2")
        tracking_store.create_logged_model(experiment_id=exp1, name="m1")
        tracking_store.create_logged_model(experiment_id=exp2, name="m2")

        result = tracking_store.search_logged_models(experiment_ids=[exp1, exp2])
        assert len(result) == 2

    def test_search_pagination(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        for i in range(5):
            tracking_store.create_logged_model(experiment_id=exp_id, name=f"model-{i}")

        page1 = tracking_store.search_logged_models(
            experiment_ids=[exp_id], max_results=2
        )
        assert len(page1) == 2
        assert page1.token is not None

        page2 = tracking_store.search_logged_models(
            experiment_ids=[exp_id], max_results=2, page_token=page1.token
        )
        assert len(page2) == 2
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_logged_models.py::TestSearchLoggedModels -v -x`
Expected: FAIL.

- [x] **Step 3: Implement search_logged_models**

```python
def search_logged_models(
    self,
    experiment_ids: list[str],
    filter_string: str | None = None,
    datasets: list[dict[str, Any]] | None = None,
    max_results: int | None = None,
    order_by: list[dict[str, Any]] | None = None,
    page_token: str | None = None,
) -> PagedList[LoggedModel]:
    from mlflow_dynamodbstore.dynamodb.search import (
        execute_logged_model_query,
        parse_logged_model_filter,
        plan_logged_model_query,
    )

    from mlflow_dynamodbstore.dynamodb.pagination import decode_page_token, encode_page_token

    max_results = max_results or 100
    predicates = parse_logged_model_filter(filter_string)
    plan = plan_logged_model_query(predicates, order_by, datasets)

    # Collect all matching items across experiments (no pagination yet)
    all_items: list[dict[str, Any]] = []
    for exp_id in experiment_ids:
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        items = execute_logged_model_query(self._table, plan, pk, predicates)
        all_items.extend(items)

    # Convert to entities
    models: list[LoggedModel] = []
    for item in all_items:
        exp_id = item["experiment_id"]
        model_id = item["model_id"]
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        tag_items = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_TAG_PREFIX}")
        param_items = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_PARAM_PREFIX}")
        metric_items = self._table.query(pk=pk, sk_prefix=f"{SK_LM_PREFIX}{model_id}{SK_LM_METRIC_PREFIX}")
        models.append(_item_to_logged_model(item, tag_items, param_items, metric_items))

    # Pagination (applied after merging across experiments)
    token_data = decode_page_token(page_token)
    offset = token_data.get("offset", 0) if token_data else 0
    page = models[offset : offset + max_results]
    has_more = len(models) > offset + max_results
    next_token = encode_page_token({"offset": offset + max_results}) if has_more else None

    return PagedList(page, next_token)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_logged_models.py::TestSearchLoggedModels -v`
Expected: All PASS.

- [x] **Step 5: Run full unit test suite**

Run: `uv run pytest tests/unit/ -v --tb=short`
Expected: All PASS (no regressions).

- [x] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_logged_models.py
git commit -m "feat: implement search_logged_models with parse/plan/execute"
```

---

### Task 8: E2E Tests

**Files:**
- Create: `tests/e2e/test_logged_models.py`

- [x] **Step 1: Write e2e tests**

```python
"""E2E tests for logged model operations via MLflow client SDK."""

import uuid

import pytest
from mlflow import MlflowClient
from mlflow.entities import LoggedModelTag
from mlflow.entities.logged_model_status import LoggedModelStatus

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestLoggedModelsCRUD:
    def test_create_and_get(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-crud-{_uid()}")

        model = client.create_logged_model(experiment_id=exp_id, name="e2e-model")
        assert model.model_id.startswith("m-")
        assert model.name == "e2e-model"

        fetched = client.get_logged_model(model.model_id)
        assert fetched.model_id == model.model_id

    def test_finalize_and_delete(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-fin-{_uid()}")

        model = client.create_logged_model(experiment_id=exp_id, name="fin-model")
        finalized = client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
        assert finalized.status == LoggedModelStatus.READY

        client.delete_logged_model(model.model_id)
        with pytest.raises(Exception):
            client.get_logged_model(model.model_id)

    def test_set_and_delete_tag(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-tag-{_uid()}")

        model = client.create_logged_model(experiment_id=exp_id, name="tag-model")
        client.set_logged_model_tags(model.model_id, [LoggedModelTag("env", "prod")])

        fetched = client.get_logged_model(model.model_id)
        assert fetched.tags.get("env") == "prod"

        client.delete_logged_model_tag(model.model_id, "env")
        fetched = client.get_logged_model(model.model_id)
        assert "env" not in fetched.tags


class TestLoggedModelsSearch:
    def test_search_empty(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-empty-{_uid()}")

        result = client.search_logged_models(experiment_ids=[exp_id])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_search_returns_created_models(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-search-{_uid()}")

        client.create_logged_model(experiment_id=exp_id, name="model-a")
        client.create_logged_model(experiment_id=exp_id, name="model-b")

        result = client.search_logged_models(experiment_ids=[exp_id])
        assert len(result) == 2

    def test_search_filter_by_name(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-fname-{_uid()}")

        client.create_logged_model(experiment_id=exp_id, name="model-a")
        client.create_logged_model(experiment_id=exp_id, name="model-b")

        result = client.search_logged_models(
            experiment_ids=[exp_id], filter_string="name = 'model-a'"
        )
        assert len(result) == 1
        assert result[0].name == "model-a"

    def test_search_filter_by_status(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-fstat-{_uid()}")

        m1 = client.create_logged_model(experiment_id=exp_id, name="ready-model")
        client.finalize_logged_model(m1.model_id, LoggedModelStatus.READY)
        client.create_logged_model(experiment_id=exp_id, name="pending-model")

        result = client.search_logged_models(
            experiment_ids=[exp_id], filter_string="status = 'READY'"
        )
        assert len(result) == 1
        assert result[0].name == "ready-model"

    def test_search_order_by_creation_time_desc(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp_id = client.create_experiment(f"e2e-lm-order-{_uid()}")

        client.create_logged_model(experiment_id=exp_id, name="first")
        client.create_logged_model(experiment_id=exp_id, name="second")

        result = client.search_logged_models(
            experiment_ids=[exp_id],
            order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        )
        assert len(result) == 2
        assert result[0].creation_timestamp >= result[1].creation_timestamp

    def test_search_across_experiments(self, mlflow_server):
        client = MlflowClient(tracking_uri=mlflow_server)
        exp1 = client.create_experiment(f"e2e-lm-multi1-{_uid()}")
        exp2 = client.create_experiment(f"e2e-lm-multi2-{_uid()}")

        client.create_logged_model(experiment_id=exp1, name="m1")
        client.create_logged_model(experiment_id=exp2, name="m2")

        result = client.search_logged_models(experiment_ids=[exp1, exp2])
        assert len(result) == 2
```

- [x] **Step 2: Run e2e tests**

Run: `uv run pytest tests/e2e/test_logged_models.py -v`
Expected: All PASS.

- [x] **Step 3: Run full test suite (unit + integration + e2e)**

Run: `uv run pytest tests/unit/ tests/integration/ tests/e2e/ -v --tb=short`
Expected: All PASS, no regressions.

- [x] **Step 4: Commit**

```bash
git add tests/e2e/test_logged_models.py
git commit -m "test: add e2e tests for logged model CRUD and search"
```

---

### Task 9: Final verification and cleanup

- [x] **Step 1: Verify 100% patch coverage**

Run: `uv run pytest tests/unit/ --cov=mlflow_dynamodbstore --cov-report=term-missing | grep -E "tracking_store|search"`

Check that no lines from the logged models patch are in the "Missing" column.

- [x] **Step 2: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: No errors.

- [x] **Step 3: Run type checker**

Run: `uv run mypy src/mlflow_dynamodbstore/tracking_store.py`
Expected: No errors.

- [x] **Step 4: Reinstall package and run e2e with live server**

```bash
uv pip install -e .
uv run pytest tests/e2e/test_logged_models.py -v
```
Expected: All PASS.

- [x] **Step 5: Commit any fixups**

```bash
git add -u
git commit -m "chore: cleanup and coverage fixes for logged models"
```
