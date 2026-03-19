# Model Version Soft-Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace model version hard-delete with soft-delete + redaction, fixing version number reuse and matching MLflow's contract.

**Architecture:** Follow existing soft-delete pattern (experiments/runs/logged models). Use `STAGE_DELETED_INTERNAL` as the deleted marker, redact sensitive fields, set optional TTL, filter deleted versions from all query paths.

**Tech Stack:** DynamoDB, boto3 FilterExpression, MLflow model_version_stages

**Spec:** `docs/superpowers/specs/2026-03-19-model-version-soft-delete.md`

---

## File Structure

- **Modify:** `src/mlflow_dynamodbstore/registry_store.py` — soft-delete, redaction, version counting, filtering
- **Modify:** `tests/compatibility/test_registry_compat.py` — remove Cat 6 xfail

---

### Task 1: Soft-delete `delete_model_version` with redaction

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py:1109-1128`

- [ ] **Step 1: Import `STAGE_DELETED_INTERNAL`**

Add near the top of the file with other MLflow imports:
```python
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
```

- [ ] **Step 2: Rewrite `delete_model_version`**

Replace lines 1109-1128:
```python
def delete_model_version(self, name: str, version: str) -> None:
    """Soft-delete a model version: redact sensitive fields, mark as deleted."""
    # Verify version exists (raises on deleted/missing versions)
    self.get_model_version(name, version)
    model_ulid = self._resolve_model_ulid(name)
    padded = _pad_version(version)
    pk = f"{PK_MODEL_PREFIX}{model_ulid}"
    now_ms = get_current_time_millis()

    # Soft-delete: redact sensitive fields, set deleted stage
    updates: dict[str, Any] = {
        "current_stage": STAGE_DELETED_INTERNAL,
        "source": "REDACTED-SOURCE-PATH",
        "run_id": "REDACTED-RUN-ID",
        "run_link": "REDACTED-RUN-LINK",
        "description": "",
        "status_message": "",
        "last_updated_timestamp": now_ms,
        LSI2_SK: now_ms,
        LSI3_SK: f"{STAGE_DELETED_INTERNAL}#{padded}",
    }

    # Optional TTL for eventual hard-delete
    ttl_seconds = self._config.get_soft_deleted_ttl_seconds()
    if ttl_seconds is not None:
        import time
        updates["ttl"] = int(time.time()) + ttl_seconds

    # Remove sparse index keys so deleted version won't appear in filtered queries
    removes = [LSI4_SK, LSI5_SK, GSI1_PK, GSI1_SK]

    self._table.update_item(
        pk=pk,
        sk=f"{SK_VERSION_PREFIX}{padded}",
        updates=updates,
        removes=removes,
    )

    # Hard-delete tags (no value in keeping redacted version's tags)
    tag_prefix = f"{SK_VERSION_PREFIX}{padded}{SK_VERSION_TAG_SUFFIX}"
    tag_items = self._table.query(pk=pk, sk_prefix=tag_prefix)
    for tag_item in tag_items:
        self._table.delete_item(pk=pk, sk=tag_item["SK"])

    # Delete aliases pointing to this version
    for alias_name in self._aliases_for_model_version(model_ulid, int(version)):
        self._table.delete_item(pk=pk, sk=f"{SK_MODEL_ALIAS_PREFIX}{alias_name}")
```

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: soft-delete model versions with redaction and TTL"
```

---

### Task 2: Add deleted check to `get_model_version` and internal retrieval method

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py:1055-1080`

- [ ] **Step 1: Add deleted check to `get_model_version`**

After line 1076 (the `if item is None` check), add:
```python
if item.get("current_stage") == STAGE_DELETED_INTERNAL:
    raise MlflowException(
        f"Model Version (name={name}, version={version}) not found",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
```

- [ ] **Step 2: Add `_get_sql_model_version_including_deleted` method**

After `get_model_version`, add:
```python
def _get_sql_model_version_including_deleted(
    self, name: str, version: str
) -> ModelVersion:
    """Retrieve a model version even if soft-deleted (for testing/audit)."""
    model_ulid = self._resolve_model_ulid(name)
    padded = _pad_version(version)
    item = self._table.get_item(
        pk=f"{PK_MODEL_PREFIX}{model_ulid}",
        sk=f"{SK_VERSION_PREFIX}{padded}",
    )
    if item is None:
        raise MlflowException(
            f"Model Version (name={name}, version={version}) not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    tags = self._get_version_tags(model_ulid, padded)
    aliases = self._aliases_for_model_version(model_ulid, int(version))
    return _item_to_model_version(item, tags, aliases)
```

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: filter deleted model versions from get_model_version, add internal retrieval"
```

---

### Task 3: Fix version number counting in `create_model_version`

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py:997-1001`

- [ ] **Step 1: Replace len-based counting with max-based**

Replace line 1001:
```python
# Before:
next_ver = len(version_items) + 1

# After:
if version_items:
    next_ver = max(int(it["version"]) for it in version_items) + 1
else:
    next_ver = 1
```

- [ ] **Step 2: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "fix: use max version number instead of count to prevent version reuse"
```

---

### Task 4: Filter deleted versions from search/list queries

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add deleted filter to `_get_versions_for_model` (line ~1410)**

After the tag/non-version skip check, add:
```python
if item.get("current_stage") == STAGE_DELETED_INTERNAL:
    continue
```

- [ ] **Step 2: Add deleted filter to `_list_all_versions` (line ~1488)**

After the tag skip check, add:
```python
if vi.get("current_stage") == STAGE_DELETED_INTERNAL:
    continue
```

- [ ] **Step 3: Add deleted filter to `get_latest_versions` no-stages path (line ~1504)**

After filtering out tag items, add:
```python
ver_items = [vi for vi in ver_items if vi.get("current_stage") != STAGE_DELETED_INTERNAL]
```

- [ ] **Step 4: Skip deleted versions in `transition_model_version_stage` archive loop (line ~1623)**

In the loop, after `if vi_padded == padded: continue`, add:
```python
if vi.get("current_stage") == STAGE_DELETED_INTERNAL:
    continue
```

Also skip tag items in this loop:
```python
if SK_VERSION_TAG_SUFFIX in vi["SK"]:
    continue
```

- [ ] **Step 5: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: filter deleted model versions from all search/list/transition queries"
```

---

### Task 5: Remove Cat 6 xfail and verify

**Files:**
- Modify: `tests/compatibility/test_registry_compat.py`

- [ ] **Step 1: Remove Cat 6 xfail block**

Remove:
```python
# --- Category 6: missing SqlAlchemy-internal method ---
_xfail_sql_internal = pytest.mark.xfail(
    reason="Test uses _get_sql_model_version_including_deleted (SqlAlchemy-specific)"
)
test_delete_model_version_redaction = _xfail_sql_internal(test_delete_model_version_redaction)
```

- [ ] **Step 2: Run the target test**

Run: `uv run pytest tests/compatibility/test_registry_compat.py::test_delete_model_version_redaction -x -v --runxfail`
Expected: PASS

- [ ] **Step 3: Run full compat suite**

Run: `uv run pytest tests/compatibility/test_registry_compat.py -v`
Expected: 45 passed, 5 xfailed.

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add tests/compatibility/test_registry_compat.py
git commit -m "test: remove Cat 6 xfail — soft-delete with redaction now implemented"
```
