# Model Version Soft-Delete with Redaction

## Problem

1. **Version number reuse**: `delete_model_version` hard-deletes the item. `create_model_version` calculates `next_ver = len(version_items) + 1`. After deleting version 3 from [1,2,3], the next version is 3 again — reusing a deleted version number.

2. **Missing redaction**: MLflow's contract requires that deleted model versions have sensitive fields (`source`, `run_id`, `run_link`) redacted to `"REDACTED-*"` values. Our hard-delete destroys the evidence entirely.

3. **Inconsistency**: Every other entity (experiments, runs, logged models) uses soft-delete with TTL. Model versions are the only hard-delete.

## Design

Follow the existing soft-delete pattern used by experiments/runs/logged models.

### `delete_model_version` — soft-delete with redaction

Instead of `delete_item`, update the version item:

```python
updates = {
    "current_stage": STAGE_DELETED_INTERNAL,  # "Deleted_Internal"
    "source": "REDACTED-SOURCE-PATH",
    "run_id": "REDACTED-RUN-ID",
    "run_link": "REDACTED-RUN-LINK",
    "description": "",
    "status_message": "",
    "last_updated_timestamp": now_ms,
    LSI2_SK: now_ms,
    LSI3_SK: f"{STAGE_DELETED_INTERNAL}#{padded}",
}
```

Remove sparse LSI keys that would make the deleted version appear in filtered queries:
```python
removes = [LSI4_SK, LSI5_SK, GSI1_PK, GSI1_SK]
```

Set TTL if configured:
```python
ttl_seconds = self._config.get_soft_deleted_ttl_seconds()
if ttl_seconds is not None:
    updates["ttl"] = int(time.time()) + ttl_seconds
```

Still hard-delete tags and aliases (same as current).

### `get_model_version` — filter deleted versions

After fetching the item, check for deleted stage:
```python
if item.get("current_stage") == STAGE_DELETED_INTERNAL:
    raise MlflowException(
        f"Model Version (name={name}, version={version}) not found",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
```

### `_get_sql_model_version_including_deleted` — internal retrieval

Same as `get_model_version` but without the deleted check. Used by the compatibility test.

### `create_model_version` — fix version counting

Replace `len(version_items) + 1` with max-based approach:
```python
version_numbers = [int(it["version"]) for it in version_items]
next_ver = max(version_numbers) + 1 if version_numbers else 1
```

This correctly handles gaps from soft-deleted versions.

### Search/list filtering

All query paths that enumerate versions must skip `STAGE_DELETED_INTERNAL` items:
- `_get_versions_for_model` — already filters via `filter_fn`; add deleted check
- `_list_all_versions` — add deleted check in loop
- `get_latest_versions` — add deleted check in loop
- `_search_versions_by_run_id` — deleted versions have `GSI1_PK`/`GSI1_SK` removed, so they won't appear in GSI1 queries (no change needed)
- `transition_model_version_stage` — `get_model_version` already rejects deleted (no change needed)

Best approach: use DynamoDB `FilterExpression` to exclude `current_stage = "Deleted_Internal"` at the query level, avoiding Python post-filtering.

### Impact on `archive_existing_versions`

In `transition_model_version_stage`, when archiving existing versions, the loop iterates all versions. Deleted versions should be skipped. Add: `if vi.get("current_stage") == STAGE_DELETED_INTERNAL: continue`.

## Changes

### `src/mlflow_dynamodbstore/registry_store.py`

1. Import `STAGE_DELETED_INTERNAL` from `mlflow.entities.model_registry.model_version_stages`
2. Rewrite `delete_model_version` to soft-delete with redaction + TTL
3. Add deleted check to `get_model_version`
4. Add `_get_sql_model_version_including_deleted` method
5. Fix `create_model_version` version counting (len → max)
6. Add `STAGE_DELETED_INTERNAL` filter to `_get_versions_for_model`, `_list_all_versions`, `get_latest_versions`
7. Skip deleted versions in `transition_model_version_stage` archive loop

### `tests/compatibility/test_registry_compat.py`

8. Remove Cat 6 xfail for `test_delete_model_version_redaction`

## Verification

```bash
uv run pytest tests/compatibility/test_registry_compat.py::test_delete_model_version_redaction -x -v --runxfail
uv run pytest tests/compatibility/test_registry_compat.py -v
uv run pytest tests/unit/ -x -q
```
