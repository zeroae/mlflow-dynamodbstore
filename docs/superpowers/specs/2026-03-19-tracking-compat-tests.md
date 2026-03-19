# Phase 2c: Tracking CRUD Compat Tests

**Date:** 2026-03-19
**Status:** Draft
**Branch:** `feat/vendor-mlflow-tests`
**Parent spec:** `docs/superpowers/specs/2026-03-19-vendor-mlflow-compatibility-tests.md`

## Problem

Phase 2c was deferred in the parent spec because MLflow's tracking store tests were estimated to have 531 SQL-specific references, making them "too coupled." Analysis of the actual test functions in v3.10.1 shows **247 of 316 tests (78%) are importable** — they only use the abstract store interface and don't reference SqlAlchemy internals.

## Goal

Import the 247 importable tracking store tests from MLflow, run them against the DynamoDB store, and xfail failures so CI stays green. The xfail markers serve as an inventory of DynamoDB store gaps.

## Approach

Same pattern as `test_registry_compat.py` (Phase 2a). One file, curated import list, fixture override for `store`.

## File

**Create:** `tests/compatibility/test_tracking_compat.py`

### Fixture override

The compatibility conftest's `store` fixture defaults to `registry_store`. Tracking tests need `tracking_store`, so the file defines its own override:

```python
@pytest.fixture
def store(tracking_store):
    return tracking_store
```

This takes precedence over the conftest-level `store` fixture for tests in this module.

### Import list

247 test functions imported from `vendor/mlflow/tests/store/tracking/test_sqlalchemy_store.py`.

### Excluded tests (69)

Tests excluded because they reference SqlAlchemy internals or take unsupported fixtures:

| Reason | Count | Examples |
|---|---|---|
| Uses `ManagedSessionMaker` / session internals | 28 | `test_create_experiments`, `test_log_metric`, `test_run_info` |
| Takes `store_and_trace_info` fixture | 12 | `test_create_and_get_assessment`, `test_update_assessment_*` |
| Takes `workspaces_enabled` fixture | 4 | `test_create_experiment_with_tags_works_correctly` |
| Takes `store_with_traces` fixture | 4 | `test_search_traces_order_by`, `test_search_traces_pagination` |
| Takes `db_uri` / `tmp_sqlite_uri` fixture | 5 | `test_get_orderby_clauses`, `test_sqlalchemy_store_*` |
| References `SqlRun`/`SqlMetric`/ORM models | 6 | `test_get_attribute_name`, `test_get_metric_history` |
| References DB-specific types (MSSQL/MYSQL) | 4 | `test_set_zero_value_insertion_*` |
| References `IntegrityError` / dialect | 3 | `test_log_batch_duplicate_metrics_across_key_batches` |
| References schema migration internals | 3 | `test_upgrade_cli_idempotence`, `test_metrics_materialization_*` |

### xfail strategy

After first run:
1. Run all 247 tests, note failures
2. Group failures by reason (e.g., "trace search filter not implemented", "scorer not implemented", "dataset feature gap")
3. Apply `pytest.mark.xfail(reason="...")` per group so CI stays green
4. Remove xfail markers as gaps are fixed

## No other changes needed

- Conftest already has `tracking_store` fixture and `sys.path` setup
- `pytest_collection_modifyitems` already marks `/compatibility/` tests
- CI already runs `pytest -m compatibility`
