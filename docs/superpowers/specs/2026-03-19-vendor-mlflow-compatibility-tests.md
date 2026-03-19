# Vendor MLflow Compatibility Tests

**Date:** 2026-03-19
**Status:** Draft
**Branch:** `feat/vendor-mlflow-tests`

## Problem

The DynamoDB stores (tracking, registry, workspace) implement MLflow's abstract store interfaces, but there is no systematic way to verify they behave identically to MLflow's reference SqlAlchemy implementation. Bugs like `Decimal` vs `float` type mismatches, missing proto fields (`valid` on expectations), and `_DatasetSummary` vs `DatasetSummary` naming were only caught during manual UI testing.

Additionally, the DynamoDB tracking and registry stores hard-code `self._workspace = "default"` and never read from `WorkspaceContext`, meaning workspace isolation is broken when the MLflow server runs with `--enable-workspaces`.

## Goals

1. Catch serialization contract mismatches between DynamoDB and SqlAlchemy stores automatically in CI
2. Reuse MLflow's own store tests against the DynamoDB implementation without depending on out-of-tree files
3. Surface the workspace isolation gap with failing tests that drive the fix

## Non-Goals

- Running MLflow's full tracking store test suite (531 SQL-specific references — too coupled)
- Building a SQLAlchemy dialect for DynamoDB
- Proposing changes upstream to MLflow's test structure

## Approach

Git submodule + import-based test collection. MLflow is vendored as a read-only submodule pinned to a release tag. Thin adapter test files import specific test functions from the vendor and run them against DynamoDB store fixtures.

## Repository Structure

```
zae-mlflow/
├── vendor/
│   └── mlflow/                          # git submodule → mlflow/mlflow @ v3.1.x
│       └── tests/store/...              # MLflow's test files (read-only)
├── tests/
│   └── compatibility/
│       ├── conftest.py                  # DynamoDB + SqlAlchemy store fixtures
│       ├── comparison.py                # Field-by-field comparison engine
│       ├── field_policy.py              # MUST_MATCH / TYPE_MUST_MATCH / IGNORE per entity
│       │
│       │  # Phase 1: Contract fidelity (own code)
│       ├── test_registry_contract.py
│       ├── test_workspace_contract.py
│       ├── test_tracking_contract.py
│       │
│       │  # Phase 2a: CRUD compat (imported from vendor)
│       ├── test_registry_compat.py
│       ├── test_workspace_compat.py
│       │
│       │  # Phase 2b: Workspace isolation compat (imported from vendor)
│       ├── test_registry_workspace_compat.py
│       └── test_tracking_workspace_compat.py
```

## Git Submodule

MLflow is added as a shallow git submodule at `vendor/mlflow`, pinned to a release tag (`>= v3.1`).

```ini
[submodule "vendor/mlflow"]
    path = vendor/mlflow
    url = https://github.com/mlflow/mlflow.git
    shallow = true
```

**Update policy:** Manual. Update the submodule pin in the same PR that bumps the `mlflow` dependency in `pyproject.toml`. They should stay in lockstep.

```bash
cd vendor/mlflow && git fetch --depth 1 origin tag v3.x.y && git checkout v3.x.y
```

**Developer experience:**
- Clone: `git clone --recurse-submodules`
- Run: `uv run pytest -m compatibility`
- CI: add `submodules: true` to GitHub Actions checkout step
- Never modify files under `vendor/mlflow/` — treat as read-only

## Conftest and Fixtures

`tests/compatibility/conftest.py` provides:

### sys.path setup

Adds `vendor/mlflow` to `sys.path` so MLflow test helpers are importable (`from tests.helper_functions import random_str`). MLflow's root `conftest.py` is NOT loaded — our conftest provides all needed fixtures.

### AWS backend selection

Same pattern as the existing e2e conftest:
- Default: moto (`@mock_aws`)
- If AWS credentials present: real DynamoDB
- No code changes needed to switch

### Store fixtures

**For Phase 1 contract tests** — both stores side by side:

```python
@pytest.fixture
def stores(dynamodb_tracking_store, sqlalchemy_tracking_store):
    return Stores(ddb=dynamodb_tracking_store, sql=sqlalchemy_tracking_store)
```

The SqlAlchemy store uses a temporary SQLite database (same as MLflow's own conftest — cheap, no external deps).

**For Phase 2 compat tests** — DynamoDB store only, named `store` to match what MLflow tests expect:

```python
@pytest.fixture
def store(dynamodb_store):
    return dynamodb_store
```

**For Phase 2b workspace tests** — DynamoDB store with workspaces enabled:

```python
@pytest.fixture
def workspace_tracking_store(monkeypatch, dynamodb_tracking_store):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    return dynamodb_tracking_store
```

## Phase 1: Contract Fidelity Tests

### Comparison Framework

**`field_policy.py`** declares per-entity-type field categories:

| Category | Behavior | Example |
|---|---|---|
| `MUST_MATCH` | Values must be equal | `name`, `description`, `tags`, `lifecycle_stage` |
| `TYPE_MUST_MATCH` | Python types must match, values may differ | `creation_timestamp`, `last_updated_timestamp` |
| `IGNORE` | Skipped entirely | `experiment_id` (ULID vs auto-increment), `artifact_location` |

New/unknown fields default to `MUST_MATCH`. This is the safety net — if MLflow adds a field and the DynamoDB store doesn't return it, the test fails.

Policies are defined for each entity type: `EXPERIMENT`, `RUN`, `METRIC`, `PARAM`, `TAG`, `REGISTERED_MODEL`, `MODEL_VERSION`, `WORKSPACE`, `TRACE_INFO`, `ASSESSMENT`, `DATASET_SUMMARY`, etc.

**`comparison.py`** implements the comparison engine:

- Takes two entity objects (or their proto/dict serializations)
- Looks up the field policy for that entity type
- For `MUST_MATCH`: asserts value equality
- For `TYPE_MUST_MATCH`: asserts same Python type
- For `IGNORE`: skips
- Unknown fields: treated as `MUST_MATCH` (fail-safe)
- Returns a structured diff report on failure, not just "not equal"

### Contract Test Pattern

```python
def test_create_and_get_registered_model(stores):
    sql_model = stores.sql.create_registered_model("test-model")
    ddb_model = stores.ddb.create_registered_model("test-model")
    assert_stores_match(sql_model, ddb_model, policy=REGISTERED_MODEL)

    sql_get = stores.sql.get_registered_model("test-model")
    ddb_get = stores.ddb.get_registered_model("test-model")
    assert_stores_match(sql_get, ddb_get, policy=REGISTERED_MODEL)
```

Tests cover: entity CRUD, search operations, batch operations, and the specific operations that have historically produced contract mismatches (traces, assessments, datasets).

## Phase 2a: CRUD Compat Tests

Thin test modules import specific test functions from vendored MLflow test files:

```python
# tests/compatibility/test_registry_compat.py
# ruff: noqa: F401
from vendor.mlflow.tests.store.model_registry.test_sqlalchemy_store import (
    test_create_registered_model,
    test_get_registered_model,
    test_update_registered_model,
    test_rename_registered_model,
    test_delete_registered_model,
    test_search_registered_models,
    test_set_registered_model_tag,
    test_delete_registered_model_tag,
    test_create_model_version,
    # ... curated list, grows over time
)
```

Pytest discovers the imported functions and runs them with our conftest fixtures.

**Excluded tests** (SQL-coupled):
- Tests that access `store.engine`, `store.ManagedSessionMaker`, or `session.query()`
- Tests that assert on SqlAlchemy ORM models (`SqlRegisteredModel`, etc.)
- Tests checking dialect-specific behavior
- Tests using `WorkspaceAwareSqlAlchemyStore` directly (covered by Phase 2b)

Same pattern for `test_workspace_compat.py` importing from `vendor/mlflow/tests/store/workspace/test_sqlalchemy_store.py`.

## Phase 2b: Workspace Isolation Compat Tests

Import MLflow's dedicated workspace isolation tests:

**Source files:**
- `vendor/mlflow/tests/store/model_registry/test_sqlalchemy_workspace_store.py` (~10 isolation tests)
- `vendor/mlflow/tests/store/tracking/test_sqlalchemy_workspace_store.py` (44 isolation tests)

**MLflow's isolation test pattern:**
1. Create resource in `WorkspaceContext("team-a")`
2. Attempt access from `WorkspaceContext("team-b")`
3. Assert `RESOURCE_DOES_NOT_EXIST` error or empty search results

**Expected initial state:** These tests will fail because the DynamoDB stores hard-code `self._workspace = "default"` and ignore `WorkspaceContext`. Mark with `pytest.mark.xfail` until workspace support is implemented, then remove the markers.

**Import files:**
- `test_registry_workspace_compat.py` — imports registry isolation tests
- `test_tracking_workspace_compat.py` — imports tracking isolation tests

## Phase 2c: Future — Tracking CRUD Compat

Deferred. MLflow's tracking store tests (`test_sqlalchemy_store.py`, 510KB) have 531 references to SqlAlchemy internals. Options for the future:
- Heavy curation of import list (labor-intensive)
- Propose upstream test decoupling to MLflow
- Expand Phase 1 contract tests to cover more tracking operations

## CI Integration

- All compatibility tests are auto-marked `compatibility` by the existing `pytest_collection_modifyitems` hook
- Run with: `pytest -m compatibility`
- GitHub Actions checkout step needs `submodules: true`
- Moto by default; real DynamoDB when AWS credentials are present
- Phase 2b workspace tests use `xfail` markers until workspace support is fixed

## Dependency Changes

- `pyproject.toml`: bump `mlflow>=3.0` to `mlflow>=3.1`
- No new Python dependencies (comparison framework is pure stdlib + pytest)
- Submodule adds MLflow repo as a git dependency (shallow, ~single commit)
