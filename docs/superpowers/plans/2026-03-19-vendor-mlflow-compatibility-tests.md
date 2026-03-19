# Vendor MLflow Compatibility Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor MLflow as a git submodule and build a compatibility test suite that catches contract mismatches between DynamoDB and SqlAlchemy store implementations.

**Architecture:** Git submodule at `vendor/mlflow` pinned to a release tag. `tests/compatibility/` contains Phase 1 contract fidelity tests (our own code comparing both stores side-by-side) and Phase 2 compat tests (imported from vendored MLflow). A comparison framework with categorized field policies drives Phase 1.

**Tech Stack:** pytest, moto, mlflow SqlAlchemyStore (SQLite), git submodules

**Spec:** `docs/superpowers/specs/2026-03-19-vendor-mlflow-compatibility-tests.md`

---

## File Structure

**New files:**
- `tests/compatibility/__init__.py` — empty, makes directory a package
- `tests/compatibility/conftest.py` — fixtures: DynamoDB + SqlAlchemy stores, sys.path setup
- `tests/compatibility/comparison.py` — field-by-field comparison engine
- `tests/compatibility/field_policy.py` — MUST_MATCH / TYPE_MUST_MATCH / IGNORE per entity type
- `tests/compatibility/test_comparison.py` — unit tests for the comparison engine itself
- `tests/compatibility/test_registry_contract.py` — Phase 1: registry contract tests
- `tests/compatibility/test_workspace_contract.py` — Phase 1: workspace contract tests
- `tests/compatibility/test_tracking_contract.py` — Phase 1: tracking contract tests
- `tests/compatibility/test_registry_compat.py` — Phase 2a: imported MLflow registry tests
- `tests/compatibility/test_workspace_compat.py` — Phase 2a: imported MLflow workspace tests
- `tests/compatibility/test_registry_workspace_compat.py` — Phase 2b: imported registry isolation tests
- `tests/compatibility/test_tracking_workspace_compat.py` — Phase 2b: imported tracking isolation tests

**Modified files:**
- `pyproject.toml:19` — bump `mlflow>=3.0` to `mlflow>=3.10`
- `.gitmodules` — new file, submodule config
- `.gitignore` — add `vendor/mlflow` build artifacts if needed

---

### Task 1: Add MLflow git submodule

**Files:**
- Create: `.gitmodules`
- Create: `vendor/mlflow/` (submodule)
- Modify: `pyproject.toml:19`

- [ ] **Step 1: Add the submodule pinned to latest v3.10.x tag**

```bash
# Find the latest v3.1 tag
# Note: if v3.10 not yet released, use latest available v3.x tag
git ls-remote --tags https://github.com/mlflow/mlflow.git 'refs/tags/v3.*' | sort -V | tail -1

# Add submodule (use the actual tag found above)
git submodule add --depth 1 https://github.com/mlflow/mlflow.git vendor/mlflow
cd vendor/mlflow
git fetch --depth 1 origin tag v3.10.0
git checkout v3.10.0
cd ../..
```

- [ ] **Step 2: Bump mlflow dependency**

In `pyproject.toml`, change line 20:
```python
# old
"mlflow>=3.0",
# new
"mlflow>=3.10",
```

- [ ] **Step 3: Verify submodule is functional**

```bash
# Verify submodule is at correct tag
cd vendor/mlflow && git describe --tags && cd ../..
# Expected: v3.10.0 (or the tag you picked)

# Verify the test files we need exist
ls vendor/mlflow/tests/store/model_registry/test_sqlalchemy_store.py
ls vendor/mlflow/tests/store/model_registry/test_sqlalchemy_workspace_store.py
ls vendor/mlflow/tests/store/workspace/test_sqlalchemy_store.py
ls vendor/mlflow/tests/store/tracking/test_sqlalchemy_workspace_store.py
ls vendor/mlflow/tests/helper_functions.py
```

- [ ] **Step 4: Commit**

```bash
git add .gitmodules vendor/mlflow pyproject.toml
git commit -m "chore: add mlflow as vendor submodule at v3.10.x

Pin mlflow submodule for compatibility testing. Bump mlflow
dependency to >=3.10 to match."
```

---

### Task 2: Create compatibility conftest with store fixtures

**Files:**
- Create: `tests/compatibility/__init__.py`
- Create: `tests/compatibility/conftest.py`

- [ ] **Step 1: Create empty `__init__.py`**

```python
# tests/compatibility/__init__.py
```

- [ ] **Step 2: Write conftest.py**

```python
# tests/compatibility/conftest.py
"""Fixtures for compatibility tests between DynamoDB and SqlAlchemy stores."""
import sys
from collections import namedtuple
from pathlib import Path

import pytest
from moto import mock_aws

# Add vendor/mlflow to sys.path so MLflow test helpers are importable.
# This enables `from tests.helper_functions import random_str`.
# IMPORTANT: vendor/mlflow/tests/ must NEVER be added to pytest's testpaths.
_VENDOR_MLFLOW = str(Path(__file__).parents[2] / "vendor" / "mlflow")
if _VENDOR_MLFLOW not in sys.path:
    sys.path.insert(0, _VENDOR_MLFLOW)

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES  # noqa: E402
from mlflow.store.model_registry.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyStore as SqlAlchemyRegistryStore,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as SqlAlchemyTrackingStore  # noqa: E402
from mlflow.store.workspace.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyStore as SqlAlchemyWorkspaceStore,
)

from mlflow_dynamodbstore.registry_store import DynamoDBRegistryStore  # noqa: E402
from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore  # noqa: E402
from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore  # noqa: E402

# Named tuples for Phase 1 contract tests
TrackingStores = namedtuple("TrackingStores", ["ddb", "sql"])
RegistryStores = namedtuple("RegistryStores", ["ddb", "sql"])
WorkspaceStores = namedtuple("WorkspaceStores", ["ddb", "sql"])


# --- DynamoDB store fixtures (moto-backed) ---


@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        yield


@pytest.fixture
def tracking_store(mock_dynamodb):
    return DynamoDBTrackingStore(
        store_uri="dynamodb://us-east-1/test-table",
        artifact_uri="/tmp/artifacts",
    )


@pytest.fixture
def registry_store(mock_dynamodb):
    return DynamoDBRegistryStore(
        store_uri="dynamodb://us-east-1/test-table",
    )


@pytest.fixture
def workspace_store(mock_dynamodb):
    return DynamoDBWorkspaceStore(
        store_uri="dynamodb://us-east-1/test-table",
    )


# --- SqlAlchemy store fixtures (SQLite-backed, for Phase 1 comparison) ---


@pytest.fixture
def sql_tracking_store(tmp_path):
    artifact_uri = str(tmp_path / "artifacts")
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return SqlAlchemyTrackingStore(db_uri, artifact_uri)


@pytest.fixture
def sql_registry_store(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return SqlAlchemyRegistryStore(db_uri)


@pytest.fixture
def sql_workspace_store(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return SqlAlchemyWorkspaceStore(db_uri)


# --- Phase 1: Side-by-side store pairs ---


@pytest.fixture
def tracking_stores(tracking_store, sql_tracking_store):
    return TrackingStores(ddb=tracking_store, sql=sql_tracking_store)


@pytest.fixture
def registry_stores(registry_store, sql_registry_store):
    return RegistryStores(ddb=registry_store, sql=sql_registry_store)


@pytest.fixture
def workspace_stores(workspace_store, sql_workspace_store):
    return WorkspaceStores(ddb=workspace_store, sql=sql_workspace_store)


# --- Phase 2a: Alias fixtures for MLflow test function compatibility ---
# MLflow tests expect a fixture named `store`.
# Each compat test file should override this via its own conftest or
# by importing from here. The `store` fixture below defaults to registry
# but Phase 2a test files may need per-file conftest.py overrides.


@pytest.fixture
def store(registry_store):
    """Default store alias for Phase 2a MLflow test imports."""
    return registry_store


# --- Phase 2b: Workspace-enabled fixtures ---
# These names are LOAD-BEARING — they must exactly match the fixture names
# in MLflow's workspace test files (test_sqlalchemy_workspace_store.py).


@pytest.fixture
def workspace_tracking_store(monkeypatch, tracking_store):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    return tracking_store


@pytest.fixture
def workspace_registry_store(monkeypatch, registry_store):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    return registry_store
```

- [ ] **Step 3: Verify conftest loads and fixtures are discoverable**

```bash
uv run pytest tests/compatibility/ --collect-only 2>&1 | head -5
```

Expected: `no tests ran` (no test files yet), but no import errors.

- [ ] **Step 4: Commit**

```bash
git add tests/compatibility/__init__.py tests/compatibility/conftest.py
git commit -m "feat(tests): add compatibility test conftest with store fixtures

Provides DynamoDB (moto-backed) and SqlAlchemy (SQLite-backed) store
fixtures side-by-side for contract fidelity testing. Also sets up
sys.path for vendored MLflow test imports."
```

---

### Task 3: Build comparison framework — field_policy.py

**Files:**
- Create: `tests/compatibility/field_policy.py`

- [ ] **Step 1: Write field_policy.py**

```python
# tests/compatibility/field_policy.py
"""Field-level comparison policies for store entity types.

Each entity type maps field names to a comparison category:
- MUST_MATCH: values must be exactly equal
- TYPE_MUST_MATCH: Python types must match, values may differ
- IGNORE: skip comparison entirely

Unknown fields default to MUST_MATCH (fail-safe).
"""
from enum import Enum


class FieldPolicy(Enum):
    MUST_MATCH = "must_match"
    TYPE_MUST_MATCH = "type_must_match"
    IGNORE = "ignore"


MUST_MATCH = FieldPolicy.MUST_MATCH
TYPE_MUST_MATCH = FieldPolicy.TYPE_MUST_MATCH
IGNORE = FieldPolicy.IGNORE

# Default policy for unknown fields — fail-safe catches new fields
DEFAULT_POLICY = MUST_MATCH


EXPERIMENT = {
    "experiment_id": IGNORE,  # ULID vs auto-increment integer
    "name": MUST_MATCH,
    "artifact_location": IGNORE,  # backend-specific paths
    "lifecycle_stage": MUST_MATCH,
    "tags": MUST_MATCH,
    "creation_time": TYPE_MUST_MATCH,
    "last_update_time": TYPE_MUST_MATCH,
}

RUN_INFO = {
    "run_id": IGNORE,  # ULID vs UUID
    "run_uuid": IGNORE,
    "experiment_id": IGNORE,
    "user_id": MUST_MATCH,
    "status": MUST_MATCH,
    "start_time": TYPE_MUST_MATCH,
    "end_time": TYPE_MUST_MATCH,
    "artifact_uri": IGNORE,  # backend-specific
    "lifecycle_stage": MUST_MATCH,
    "run_name": MUST_MATCH,
}

RUN_DATA = {
    "metrics": MUST_MATCH,
    "params": MUST_MATCH,
    "tags": MUST_MATCH,
}

METRIC = {
    "key": MUST_MATCH,
    "value": MUST_MATCH,
    "timestamp": TYPE_MUST_MATCH,
    "step": MUST_MATCH,
}

PARAM = {
    "key": MUST_MATCH,
    "value": MUST_MATCH,
}

TAG = {
    "key": MUST_MATCH,
    "value": MUST_MATCH,
}

REGISTERED_MODEL = {
    "name": MUST_MATCH,
    "creation_timestamp": TYPE_MUST_MATCH,
    "last_updated_timestamp": TYPE_MUST_MATCH,
    "description": MUST_MATCH,
    "latest_versions": MUST_MATCH,
    "tags": MUST_MATCH,
    "aliases": MUST_MATCH,
}

MODEL_VERSION = {
    "name": MUST_MATCH,
    "version": MUST_MATCH,
    "creation_timestamp": TYPE_MUST_MATCH,
    "last_updated_timestamp": TYPE_MUST_MATCH,
    "description": MUST_MATCH,
    "user_id": MUST_MATCH,
    "current_stage": MUST_MATCH,
    "source": MUST_MATCH,
    "run_id": IGNORE,  # different ID schemes
    "status": MUST_MATCH,
    "status_message": MUST_MATCH,
    "tags": MUST_MATCH,
    "run_link": MUST_MATCH,
    "aliases": MUST_MATCH,
}

WORKSPACE = {
    "name": MUST_MATCH,
    "description": MUST_MATCH,
    "creation_time": TYPE_MUST_MATCH,
    "last_update_time": TYPE_MUST_MATCH,
}

TRACE_INFO = {
    "request_id": IGNORE,  # different ID schemes
    "experiment_id": IGNORE,
    "timestamp_ms": TYPE_MUST_MATCH,
    "execution_time_ms": TYPE_MUST_MATCH,
    "status": MUST_MATCH,
    "request_metadata": MUST_MATCH,
    "tags": MUST_MATCH,
}
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
uv run python -c "from tests.compatibility.field_policy import REGISTERED_MODEL, FieldPolicy; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/field_policy.py
git commit -m "feat(tests): add field comparison policies for compatibility tests

Defines MUST_MATCH, TYPE_MUST_MATCH, IGNORE categories per entity
type. Unknown fields default to MUST_MATCH as a safety net."
```

---

### Task 4: Build comparison framework — comparison.py

**Files:**
- Create: `tests/compatibility/comparison.py`

- [ ] **Step 1: Write failing test for the comparison engine**

Create a temporary test inline in comparison.py or test it via test_registry_contract.py. For now, write a standalone unit test:

```python
# tests/compatibility/test_comparison.py
"""Tests for the comparison engine itself."""
from tests.compatibility.comparison import assert_entities_match, ComparisonError
from tests.compatibility.field_policy import MUST_MATCH, TYPE_MUST_MATCH, IGNORE


class FakeEntity:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_must_match_equal_passes():
    policy = {"name": MUST_MATCH}
    a = FakeEntity(name="foo")
    b = FakeEntity(name="foo")
    assert_entities_match(a, b, policy)  # should not raise


def test_must_match_unequal_fails():
    policy = {"name": MUST_MATCH}
    a = FakeEntity(name="foo")
    b = FakeEntity(name="bar")
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "name" in str(e)


def test_type_must_match_same_type_passes():
    policy = {"ts": TYPE_MUST_MATCH}
    a = FakeEntity(ts=100)
    b = FakeEntity(ts=200)
    assert_entities_match(a, b, policy)  # values differ, types match


def test_type_must_match_different_type_fails():
    policy = {"ts": TYPE_MUST_MATCH}
    a = FakeEntity(ts=100)
    b = FakeEntity(ts=100.0)
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "ts" in str(e)
        assert "int" in str(e)
        assert "float" in str(e)


def test_ignore_skips_field():
    policy = {"id": IGNORE, "name": MUST_MATCH}
    a = FakeEntity(id="abc", name="same")
    b = FakeEntity(id="xyz", name="same")
    assert_entities_match(a, b, policy)


def test_unknown_field_defaults_to_must_match():
    policy = {"name": MUST_MATCH}  # "new_field" not in policy
    a = FakeEntity(name="same", new_field="a")
    b = FakeEntity(name="same", new_field="b")
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "new_field" in str(e)


def test_missing_field_on_one_side_fails():
    policy = {"name": MUST_MATCH, "desc": MUST_MATCH}
    a = FakeEntity(name="same", desc="hello")
    b = FakeEntity(name="same")  # missing desc
    try:
        assert_entities_match(a, b, policy)
        assert False, "Should have raised"
    except ComparisonError as e:
        assert "desc" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/compatibility/test_comparison.py -v
```

Expected: FAIL — `comparison` module doesn't exist yet.

- [ ] **Step 3: Write comparison.py**

```python
# tests/compatibility/comparison.py
"""Field-by-field comparison engine for store entity contract testing."""
from tests.compatibility.field_policy import DEFAULT_POLICY, FieldPolicy


class ComparisonError(AssertionError):
    """Raised when entity comparison finds mismatches."""

    def __init__(self, diffs: list[str]):
        self.diffs = diffs
        super().__init__("\n".join(diffs))


def _get_fields(obj) -> set[str]:
    """Get comparable fields from an entity object."""
    if hasattr(obj, "__dict__"):
        return {k for k in obj.__dict__ if not k.startswith("_")}
    return set()


def assert_entities_match(
    entity_a,
    entity_b,
    policy: dict[str, FieldPolicy],
    label_a: str = "sql",
    label_b: str = "ddb",
) -> None:
    """Compare two entity objects field-by-field using the given policy.

    Args:
        entity_a: First entity (typically SqlAlchemy store result).
        entity_b: Second entity (typically DynamoDB store result).
        policy: Dict mapping field names to FieldPolicy.
        label_a: Label for entity_a in error messages.
        label_b: Label for entity_b in error messages.

    Raises:
        ComparisonError: If any field mismatches are found.
    """
    fields_a = _get_fields(entity_a)
    fields_b = _get_fields(entity_b)
    all_fields = fields_a | fields_b

    diffs: list[str] = []

    for field in sorted(all_fields):
        field_policy = policy.get(field, DEFAULT_POLICY)

        if field_policy == FieldPolicy.IGNORE:
            continue

        has_a = hasattr(entity_a, field)
        has_b = hasattr(entity_b, field)

        if has_a and not has_b:
            diffs.append(f"  {field}: present in {label_a} but missing in {label_b}")
            continue
        if has_b and not has_a:
            diffs.append(f"  {field}: present in {label_b} but missing in {label_a}")
            continue

        val_a = getattr(entity_a, field)
        val_b = getattr(entity_b, field)

        if field_policy == FieldPolicy.MUST_MATCH:
            if val_a != val_b:
                diffs.append(
                    f"  {field}: {label_a}={val_a!r} != {label_b}={val_b!r}"
                )
        elif field_policy == FieldPolicy.TYPE_MUST_MATCH:
            if type(val_a) is not type(val_b):
                diffs.append(
                    f"  {field}: type mismatch — {label_a}={type(val_a).__name__}, "
                    f"{label_b}={type(val_b).__name__}"
                )

    if diffs:
        raise ComparisonError(diffs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/compatibility/test_comparison.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/compatibility/comparison.py tests/compatibility/test_comparison.py
git commit -m "feat(tests): add field-by-field comparison engine

Compares store entities using categorized field policies. Unknown
fields default to MUST_MATCH for fail-safe regression detection."
```

---

### Task 5: Phase 1 — Registry contract tests

**Files:**
- Create: `tests/compatibility/test_registry_contract.py`

- [ ] **Step 1: Write registry contract tests**

```python
# tests/compatibility/test_registry_contract.py
"""Phase 1: Contract fidelity tests for DynamoDB vs SqlAlchemy registry stores."""
from tests.compatibility.comparison import assert_entities_match
from tests.compatibility.field_policy import REGISTERED_MODEL, MODEL_VERSION


def test_create_and_get_registered_model(registry_stores):
    """Model returned by create/get must match between backends."""
    sql_model = registry_stores.sql.create_registered_model("test-model")
    ddb_model = registry_stores.ddb.create_registered_model("test-model")
    assert_entities_match(sql_model, ddb_model, REGISTERED_MODEL)

    sql_get = registry_stores.sql.get_registered_model("test-model")
    ddb_get = registry_stores.ddb.get_registered_model("test-model")
    assert_entities_match(sql_get, ddb_get, REGISTERED_MODEL)


def test_update_registered_model(registry_stores):
    """Updated model fields must match."""
    registry_stores.sql.create_registered_model("upd-model")
    registry_stores.ddb.create_registered_model("upd-model")

    sql_upd = registry_stores.sql.update_registered_model("upd-model", "new desc")
    ddb_upd = registry_stores.ddb.update_registered_model("upd-model", "new desc")
    assert_entities_match(sql_upd, ddb_upd, REGISTERED_MODEL)


def test_delete_registered_model(registry_stores):
    """Deleting a model should not raise and model should be gone."""
    registry_stores.sql.create_registered_model("del-model")
    registry_stores.ddb.create_registered_model("del-model")

    registry_stores.sql.delete_registered_model("del-model")
    registry_stores.ddb.delete_registered_model("del-model")

    # Both should raise on get
    import pytest
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException):
        registry_stores.sql.get_registered_model("del-model")
    with pytest.raises(MlflowException):
        registry_stores.ddb.get_registered_model("del-model")


def test_search_registered_models(registry_stores):
    """Search results must return same models (ignoring order)."""
    for name in ["search-a", "search-b", "search-c"]:
        registry_stores.sql.create_registered_model(name)
        registry_stores.ddb.create_registered_model(name)

    sql_results = registry_stores.sql.search_registered_models()
    ddb_results = registry_stores.ddb.search_registered_models()

    assert len(sql_results) == len(ddb_results)
    # Compare sorted by name
    sql_sorted = sorted(sql_results, key=lambda m: m.name)
    ddb_sorted = sorted(ddb_results, key=lambda m: m.name)
    for sql_m, ddb_m in zip(sql_sorted, ddb_sorted):
        assert_entities_match(sql_m, ddb_m, REGISTERED_MODEL)


def test_create_and_get_model_version(registry_stores):
    """Model version fields must match between backends."""
    registry_stores.sql.create_registered_model("mv-model")
    registry_stores.ddb.create_registered_model("mv-model")

    sql_mv = registry_stores.sql.create_model_version("mv-model", "s3://source", "run123")
    ddb_mv = registry_stores.ddb.create_model_version("mv-model", "s3://source", "run123")
    assert_entities_match(sql_mv, ddb_mv, MODEL_VERSION)


def test_set_and_get_registered_model_tag(registry_stores):
    """Tags must round-trip identically."""
    registry_stores.sql.create_registered_model("tag-model")
    registry_stores.ddb.create_registered_model("tag-model")

    from mlflow.entities.model_registry import RegisteredModelTag

    tag = RegisteredModelTag("env", "prod")
    registry_stores.sql.set_registered_model_tag("tag-model", tag)
    registry_stores.ddb.set_registered_model_tag("tag-model", tag)

    sql_model = registry_stores.sql.get_registered_model("tag-model")
    ddb_model = registry_stores.ddb.get_registered_model("tag-model")
    assert_entities_match(sql_model, ddb_model, REGISTERED_MODEL)


def test_set_registered_model_alias(registry_stores):
    """Alias operations must match."""
    registry_stores.sql.create_registered_model("alias-model")
    registry_stores.ddb.create_registered_model("alias-model")

    registry_stores.sql.create_model_version("alias-model", "s3://src", "run1")
    registry_stores.ddb.create_model_version("alias-model", "s3://src", "run1")

    registry_stores.sql.set_registered_model_alias("alias-model", "champion", "1")
    registry_stores.ddb.set_registered_model_alias("alias-model", "champion", "1")

    sql_mv = registry_stores.sql.get_model_version_by_alias("alias-model", "champion")
    ddb_mv = registry_stores.ddb.get_model_version_by_alias("alias-model", "champion")
    assert_entities_match(sql_mv, ddb_mv, MODEL_VERSION)
```

- [ ] **Step 2: Run to see which pass and which fail**

```bash
uv run pytest tests/compatibility/test_registry_contract.py -v
```

Expected: some tests may fail due to contract mismatches — that's the point. Note which fields differ.

- [ ] **Step 3: Adjust field policies if needed**

If tests reveal legitimate differences (e.g., fields the DynamoDB store doesn't populate yet), update `field_policy.py` to either IGNORE them (known acceptable difference) or file a bug.

- [ ] **Step 4: Commit**

```bash
git add tests/compatibility/test_registry_contract.py
git commit -m "feat(tests): add Phase 1 registry contract fidelity tests

Compare DynamoDB and SqlAlchemy registry store responses field-by-field
to catch serialization contract mismatches."
```

---

### Task 6: Phase 1 — Workspace contract tests

**Files:**
- Create: `tests/compatibility/test_workspace_contract.py`

- [ ] **Step 1: Write workspace contract tests**

```python
# tests/compatibility/test_workspace_contract.py
"""Phase 1: Contract fidelity tests for DynamoDB vs SqlAlchemy workspace stores."""
from tests.compatibility.comparison import assert_entities_match
from tests.compatibility.field_policy import WORKSPACE


def test_list_workspaces(workspace_stores):
    """Default workspace should be present and match."""
    sql_workspaces = workspace_stores.sql.list_workspaces()
    ddb_workspaces = workspace_stores.ddb.list_workspaces()

    assert len(sql_workspaces) == len(ddb_workspaces)
    for sql_w, ddb_w in zip(
        sorted(sql_workspaces, key=lambda w: w.name),
        sorted(ddb_workspaces, key=lambda w: w.name),
    ):
        assert_entities_match(sql_w, ddb_w, WORKSPACE)


def test_create_and_get_workspace(workspace_stores):
    """Created workspace fields must match."""
    sql_ws = workspace_stores.sql.create_workspace("test-ws", description="desc")
    ddb_ws = workspace_stores.ddb.create_workspace("test-ws", description="desc")
    assert_entities_match(sql_ws, ddb_ws, WORKSPACE)

    sql_get = workspace_stores.sql.get_workspace("test-ws")
    ddb_get = workspace_stores.ddb.get_workspace("test-ws")
    assert_entities_match(sql_get, ddb_get, WORKSPACE)


def test_delete_workspace(workspace_stores):
    """Deleting a workspace should behave the same."""
    workspace_stores.sql.create_workspace("del-ws")
    workspace_stores.ddb.create_workspace("del-ws")

    workspace_stores.sql.delete_workspace("del-ws")
    workspace_stores.ddb.delete_workspace("del-ws")

    import pytest
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException):
        workspace_stores.sql.get_workspace("del-ws")
    with pytest.raises(MlflowException):
        workspace_stores.ddb.get_workspace("del-ws")
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/compatibility/test_workspace_contract.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/test_workspace_contract.py
git commit -m "feat(tests): add Phase 1 workspace contract fidelity tests"
```

---

### Task 7: Phase 1 — Tracking contract tests

**Files:**
- Create: `tests/compatibility/test_tracking_contract.py`

- [ ] **Step 1: Write tracking contract tests**

Focus on the operations that have historically produced mismatches.

```python
# tests/compatibility/test_tracking_contract.py
"""Phase 1: Contract fidelity tests for DynamoDB vs SqlAlchemy tracking stores.

Focused on operations that have historically produced contract mismatches:
Decimal vs float, missing proto fields, entity type naming differences.
"""
from tests.compatibility.comparison import assert_entities_match
from tests.compatibility.field_policy import EXPERIMENT, RUN_INFO, METRIC, PARAM, TAG


def test_create_and_get_experiment(tracking_stores):
    """Experiment fields must match between backends."""
    sql_id = tracking_stores.sql.create_experiment("test-exp")
    ddb_id = tracking_stores.ddb.create_experiment("test-exp")

    sql_exp = tracking_stores.sql.get_experiment(sql_id)
    ddb_exp = tracking_stores.ddb.get_experiment(ddb_id)
    assert_entities_match(sql_exp, ddb_exp, EXPERIMENT)


def test_search_experiments(tracking_stores):
    """Search should return matching experiments."""
    tracking_stores.sql.create_experiment("search-a")
    tracking_stores.sql.create_experiment("search-b")
    tracking_stores.ddb.create_experiment("search-a")
    tracking_stores.ddb.create_experiment("search-b")

    from mlflow.entities import ViewType

    sql_results = tracking_stores.sql.search_experiments(view_type=ViewType.ALL)
    ddb_results = tracking_stores.ddb.search_experiments(view_type=ViewType.ALL)

    # Filter to just our experiments (exclude Default)
    sql_names = sorted([e.name for e in sql_results if e.name.startswith("search-")])
    ddb_names = sorted([e.name for e in ddb_results if e.name.startswith("search-")])
    assert sql_names == ddb_names


def test_create_run_and_log_metrics(tracking_stores):
    """Run info and logged metrics must match."""
    from mlflow.entities import Metric

    sql_exp_id = tracking_stores.sql.create_experiment("metric-exp")
    ddb_exp_id = tracking_stores.ddb.create_experiment("metric-exp")

    import time

    now = int(time.time() * 1000)
    sql_run = tracking_stores.sql.create_run(sql_exp_id, "user1", now, [], "test-run")
    ddb_run = tracking_stores.ddb.create_run(ddb_exp_id, "user1", now, [], "test-run")
    assert_entities_match(sql_run.info, ddb_run.info, RUN_INFO)

    # Log a metric — this is where Decimal vs float bugs live
    sql_metric = Metric("accuracy", 0.95, 1000, 0)
    ddb_metric = Metric("accuracy", 0.95, 1000, 0)
    tracking_stores.sql.log_metric(sql_run.info.run_id, sql_metric)
    tracking_stores.ddb.log_metric(ddb_run.info.run_id, ddb_metric)

    sql_run_data = tracking_stores.sql.get_run(sql_run.info.run_id)
    ddb_run_data = tracking_stores.ddb.get_run(ddb_run.info.run_id)

    # Compare metrics
    assert len(sql_run_data.data.metrics) == len(ddb_run_data.data.metrics)
    for key in sql_run_data.data.metrics:
        sql_val = sql_run_data.data.metrics[key]
        ddb_val = ddb_run_data.data.metrics[key]
        assert type(sql_val) is type(ddb_val), (
            f"Metric '{key}': type mismatch sql={type(sql_val).__name__} "
            f"ddb={type(ddb_val).__name__}"
        )
        assert sql_val == ddb_val


def test_log_params_and_tags(tracking_stores):
    """Params and tags must round-trip identically."""
    from mlflow.entities import Param, RunTag

    sql_exp_id = tracking_stores.sql.create_experiment("param-exp")
    ddb_exp_id = tracking_stores.ddb.create_experiment("param-exp")

    import time

    now = int(time.time() * 1000)
    sql_run = tracking_stores.sql.create_run(sql_exp_id, "user1", now, [], "test-run")
    ddb_run = tracking_stores.ddb.create_run(ddb_exp_id, "user1", now, [], "test-run")

    tracking_stores.sql.log_param(sql_run.info.run_id, Param("lr", "0.001"))
    tracking_stores.ddb.log_param(ddb_run.info.run_id, Param("lr", "0.001"))

    tracking_stores.sql.set_tag(sql_run.info.run_id, RunTag("env", "prod"))
    tracking_stores.ddb.set_tag(ddb_run.info.run_id, RunTag("env", "prod"))

    sql_data = tracking_stores.sql.get_run(sql_run.info.run_id)
    ddb_data = tracking_stores.ddb.get_run(ddb_run.info.run_id)

    assert sql_data.data.params == ddb_data.data.params
    # Compare tags (filter out internal mlflow.* tags that may differ)
    sql_tags = {k: v for k, v in sql_data.data.tags.items() if not k.startswith("mlflow.")}
    ddb_tags = {k: v for k, v in ddb_data.data.tags.items() if not k.startswith("mlflow.")}
    assert sql_tags == ddb_tags
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/compatibility/test_tracking_contract.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/test_tracking_contract.py
git commit -m "feat(tests): add Phase 1 tracking contract fidelity tests

Focused on operations with historical mismatches: metric types,
experiment fields, run info, params, and tags."
```

---

### Task 8: Phase 2a — Registry compat tests (imported from MLflow)

**Files:**
- Create: `tests/compatibility/test_registry_compat.py`

- [ ] **Step 1: Write the import module**

The curated list excludes:
- `test_parse_search_registered_models_order_by` — no fixture, pure unit test for SQL internals
- `test_webhook_secret_encryption` — accesses `store.engine.connect()`
- All webhook tests initially (DynamoDB store may not implement webhooks)

```python
# tests/compatibility/test_registry_compat.py
"""Phase 2a: MLflow registry store tests run against DynamoDB.

Test functions are imported from the vendored MLflow test suite.
Our conftest provides the `store` fixture backed by DynamoDB (moto).

Excluded tests:
- test_parse_search_registered_models_order_by: pure unit test, no store fixture
- test_webhook_secret_encryption: accesses store.engine (SqlAlchemy-specific)
- Tests with cached_db/db_uri/workspaces_enabled fixture dependencies
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.model_registry.test_sqlalchemy_store import (  # noqa: E402, F401
    test_create_registered_model,
    test_get_registered_model,
    test_update_registered_model,
    test_rename_registered_model,
    test_delete_registered_model,
    test_get_latest_versions,
    test_set_registered_model_tag,
    test_delete_registered_model_tag,
    test_create_model_version,
    test_update_model_version,
    test_delete_model_version,
    test_delete_model_version_redaction,
    test_get_model_version_download_uri,
    test_search_model_versions,
    test_search_model_versions_order_by_simple,
    test_search_model_versions_order_by_errors,
    test_search_model_versions_pagination,
    test_search_model_versions_by_tag,
    test_search_registered_models,
    test_search_registered_models_by_tag,
    test_search_registered_model_pagination,
    test_search_registered_model_order_by,
    test_search_registered_model_order_by_errors,
    test_set_model_version_tag,
    test_delete_model_version_tag,
    test_set_registered_model_alias,
    test_delete_registered_model_alias,
    test_get_model_version_by_alias,
    test_delete_model_version_deletes_alias,
    test_delete_model_deletes_alias,
    test_copy_model_version,
    test_search_prompts,
    test_search_prompts_versions,
    test_search_prompt_versions,
    test_create_registered_model_handle_prompt_properly,
    test_create_model_version_with_model_id_and_no_run_id,
    test_transition_model_version_stage_when_archive_existing_versions_is_false,
    test_transition_model_version_stage_when_archive_existing_versions_is_true,
)
```

- [ ] **Step 2: Run to see collection and initial results**

```bash
uv run pytest tests/compatibility/test_registry_compat.py --collect-only 2>&1 | head -20
```

Expected: tests are collected. Some may have fixture issues — note them.

```bash
uv run pytest tests/compatibility/test_registry_compat.py -v --tb=short 2>&1 | tail -30
```

Note which tests pass and which fail. Failures are expected — they reveal what the DynamoDB store doesn't implement yet.

- [ ] **Step 3: Mark known-failing tests with xfail if needed**

If specific tests consistently fail due to unimplemented features (e.g., webhooks, prompts), add xfail markers:

```python
import pytest

# Re-import and mark as needed
test_search_prompts = pytest.mark.xfail(reason="prompts not yet implemented")(test_search_prompts)
```

- [ ] **Step 4: Commit**

```bash
git add tests/compatibility/test_registry_compat.py
git commit -m "feat(tests): add Phase 2a registry compat tests from MLflow

Import 38 curated test functions from MLflow's registry store test
suite. Runs against DynamoDB via moto."
```

---

### Task 9: Phase 2a — Workspace compat tests (imported from MLflow)

**Files:**
- Create: `tests/compatibility/test_workspace_compat.py`

- [ ] **Step 1: Write the import module**

MLflow's workspace tests use `workspace_store` fixture. Our conftest already provides this.

```python
# tests/compatibility/test_workspace_compat.py
"""Phase 2a: MLflow workspace store tests run against DynamoDB.

MLflow's workspace tests use a `workspace_store` fixture.
Our conftest provides this backed by DynamoDB (moto).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.workspace.test_sqlalchemy_store import (  # noqa: E402, F401
    test_list_workspaces_returns_all,
    test_get_workspace_success,
    test_get_workspace_not_found,
    test_create_workspace_persists_record,
    test_create_workspace_duplicate_raises,
    test_create_workspace_invalid_name_raises,
    test_update_workspace_changes_description,
    test_update_workspace_sets_default_artifact_root,
    test_update_workspace_can_clear_default_artifact_root,
    test_delete_workspace_removes_empty_workspace,
    test_delete_default_workspace_rejected,
    test_delete_workspace_restrict_blocks_when_resources_exist,
    test_delete_workspace_restrict_allows_empty_workspace,
    # Exclude: test_delete_workspace_reassigns_resources_to_default — requires tracking store
    # Exclude: test_delete_workspace_cascade_removes_resources — requires tracking store
    # Exclude: test_delete_workspace_cascade_removes_experiment_with_runs — requires tracking store
    # Exclude: test_resolve_artifact_root_* — SqlAlchemy-specific artifact root caching
)
```

- [ ] **Step 2: Run and check results**

```bash
uv run pytest tests/compatibility/test_workspace_compat.py -v --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/test_workspace_compat.py
git commit -m "feat(tests): add Phase 2a workspace compat tests from MLflow

Import curated workspace CRUD tests from MLflow's workspace store
test suite."
```

---

### Task 10: Phase 2b — Registry workspace isolation compat tests

**Files:**
- Create: `tests/compatibility/test_registry_workspace_compat.py`

- [ ] **Step 1: Write the import module**

Exclude tests that require `cached_db`/`db_uri` fixtures.

```python
# tests/compatibility/test_registry_workspace_compat.py
"""Phase 2b: MLflow registry workspace isolation tests run against DynamoDB.

Uses `workspace_registry_store` fixture (LOAD-BEARING name — must match
MLflow's fixture name in test_sqlalchemy_workspace_store.py).

Expected: these tests will FAIL until DynamoDB stores read from
WorkspaceContext instead of hard-coding self._workspace = "default".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.model_registry.test_sqlalchemy_workspace_store import (  # noqa: E402, F401
    test_registered_model_operations_are_workspace_scoped,
    test_model_version_operations_are_workspace_scoped,
    test_model_version_read_helpers_are_workspace_scoped,
    test_same_model_name_allowed_in_different_workspaces,
    test_update_and_delete_registered_model_metadata_are_workspace_scoped,
    test_webhook_operations_are_workspace_scoped,
    test_default_workspace_context_allows_operations,
    # Excluded: test_default_workspace_behavior_when_workspaces_disabled — creates SqlAlchemyStore
    # Excluded: test_single_tenant_registry_startup_rejects_non_default_workspace_models — uses cached_db
    # Excluded: test_model_version_allows_workspace_scoped_proxied_artifacts — uses monkeypatch on store internals
)

# Mark all as xfail until workspace support is implemented
test_registered_model_operations_are_workspace_scoped = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_registered_model_operations_are_workspace_scoped)

test_model_version_operations_are_workspace_scoped = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_model_version_operations_are_workspace_scoped)

test_model_version_read_helpers_are_workspace_scoped = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_model_version_read_helpers_are_workspace_scoped)

test_same_model_name_allowed_in_different_workspaces = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_same_model_name_allowed_in_different_workspaces)

test_update_and_delete_registered_model_metadata_are_workspace_scoped = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_update_and_delete_registered_model_metadata_are_workspace_scoped)

test_webhook_operations_are_workspace_scoped = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_webhook_operations_are_workspace_scoped)

test_default_workspace_context_allows_operations = pytest.mark.xfail(
    reason="DynamoDB stores hard-code workspace='default'"
)(test_default_workspace_context_allows_operations)
```

- [ ] **Step 2: Run and verify xfail behavior**

```bash
uv run pytest tests/compatibility/test_registry_workspace_compat.py -v
```

Expected: all tests show as `xfail` (expected failure).

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/test_registry_workspace_compat.py
git commit -m "feat(tests): add Phase 2b registry workspace isolation compat tests

Import workspace isolation tests from MLflow. All xfail until
DynamoDB stores implement WorkspaceContext support."
```

---

### Task 11: Phase 2b — Tracking workspace isolation compat tests

**Files:**
- Create: `tests/compatibility/test_tracking_workspace_compat.py`

- [ ] **Step 1: Write the import module**

Only import tests that use `workspace_tracking_store` fixture (not `db_uri`, `tmp_path`, `gateway_workspace_store`).

```python
# tests/compatibility/test_tracking_workspace_compat.py
"""Phase 2b: MLflow tracking workspace isolation tests run against DynamoDB.

Uses `workspace_tracking_store` fixture (LOAD-BEARING name).

Expected: these tests will FAIL until DynamoDB stores read from
WorkspaceContext instead of hard-coding self._workspace = "default".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.tracking.test_sqlalchemy_workspace_store import (  # noqa: E402, F401
    test_experiments_are_workspace_scoped,
    test_runs_are_workspace_scoped,
    test_search_datasets_is_workspace_scoped,
    test_entity_associations_are_workspace_scoped,
    test_run_data_logging_enforces_workspaces,
    test_run_lifecycle_operations_workspace_isolation,
    test_search_and_history_calls_are_workspace_scoped,
    test_experiment_lifecycle_operations_are_workspace_scoped,
    test_experiment_tags_are_workspace_scoped,
    test_trace_tag_operations_are_workspace_scoped,
    test_search_traces_is_workspace_scoped,
    # Excluded: tests requiring db_uri, tmp_path, gateway_workspace_store fixtures
    # Excluded: test_sqlalchemy_store_returns_workspace_aware_when_enabled — instantiates SqlAlchemy directly
    # Excluded: test_sqlalchemy_store_is_single_tenant_when_disabled — instantiates SqlAlchemy directly
    # Excluded: test_artifact_* — backend-specific artifact path assertions
)

# Mark all as xfail until workspace support is implemented
_xfail = pytest.mark.xfail(reason="DynamoDB stores hard-code workspace='default'")

test_experiments_are_workspace_scoped = _xfail(test_experiments_are_workspace_scoped)
test_runs_are_workspace_scoped = _xfail(test_runs_are_workspace_scoped)
test_search_datasets_is_workspace_scoped = _xfail(test_search_datasets_is_workspace_scoped)
test_entity_associations_are_workspace_scoped = _xfail(
    test_entity_associations_are_workspace_scoped
)
test_run_data_logging_enforces_workspaces = _xfail(test_run_data_logging_enforces_workspaces)
test_run_lifecycle_operations_workspace_isolation = _xfail(
    test_run_lifecycle_operations_workspace_isolation
)
test_search_and_history_calls_are_workspace_scoped = _xfail(
    test_search_and_history_calls_are_workspace_scoped
)
test_experiment_lifecycle_operations_are_workspace_scoped = _xfail(
    test_experiment_lifecycle_operations_are_workspace_scoped
)
test_experiment_tags_are_workspace_scoped = _xfail(test_experiment_tags_are_workspace_scoped)
test_trace_tag_operations_are_workspace_scoped = _xfail(
    test_trace_tag_operations_are_workspace_scoped
)
test_search_traces_is_workspace_scoped = _xfail(test_search_traces_is_workspace_scoped)
```

- [ ] **Step 2: Run and verify xfail behavior**

```bash
uv run pytest tests/compatibility/test_tracking_workspace_compat.py -v
```

Expected: all tests show as `xfail`.

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/test_tracking_workspace_compat.py
git commit -m "feat(tests): add Phase 2b tracking workspace isolation compat tests

Import 11 tracking workspace isolation tests from MLflow. All xfail
until DynamoDB stores implement WorkspaceContext support."
```

---

### Task 12: CI integration and final verification

**Files:**
- Modify: `.github/workflows/` (if CI workflow exists)

- [ ] **Step 1: Run the full compatibility test suite**

```bash
uv run pytest -m compatibility -v --tb=short 2>&1 | tail -40
```

Verify:
- Phase 1 contract tests run (some may fail — expected, reveals real mismatches)
- Phase 2a tests are collected and run
- Phase 2b tests show as xfail

- [ ] **Step 2: Check that existing tests still pass**

```bash
uv run pytest -m "unit or integration" --tb=short 2>&1 | tail -10
```

Expected: all existing tests still pass — the compatibility suite is isolated.

- [ ] **Step 3: Update CI workflow if needed**

If `.github/workflows/ci.yml` exists, ensure it has:
```yaml
- uses: actions/checkout@v4
  with:
    submodules: true
```

And add a compatibility test job or step:
```yaml
- name: Run compatibility tests
  run: uv run pytest -m compatibility -v
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: verify CI integration for compatibility tests"
```
