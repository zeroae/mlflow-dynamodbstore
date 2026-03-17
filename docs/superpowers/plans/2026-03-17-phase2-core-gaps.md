# Phase 2: Core Gaps Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two genuinely missing core tracking store methods (`delete_experiment_tag`, `supports_workspaces`) to complete the base tracking store API.

**Architecture:** Both methods follow existing patterns in the codebase. `delete_experiment_tag` mirrors the existing `delete_tag` (run tag deletion) pattern — delete the tag item, remove from denormalized META map. No FTS cleanup needed since experiment tags are not FTS-indexed. `supports_workspaces` reads the `MLFLOW_ENABLE_WORKSPACES` env var, consistent with the existing workspace store and configuration docs.

**Tech Stack:** Python, moto (testing), DynamoDB single-table design

**Note on scope:** The original Phase 2 audit identified 5 methods (`search_runs`, `log_metric`, `log_param`, `delete_experiment_tag`, `supports_workspaces`). Investigation revealed that `search_runs`, `log_metric`, and `log_param` already work — `AbstractStore` provides concrete implementations that delegate to `_search_runs()` and `log_batch()`, both of which are implemented. Only 2 methods need new code.

---

### Task 1: Implement `delete_experiment_tag`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py:728-753` (experiment tags section)
- Test: `tests/unit/test_tracking_store.py`

- [ ] **Step 1: Write the failing unit test**

Add to `tests/unit/test_tracking_store.py` in the experiment tests section, after `test_set_experiment_tag`:

```python
def test_delete_experiment_tag(self, tracking_store):
    exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
    tag = ExperimentTag("my-key", "my-value")
    tracking_store.set_experiment_tag(exp_id, tag)
    exp = tracking_store.get_experiment(exp_id)
    assert exp.tags["my-key"] == "my-value"

    tracking_store.delete_experiment_tag(exp_id, "my-key")
    exp = tracking_store.get_experiment(exp_id)
    assert "my-key" not in exp.tags

def test_delete_experiment_tag_nonexistent_is_silent(self, tracking_store):
    exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
    # Should not raise — idempotent delete
    tracking_store.delete_experiment_tag(exp_id, "does-not-exist")

def test_delete_experiment_tag_nonexistent_experiment_raises(self, tracking_store):
    from mlflow.exceptions import MlflowException
    with pytest.raises(MlflowException, match="does not exist"):
        tracking_store.delete_experiment_tag("nonexistent-id", "key")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_store.py::TestExperiments::test_delete_experiment_tag tests/unit/test_tracking_store.py::TestExperiments::test_delete_experiment_tag_nonexistent_is_silent tests/unit/test_tracking_store.py::TestExperiments::test_delete_experiment_tag_nonexistent_experiment_raises -v`

Expected: FAIL — `delete_experiment_tag` is a no-op stub inherited from `AbstractStore`.

- [ ] **Step 3: Implement `delete_experiment_tag`**

Add to `src/mlflow_dynamodbstore/tracking_store.py` after `set_experiment_tag` (around line 733):

```python
def delete_experiment_tag(self, experiment_id: str, key: str) -> None:
    """Delete a tag from an experiment."""
    # Verify experiment exists
    self.get_experiment(experiment_id)
    pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
    sk = f"{SK_EXPERIMENT_TAG_PREFIX}{key}"
    self._table.delete_item(pk=pk, sk=sk)
    if self._config.should_denormalize(None, key):
        self._remove_denormalized_tag(pk, SK_EXPERIMENT_META, key)
```

This mirrors the existing `delete_tag` (run tag deletion) pattern at line 1450-1457, adapted for experiment tags. No FTS cleanup needed since experiment tags do not have FTS indexing (only run and trace tags do — see `_write_run_tag` and `_write_trace_tag`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_store.py::TestExperiments::test_delete_experiment_tag tests/unit/test_tracking_store.py::TestExperiments::test_delete_experiment_tag_nonexistent_is_silent tests/unit/test_tracking_store.py::TestExperiments::test_delete_experiment_tag_nonexistent_experiment_raises -v`

Expected: PASS

- [ ] **Step 5: Run full unit test suite**

Run: `uv run pytest tests/unit/ -v --tb=short`

Expected: All pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_store.py
git commit -m "feat: implement delete_experiment_tag"
```

---

### Task 2: Implement `supports_workspaces`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py:199-216` (constructor / class definition area)
- Test: `tests/unit/test_tracking_store.py`

**Design note:** The DynamoDB store's schema has workspace scoping built into every entity (GSI2/GSI3 prefixes include workspace, META items carry workspace attribute). The store always supports workspaces — the user controls whether to enable them at server startup via `--enable-workspaces` / `MLFLOW_ENABLE_WORKSPACES=1`. The SQLAlchemy store returns `False` because it lacks workspace schema; we return `True` because we have it.

- [ ] **Step 1: Write the failing unit tests**

Add to `tests/unit/test_tracking_store.py`:

```python
def test_supports_workspaces(self, tracking_store):
    """DynamoDB store always supports workspaces."""
    assert tracking_store.supports_workspaces is True
```

Add to `tests/unit/test_workspace_store.py` (existing file):

```python
def test_create_and_delete_workspace(self, workspace_store):
    """Create a workspace, verify it exists, delete it, verify gone."""
    workspace_store.create_workspace("test-ws", description="Unit test")
    ws = workspace_store.get_workspace("test-ws")
    assert ws["name"] == "test-ws"
    assert ws["description"] == "Unit test"

    workspace_store.delete_workspace("test-ws")
    assert workspace_store.get_workspace("test-ws") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_store.py -k "supports_workspaces" tests/unit/test_workspace_store.py -k "create_and_delete" -v`

Expected: `supports_workspaces` FAILS (inherited `AbstractStore` returns `False`). The workspace CRUD test may already pass if that functionality is implemented — verify.

- [ ] **Step 3: Implement `supports_workspaces` property**

Add to `src/mlflow_dynamodbstore/tracking_store.py` in the `DynamoDBTrackingStore` class, after `__init__`:

```python
@property
def supports_workspaces(self) -> bool:
    """DynamoDB store always supports workspaces.

    Workspace scoping is built into the schema (GSI2/GSI3 prefixes,
    META workspace attribute). The --enable-workspaces server flag
    controls whether workspace features are active at runtime.
    """
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tracking_store.py -k "supports_workspaces" -v`

Expected: PASS

- [ ] **Step 5: Run full unit test suite**

Run: `uv run pytest tests/unit/ -v --tb=short`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_store.py
git commit -m "feat: implement supports_workspaces property (always True)"
```

---

### Task 3: E2E tests — delete_experiment_tag + workspace CRUD

**Files:**
- Modify: `tests/e2e/conftest.py` (enable workspaces on the e2e server)
- Modify: `tests/e2e/test_experiments.py`
- Create: `tests/e2e/test_workspaces.py`

- [ ] **Step 1: Enable workspaces on the e2e server**

In `tests/e2e/conftest.py`, add `MLFLOW_ENABLE_WORKSPACES=true` to the env dict in both `_start_mlflow_moto` and `_start_mlflow_aws`. This is safe because all existing tests use the "default" workspace implicitly.

In `_start_mlflow_moto` (around line 115):
```python
env["MLFLOW_ENABLE_WORKSPACES"] = "true"
```

In `_start_mlflow_aws` (around line 197):
```python
env["MLFLOW_ENABLE_WORKSPACES"] = "true"
```

- [ ] **Step 2: Add delete_experiment_tag e2e test**

Add after `test_set_experiment_tag` in `tests/e2e/test_experiments.py`:

```python
def test_delete_experiment_tag(self, client: MlflowClient):
    exp_id = client.create_experiment(f"e2e-deltag-{_uid()}")
    client.set_experiment_tag(exp_id, "team", "ml-platform")
    exp = client.get_experiment(exp_id)
    assert exp.tags["team"] == "ml-platform"

    client.delete_experiment_tag(exp_id, "team")
    exp = client.get_experiment(exp_id)
    assert "team" not in exp.tags
```

- [ ] **Step 3: Create workspace CRUD e2e tests**

Create `tests/e2e/test_workspaces.py`:

```python
"""E2E tests for workspace CRUD via REST API."""

import uuid

import pytest
import requests

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestWorkspaces:
    def test_list_workspaces_has_default(self, mlflow_server):
        """Default workspace exists on startup."""
        resp = requests.get(
            f"{mlflow_server}/ajax-api/2.0/mlflow/workspaces",
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        names = [ws["name"] for ws in data.get("workspaces", [])]
        assert "default" in names

    def test_create_and_delete_workspace(self, mlflow_server):
        """Create a workspace, verify it exists, then delete it."""
        ws_name = f"e2e-ws-{_uid()}"

        # Create
        resp = requests.post(
            f"{mlflow_server}/ajax-api/2.0/mlflow/workspaces",
            json={"name": ws_name, "description": "test workspace"},
            timeout=10,
        )
        assert resp.status_code == 200

        # Get
        resp = requests.get(
            f"{mlflow_server}/ajax-api/2.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["workspace"]["name"] == ws_name

        # Delete
        resp = requests.delete(
            f"{mlflow_server}/ajax-api/2.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 200

        # Verify deleted
        resp = requests.get(
            f"{mlflow_server}/ajax-api/2.0/mlflow/workspaces/{ws_name}",
            timeout=10,
        )
        assert resp.status_code == 404
```

- [ ] **Step 4: Run e2e tests**

Run: `uv run pytest tests/e2e/test_experiments.py tests/e2e/test_workspaces.py -v`

Expected: All pass. If workspace endpoints return 503, check that `MLFLOW_ENABLE_WORKSPACES` is set in conftest.

- [ ] **Step 5: Run full e2e suite to verify no regressions from enabling workspaces**

Run: `uv run pytest tests/e2e/ -v --tb=short`

Expected: All pass (enabling workspaces should not break existing tests since they all use the "default" workspace implicitly).

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/conftest.py tests/e2e/test_experiments.py tests/e2e/test_workspaces.py
git commit -m "test: add e2e tests for delete_experiment_tag and workspace CRUD

Enable --enable-workspaces on e2e server to test workspace REST
endpoints (create, get, list, delete)."
```

---

### Task 4: Verify coverage and finalize

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/unit/ tests/integration/ tests/e2e/ -v --tb=short`

Expected: All pass.

- [ ] **Step 2: Verify patch coverage**

Run: `uv run pytest tests/unit/ --cov=mlflow_dynamodbstore.tracking_store --cov-report=term-missing`

Verify that all new lines in `tracking_store.py` (the `delete_experiment_tag` method and `supports_workspaces` property) are covered.

- [ ] **Step 3: Final commit if any cleanup needed**

Only if linting or minor adjustments are needed.
