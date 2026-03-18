# Phase 4a: Remaining Store Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 3 store methods (`log_outputs`, `log_spans_async`, `get_metric_history_bulk_interval_from_steps`) to close the simple gaps in AbstractStore parity.

**Architecture:** Each method follows existing patterns: `log_outputs` writes run sub-items (same pattern as `log_batch`), `log_spans_async` is a one-line delegate to `log_spans` (Phase 3), and `get_metric_history_bulk_interval_from_steps` queries the existing `#MHIST#` prefix with an in-memory step filter.

**Tech Stack:** Python 3.11, boto3 (DynamoDB resource API), moto (testing)

**Spec:** `docs/superpowers/specs/2026-03-17-remaining-parity-design.md`

---

### Task 1: Schema constant + `log_outputs`

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py`
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Test: `tests/unit/test_tracking_store.py`

- [ ] **Step 1: Add schema constant**

Add to `src/mlflow_dynamodbstore/dynamodb/schema.py` after `SK_LOGGED_MODEL_PREFIX`:

```python
SK_OUTPUT_PREFIX = "#OUTPUT#"
```

- [ ] **Step 2: Write the failing tests**

Add a new test class at the end of `tests/unit/test_tracking_store.py`:

```python
class TestLogOutputs:
    """Tests for log_outputs — run-to-model associations."""

    def test_log_single_output(self, tracking_store):
        from mlflow.entities import LoggedModelOutput

        exp_id = tracking_store.create_experiment("test-outputs", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r1",
        )
        run_id = run.info.run_id

        tracking_store.log_outputs(run_id, [LoggedModelOutput(model_id="m-abc", step=1)])

        # Verify item exists under the run's SK prefix
        pk = f"EXP#{exp_id}"
        items = tracking_store._table.query(pk=pk, sk_prefix=f"R#{run_id}#OUTPUT#")
        assert len(items) == 1
        assert items[0]["destination_id"] == "m-abc"
        assert items[0]["step"] == 1

    def test_log_multiple_outputs(self, tracking_store):
        from mlflow.entities import LoggedModelOutput

        exp_id = tracking_store.create_experiment("test-outputs-multi", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r2",
        )
        run_id = run.info.run_id

        models = [
            LoggedModelOutput(model_id="m-1", step=1),
            LoggedModelOutput(model_id="m-2", step=2),
            LoggedModelOutput(model_id="m-3", step=3),
        ]
        tracking_store.log_outputs(run_id, models)

        pk = f"EXP#{exp_id}"
        items = tracking_store._table.query(pk=pk, sk_prefix=f"R#{run_id}#OUTPUT#")
        assert len(items) == 3
        model_ids = {item["destination_id"] for item in items}
        assert model_ids == {"m-1", "m-2", "m-3"}

    def test_log_output_nonexistent_run_raises(self, tracking_store):
        from mlflow.entities import LoggedModelOutput
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.log_outputs("nonexistent-run", [LoggedModelOutput(model_id="m-x", step=0)])

    def test_log_output_deleted_run_raises(self, tracking_store):
        from mlflow.entities import LoggedModelOutput
        from mlflow.exceptions import MlflowException

        exp_id = tracking_store.create_experiment("test-outputs-del", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r-del",
        )
        run_id = run.info.run_id
        tracking_store.delete_run(run_id)

        with pytest.raises(MlflowException):
            tracking_store.log_outputs(run_id, [LoggedModelOutput(model_id="m-x", step=0)])

    def test_log_output_empty_list_is_noop(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-outputs-empty", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r3",
        )
        # Should not raise
        tracking_store.log_outputs(run.info.run_id, [])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_store.py::TestLogOutputs -v`
Expected: FAIL — `log_outputs` raises `NotImplementedError` from `AbstractStore`.

- [ ] **Step 4: Implement `log_outputs`**

Add two imports to `tracking_store.py`:

1. Schema import — add `SK_OUTPUT_PREFIX` to the existing `from mlflow_dynamodbstore.dynamodb.schema import (...)` block:
```python
SK_OUTPUT_PREFIX,
```

2. Entity import — add `LoggedModelOutput` to the existing `from mlflow.entities import (...)` block:
```python
LoggedModelOutput,
```

Add method to `DynamoDBTrackingStore` (near `log_batch`, around line 1879):

```python
    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]) -> None:
        """Associate logged model outputs with a run."""
        if not models:
            return

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

        # Verify run is active (not deleted)
        meta = self._table.get_item(pk=pk, sk=f"{SK_RUN_PREFIX}{run_id}")
        if meta is None:
            raise MlflowException(
                f"Run '{run_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if meta.get("lifecycle_stage") == "deleted":
            raise MlflowException(
                f"Run '{run_id}' is deleted.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        items: list[dict[str, Any]] = []
        for model in models:
            output_id = generate_ulid()
            items.append({
                "PK": pk,
                "SK": f"{SK_RUN_PREFIX}{run_id}{SK_OUTPUT_PREFIX}{output_id}",
                "source_type": "RUN_OUTPUT",
                "source_id": run_id,
                "destination_type": "MODEL_OUTPUT",
                "destination_id": model.model_id,
                "step": model.step,
            })

        self._table.batch_write(items)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_store.py::TestLogOutputs -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/schema.py src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_store.py
git commit -m "feat: implement log_outputs for run-to-model associations"
```

---

### Task 2: `log_spans_async`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Test: `tests/unit/test_tracking_traces.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_tracking_traces.py`:

```python
class TestLogSpansAsync:
    """Tests for log_spans_async."""

    @staticmethod
    def _make_mock_span(trace_id, name="root"):
        span = MagicMock()
        span.trace_id = trace_id
        span.to_dict.return_value = {"name": name, "trace_id": trace_id, "span_id": "s1"}
        return span

    def test_async_delegates_to_sync(self, tracking_store):
        import asyncio

        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-async-1")
        tracking_store.start_trace(trace_info)

        span = self._make_mock_span("tr-async-1")
        result = asyncio.run(tracking_store.log_spans_async(exp_id, [span]))
        assert len(result) == 1

        # Verify SPANS item was written
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        cached = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-async-1#SPANS")
        assert cached is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestLogSpansAsync -v`
Expected: FAIL — `log_spans_async` raises `NotImplementedError`.

- [ ] **Step 3: Implement `log_spans_async`**

Add to `DynamoDBTrackingStore` (near `log_spans`):

```python
    async def log_spans_async(self, location: str, spans: list[Any]) -> list[Any]:
        """Async version of log_spans — delegates to synchronous implementation."""
        return self.log_spans(location, spans)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestLogSpansAsync -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat: implement log_spans_async (delegates to log_spans)"
```

---

### Task 3: `get_metric_history_bulk_interval_from_steps`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Test: `tests/unit/test_tracking_store.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_tracking_store.py`:

```python
class TestGetMetricHistoryBulkIntervalFromSteps:
    """Tests for get_metric_history_bulk_interval_from_steps."""

    def _log_metric_history(self, tracking_store, run_id, key, steps):
        """Helper: log metrics at specific steps."""
        from mlflow.entities import Metric

        metrics = [Metric(key=key, value=float(s) * 0.1, timestamp=1000 + s, step=s) for s in steps]
        tracking_store.log_batch(run_id, metrics=metrics, params=[], tags=[])

    def test_returns_only_requested_steps(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r1",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "loss", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id, metric_key="loss", steps=[3, 7], max_results=100,
        )
        steps_returned = [m.step for m in result]
        assert steps_returned == [3, 7]

    def test_missing_steps_silently_skipped(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-skip", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r2",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "acc", [1, 3, 5])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id, metric_key="acc", steps=[1, 2, 3, 4, 5], max_results=100,
        )
        steps_returned = [m.step for m in result]
        assert steps_returned == [1, 3, 5]

    def test_sorted_by_step_timestamp(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-sort", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r3",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "lr", [5, 1, 3])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id, metric_key="lr", steps=[5, 1, 3], max_results=100,
        )
        steps_returned = [m.step for m in result]
        assert steps_returned == [1, 3, 5]

    def test_max_results_limits_output(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-max", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r4",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "loss", list(range(1, 21)))

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id, metric_key="loss", steps=list(range(1, 21)), max_results=5,
        )
        assert len(result) == 5

    def test_empty_steps_returns_empty(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-mhbi-empty", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r5",
        )
        run_id = run.info.run_id

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id, metric_key="loss", steps=[], max_results=100,
        )
        assert result == []

    def test_returns_metric_with_run_id(self, tracking_store):
        """Verify return type is list[MetricWithRunId] with run_id attribute."""
        exp_id = tracking_store.create_experiment("test-mhbi-type", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r6",
        )
        run_id = run.info.run_id
        self._log_metric_history(tracking_store, run_id, "val", [1])

        result = tracking_store.get_metric_history_bulk_interval_from_steps(
            run_id=run_id, metric_key="val", steps=[1], max_results=100,
        )
        assert len(result) == 1
        assert result[0].run_id == run_id
        assert result[0].key == "val"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_store.py::TestGetMetricHistoryBulkIntervalFromSteps -v`
Expected: Tests pass using the inherited default implementation (which loads full history + filters). The override is a performance optimization — the tests verify correctness, not speed.

Note: If the default implementation already passes all tests, that's expected. The override avoids constructing unnecessary `Metric` objects for runs with large histories. We still implement it for efficiency.

- [ ] **Step 3: Implement `get_metric_history_bulk_interval_from_steps`**

Add to `DynamoDBTrackingStore` (near `get_metric_history`):

```python
    def get_metric_history_bulk_interval_from_steps(
        self, run_id: str, metric_key: str, steps: list[int], max_results: int
    ) -> list[MetricWithRunId]:
        """Return metric history for specific steps, optimized for DynamoDB."""
        from mlflow.entities.metric import MetricWithRunId

        if not steps:
            return []

        experiment_id = self._resolve_run_experiment(run_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        prefix = f"{SK_RUN_PREFIX}{run_id}{SK_METRIC_HISTORY_PREFIX}{metric_key}#"

        steps_set = set(steps)
        items = self._table.query(pk=pk, sk_prefix=prefix)

        metrics = sorted(
            [
                Metric(
                    key=item["key"],
                    value=float(item["value"]),
                    timestamp=int(item.get("timestamp", 0)),
                    step=int(item.get("step", 0)),
                )
                for item in items
                if int(item.get("step", 0)) in steps_set
            ],
            key=lambda m: (m.step, m.timestamp),
        )[:max_results]

        return [MetricWithRunId(run_id=run_id, metric=m) for m in metrics]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_store.py::TestGetMetricHistoryBulkIntervalFromSteps -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_store.py
git commit -m "feat: implement get_metric_history_bulk_interval_from_steps"
```

---

### Task 4: Integration tests + coverage verification

**Files:**
- Modify: `tests/integration/test_traces.py`
- Modify: `tests/e2e/test_traces.py`

- [ ] **Step 1: Add integration test for log_outputs**

Add to `tests/integration/test_traces.py` (or create new `tests/integration/test_runs.py` if more appropriate — check which file has run tests):

```python
class TestLogOutputsIntegration:
    def test_log_outputs_round_trip(self, tracking_store):
        from mlflow.entities import LoggedModelOutput

        from mlflow_dynamodbstore.dynamodb.schema import PK_EXPERIMENT_PREFIX

        exp_id = tracking_store.create_experiment("test-log-outputs", artifact_location="s3://b")
        run = tracking_store.create_run(
            experiment_id=exp_id, user_id="user", start_time=1000, tags=[], run_name="r1",
        )
        tracking_store.log_outputs(run.info.run_id, [LoggedModelOutput(model_id="m-1", step=1)])

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        items = tracking_store._table.query(pk=pk, sk_prefix=f"R#{run.info.run_id}#OUTPUT#")
        assert len(items) == 1
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/integration/ -v --tb=short`
Expected: All pass.

- [ ] **Step 3: Run full unit test suite**

Run: `uv run pytest tests/unit/ -v --tb=short -q`
Expected: All pass.

- [ ] **Step 4: Run e2e tests**

Run: `uv run pytest tests/e2e/ -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Verify patch coverage**

Run: `uv run pytest tests/unit/ --cov=mlflow_dynamodbstore.tracking_store --cov-report=term-missing`
Verify: All new lines covered.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: add integration tests for Phase 4a methods"
```
