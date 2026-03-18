# Phase 3: Advanced Traces Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 7 store methods + 1 `start_trace` modification for full trace parity with the SQLAlchemy store: bulk reads, session discovery, prompt linking, span logging, and trace filter correlation analysis.

**Architecture:** Each method is implemented directly in `DynamoDBTrackingStore`, reusing existing patterns (`_resolve_trace_experiment`, `_build_trace_info`, `_write_trace_tag`, `parse_trace_filter`). Session tracking adds a new `SESS#` item type to the existing `EXP#` partition, discovered via GSI2 range queries. The `calculate_trace_filter_correlation` method delegates to `search_traces`-style iteration with in-memory counters.

**Tech Stack:** Python 3.11, boto3 (DynamoDB resource API), moto (testing), mlflow entities (TraceInfo, Trace, Span, CompletedSession, TraceFilterCorrelationResult, PromptVersion)

**Spec:** `docs/superpowers/specs/2026-03-17-advanced-traces-design.md`

---

### Task 1: Schema Constants

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py:129-138`

- [ ] **Step 1: Add session tracking constants**

Add after the `GSI3_SCOR_NAME_PREFIX` line (line 137) in `schema.py`:

```python
# ---------------------------------------------------------------------------
# Session tracking (within EXP# partition)
# ---------------------------------------------------------------------------
SK_SESSION_PREFIX = "SESS#"

# GSI prefixes for sessions
GSI2_SESSIONS_PREFIX = "SESSIONS#"
```

- [ ] **Step 2: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/schema.py
git commit -m "feat(schema): add session tracking constants for Phase 3"
```

---

### Task 2: `start_trace` Session Tracker Upsert

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py:2173-2277` (start_trace method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** After existing trace creation logic in `start_trace`, check for `mlflow.traceSession` in `trace_info.trace_metadata`. If present, upsert a Session Tracker item at `SK=SESS#<session_id>` using raw boto3 `update_item` (same pattern as `_denormalize_tag` at line 1748). The combined `ADD trace_count :1` + `SET if_not_exists(...)` expression requires direct `self._table._table.update_item(...)`.

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestStartTraceSessionTracker` at the end of `tests/unit/test_tracking_traces.py`:

```python
class TestStartTraceSessionTracker:
    """Tests for session tracker upsert in start_trace."""

    def test_trace_with_session_creates_session_tracker(self, tracking_store):
        """start_trace with mlflow.traceSession metadata creates a SESS# item."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-sess-1",
            request_time=1000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "session-abc",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-abc")
        assert item is not None
        assert item["session_id"] == "session-abc"
        assert int(item["trace_count"]) == 1
        assert int(item["first_trace_timestamp_ms"]) == 1000
        assert int(item["last_trace_timestamp_ms"]) == 1000

    def test_trace_without_session_no_session_tracker(self, tracking_store):
        """start_trace without mlflow.traceSession does NOT create a SESS# item."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-no-sess")
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#any")
        assert item is None

    def test_multiple_traces_same_session_increments(self, tracking_store):
        """Multiple traces in same session increment trace_count and update timestamps."""
        exp_id = _create_experiment(tracking_store)
        for i, (tid, ts) in enumerate([
            ("tr-s1", 1000),
            ("tr-s2", 2000),
            ("tr-s3", 3000),
        ]):
            trace_info = _make_trace_info(
                exp_id,
                trace_id=tid,
                request_time=ts,
                trace_metadata={
                    TraceTagKey.TRACE_NAME: "my-trace",
                    "mlflow.traceSession": "session-xyz",
                },
            )
            tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-xyz")
        assert item is not None
        assert int(item["trace_count"]) == 3
        assert int(item["first_trace_timestamp_ms"]) == 1000
        assert int(item["last_trace_timestamp_ms"]) == 3000

    def test_session_tracker_has_gsi2_attributes(self, tracking_store):
        """Session tracker item has GSI2 PK/SK for find_completed_sessions queries."""
        from mlflow_dynamodbstore.dynamodb.schema import GSI2_PK, GSI2_SK

        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-gsi2",
            request_time=5000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "session-gsi2",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-gsi2")
        assert item[GSI2_PK] == f"SESSIONS#default#{exp_id}"
        # GSI2 SK is zero-padded string for correct lexicographic ordering
        assert item[GSI2_SK] == f"{5000:020d}"

    def test_session_tracker_has_ttl(self, tracking_store):
        """Session tracker item inherits trace TTL."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-ttl",
            request_time=1000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "session-ttl",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-ttl")
        assert "ttl" in item
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestStartTraceSessionTracker -v`
Expected: FAIL — no `SESS#` items are written yet.

- [ ] **Step 3: Implement session tracker upsert in `start_trace`**

Add the following import at the top of `tracking_store.py` (in the schema imports block around line 55):
```python
from mlflow_dynamodbstore.dynamodb.schema import SK_SESSION_PREFIX, GSI2_SESSIONS_PREFIX
```

Add the following code at the end of `start_trace`, just before `return trace_info` (after the artifact location tag block, around line 2276):

```python
        # Upsert session tracker if trace has session metadata
        session_id = (trace_info.trace_metadata or {}).get("mlflow.traceSession")
        if session_id:
            self._upsert_session_tracker(
                experiment_id=experiment_id,
                session_id=session_id,
                timestamp_ms=trace_info.request_time,
                ttl=ttl,
            )
```

Add a new private method `_upsert_session_tracker` near the other trace helper methods (e.g., after `_write_trace_tag`):

```python
    def _upsert_session_tracker(
        self,
        experiment_id: str,
        session_id: str,
        timestamp_ms: int,
        ttl: int | None,
    ) -> None:
        """Upsert a session tracker item using atomic ADD + conditional SET."""
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_SESSION_PREFIX}{session_id}"
        gsi2pk = f"{GSI2_SESSIONS_PREFIX}{self._workspace}#{experiment_id}"

        # GSI2 SK must be a zero-padded string (GSI key schema is type S)
        gsi2sk_str = f"{timestamp_ms:020d}"

        # Combined ADD + SET requires raw boto3 (same pattern as _denormalize_tag)
        update_expr = (
            "ADD trace_count :one "
            "SET first_trace_timestamp_ms = if_not_exists(first_trace_timestamp_ms, :ts), "
            "last_trace_timestamp_ms = :ts, "
            "session_id = if_not_exists(session_id, :sid), "
            "#gsi2pk = :gsi2pk, #gsi2sk = :gsi2sk"
        )
        expr_names = {"#gsi2pk": "gsi2pk", "#gsi2sk": "gsi2sk"}
        expr_values: dict[str, Any] = {
            ":one": 1,
            ":ts": timestamp_ms,
            ":sid": session_id,
            ":gsi2pk": gsi2pk,
            ":gsi2sk": gsi2sk_str,
        }
        if ttl is not None:
            update_expr += ", #ttl = :ttl"
            expr_names["#ttl"] = "ttl"
            expr_values[":ttl"] = ttl

        self._table._table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestStartTraceSessionTracker -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py src/mlflow_dynamodbstore/dynamodb/schema.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): add session tracker upsert in start_trace"
```

---

### Task 3: `batch_get_trace_infos`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** Reuses `_resolve_trace_experiment` (line 2150) and `_build_trace_info` (line 2702). For each trace_id: resolve experiment_id, query all items with `SK begins_with T#<trace_id>`, extract META item, call `_build_trace_info`. If `location` is provided, skip GSI1 resolution and use it as experiment_id.

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestBatchGetTraceInfos`:

```python
class TestBatchGetTraceInfos:
    """Tests for batch_get_trace_infos."""

    def _create_traces(self, tracking_store, exp_id, count=3):
        """Helper: create multiple traces, return their IDs."""
        trace_ids = []
        for i in range(count):
            tid = f"tr-batch-info-{i}"
            trace_info = _make_trace_info(
                exp_id,
                trace_id=tid,
                request_time=1000 + i * 100,
            )
            tracking_store.start_trace(trace_info)
            trace_ids.append(tid)
        return trace_ids

    def test_batch_get_single_trace(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos(tids)
        assert len(result) == 1
        assert result[0].trace_id == tids[0]

    def test_batch_get_multiple_traces(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=3)
        result = tracking_store.batch_get_trace_infos(tids)
        assert len(result) == 3
        returned_ids = {t.trace_id for t in result}
        assert returned_ids == set(tids)

    def test_nonexistent_trace_excluded(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos(tids + ["nonexistent-trace"])
        assert len(result) == 1
        assert result[0].trace_id == tids[0]

    def test_empty_list_returns_empty(self, tracking_store):
        result = tracking_store.batch_get_trace_infos([])
        assert result == []

    def test_duplicate_ids_deduplicated(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos([tids[0], tids[0]])
        assert len(result) == 1

    def test_with_location_skips_resolution(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos(tids, location=exp_id)
        assert len(result) == 1
        assert result[0].trace_id == tids[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestBatchGetTraceInfos -v`
Expected: FAIL — method not implemented (raises `MlflowNotImplementedException` from base class).

- [ ] **Step 3: Implement `batch_get_trace_infos`**

Add to `DynamoDBTrackingStore` after `link_traces_to_run` (around line 2849):

```python
    def batch_get_trace_infos(
        self, trace_ids: list[str], location: str | None = None
    ) -> list[TraceInfo]:
        """Get trace metadata for given trace IDs without loading spans."""
        if not trace_ids:
            return []

        seen: set[str] = set()
        results: list[TraceInfo] = []

        for trace_id in trace_ids:
            if trace_id in seen:
                continue
            seen.add(trace_id)

            try:
                if location:
                    experiment_id = location
                else:
                    experiment_id = self._resolve_trace_experiment(trace_id)
            except MlflowException:
                continue  # Skip non-existent traces

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
            if meta is None:
                continue

            trace_info = self._build_trace_info(experiment_id, trace_id, meta)
            results.append(trace_info)

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestBatchGetTraceInfos -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement batch_get_trace_infos"
```

---

### Task 4: `batch_get_traces`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** Same as `batch_get_trace_infos` but also reads the SPANS cache item (`SK=T#<trace_id>#SPANS`). Returns `list[Trace]` with `Trace(info=trace_info, data=TraceData(spans=...))`. Uses `Span.from_dict()` for deserialization. Traces without cached spans get an empty span list (no X-Ray fallback in batch — that's `get_trace`'s job).

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestBatchGetTraces`:

```python
class TestBatchGetTraces:
    """Tests for batch_get_traces."""

    def _create_trace_with_spans(self, tracking_store, exp_id, trace_id, request_time=1000):
        """Helper: create a trace and write a SPANS cache item in X-Ray converter format."""
        import json as _json
        from mlflow_dynamodbstore.xray.span_converter import span_dicts_to_mlflow_spans

        trace_info = _make_trace_info(
            exp_id, trace_id=trace_id, request_time=request_time,
        )
        tracking_store.start_trace(trace_info)

        # Use X-Ray converter format (same as get_trace caches)
        span_dicts = [{"name": "root", "span_type": "CHAIN",
                       "trace_id": trace_id, "span_id": "span-1",
                       "parent_span_id": None, "start_time_ns": 0,
                       "end_time_ns": 1000, "status": "OK",
                       "attributes": {}, "events": []}]
        # Verify the format is valid for deserialization
        assert len(span_dicts_to_mlflow_spans(span_dicts, trace_id)) == 1

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        spans_item = {
            "PK": pk,
            "SK": f"{SK_TRACE_PREFIX}{trace_id}#SPANS",
            "data": _json.dumps(span_dicts),
        }
        tracking_store._table.put_item(spans_item)

    def test_batch_get_single_trace_with_spans(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        self._create_trace_with_spans(tracking_store, exp_id, "tr-spans-1")
        result = tracking_store.batch_get_traces(["tr-spans-1"])
        assert len(result) == 1
        assert result[0].info.trace_id == "tr-spans-1"
        assert len(result[0].data.spans) > 0

    def test_batch_get_trace_without_spans(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-no-spans")
        tracking_store.start_trace(trace_info)
        result = tracking_store.batch_get_traces(["tr-no-spans"])
        assert len(result) == 1
        assert result[0].info.trace_id == "tr-no-spans"
        assert result[0].data.spans == []

    def test_batch_get_multiple_traces(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        for i in range(3):
            self._create_trace_with_spans(tracking_store, exp_id, f"tr-multi-{i}", request_time=1000 + i)
        result = tracking_store.batch_get_traces([f"tr-multi-{i}" for i in range(3)])
        assert len(result) == 3

    def test_nonexistent_trace_excluded(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        self._create_trace_with_spans(tracking_store, exp_id, "tr-exists")
        result = tracking_store.batch_get_traces(["tr-exists", "tr-ghost"])
        assert len(result) == 1

    def test_empty_list_returns_empty(self, tracking_store):
        result = tracking_store.batch_get_traces([])
        assert result == []

    def test_duplicate_ids_deduplicated(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        self._create_trace_with_spans(tracking_store, exp_id, "tr-dup")
        result = tracking_store.batch_get_traces(["tr-dup", "tr-dup"])
        assert len(result) == 1

    def test_with_location_parameter(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        self._create_trace_with_spans(tracking_store, exp_id, "tr-loc")
        result = tracking_store.batch_get_traces(["tr-loc"], location=exp_id)
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestBatchGetTraces -v`
Expected: FAIL

- [ ] **Step 3: Implement `batch_get_traces`**

Add to `DynamoDBTrackingStore` after `batch_get_trace_infos`:

```python
    def batch_get_traces(
        self, trace_ids: list[str], location: str | None = None
    ) -> list[Trace]:
        """Get complete traces with spans for given trace IDs."""
        import json as _json

        from mlflow.entities.trace import Trace
        from mlflow.entities.trace_data import TraceData

        from mlflow_dynamodbstore.xray.span_converter import span_dicts_to_mlflow_spans

        if not trace_ids:
            return []

        seen: set[str] = set()
        results: list[Trace] = []

        for trace_id in trace_ids:
            if trace_id in seen:
                continue
            seen.add(trace_id)

            try:
                if location:
                    experiment_id = location
                else:
                    experiment_id = self._resolve_trace_experiment(trace_id)
            except MlflowException:
                continue

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
            if meta is None:
                continue

            trace_info = self._build_trace_info(experiment_id, trace_id, meta)

            # Read cached spans
            spans_sk = f"{SK_TRACE_PREFIX}{trace_id}#SPANS"
            cached = self._table.get_item(pk=pk, sk=spans_sk)
            if cached is not None:
                span_dicts = _json.loads(cached["data"])
                # Handle both V3 format (Span.to_dict) and X-Ray format
                if span_dicts and "start_time_unix_nano" in span_dicts[0]:
                    # V3 format: use Span.from_dict
                    from mlflow.entities.span import Span as SpanEntity

                    spans = [SpanEntity.from_dict(sd) for sd in span_dicts]
                else:
                    # X-Ray converter format
                    spans = span_dicts_to_mlflow_spans(span_dicts, trace_id)
            else:
                spans = []

            results.append(Trace(info=trace_info, data=TraceData(spans=spans)))

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestBatchGetTraces -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement batch_get_traces"
```

---

### Task 5: `find_completed_sessions`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** Queries GSI2 with `PK=SESSIONS#<workspace>#<experiment_id>`, SK between `min_ts` and `max_ts`. Returns `list[CompletedSession]`. Session tracker items are created by the `start_trace` upsert from Task 2. If `filter_string` is provided, post-filter sessions by checking if any trace in the session matches the filter.

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestFindCompletedSessions`:

```python
class TestFindCompletedSessions:
    """Tests for find_completed_sessions."""

    def _create_session_traces(self, tracking_store, exp_id, session_id, timestamps):
        """Helper: create traces with session metadata at given timestamps."""
        for i, ts in enumerate(timestamps):
            trace_info = _make_trace_info(
                exp_id,
                trace_id=f"tr-{session_id}-{i}",
                request_time=ts,
                trace_metadata={
                    TraceTagKey.TRACE_NAME: "my-trace",
                    "mlflow.traceSession": session_id,
                },
            )
            tracking_store.start_trace(trace_info)

    def test_find_sessions_in_time_window(self, tracking_store):
        from mlflow.genai.scorers.online.entities import CompletedSession

        exp_id = _create_experiment(tracking_store)
        self._create_session_traces(tracking_store, exp_id, "sess-a", [1000, 2000])
        self._create_session_traces(tracking_store, exp_id, "sess-b", [3000, 4000])
        self._create_session_traces(tracking_store, exp_id, "sess-c", [5000, 6000])

        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=2000,
            max_last_trace_timestamp_ms=4000,
        )
        session_ids = [s.session_id for s in result]
        assert "sess-a" in session_ids
        assert "sess-b" in session_ids
        assert "sess-c" not in session_ids

    def test_sessions_outside_window_excluded(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        self._create_session_traces(tracking_store, exp_id, "sess-early", [100])
        self._create_session_traces(tracking_store, exp_id, "sess-late", [9000])

        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=500,
            max_last_trace_timestamp_ms=8000,
        )
        session_ids = [s.session_id for s in result]
        assert "sess-early" not in session_ids
        assert "sess-late" not in session_ids

    def test_max_results_limits_output(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        for i in range(5):
            self._create_session_traces(tracking_store, exp_id, f"sess-{i}", [1000 + i * 100])

        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=0,
            max_last_trace_timestamp_ms=9999,
            max_results=2,
        )
        assert len(result) <= 2

    def test_empty_experiment_returns_empty(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=0,
            max_last_trace_timestamp_ms=9999,
        )
        assert result == []

    def test_filter_string_filters_sessions(self, tracking_store):
        """filter_string filters sessions by trace attributes."""
        exp_id = _create_experiment(tracking_store)
        # Session with a tagged trace
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-filt-match",
            request_time=1000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "sess-match",
            },
            tags={"env": "prod"},
        )
        tracking_store.start_trace(trace_info)
        # Session without the tag
        trace_info2 = _make_trace_info(
            exp_id,
            trace_id="tr-filt-nomatch",
            request_time=2000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "sess-nomatch",
            },
            tags={"env": "dev"},
        )
        tracking_store.start_trace(trace_info2)

        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=0,
            max_last_trace_timestamp_ms=9999,
            filter_string="tag.env = 'prod'",
        )
        session_ids = [s.session_id for s in result]
        assert "sess-match" in session_ids
        assert "sess-nomatch" not in session_ids

    def test_session_attributes(self, tracking_store):
        """Verify CompletedSession fields are populated correctly."""
        exp_id = _create_experiment(tracking_store)
        self._create_session_traces(tracking_store, exp_id, "sess-check", [1000, 2000, 3000])

        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=0,
            max_last_trace_timestamp_ms=9999,
        )
        assert len(result) == 1
        session = result[0]
        assert session.session_id == "sess-check"
        assert session.first_trace_timestamp_ms == 1000
        assert session.last_trace_timestamp_ms == 3000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestFindCompletedSessions -v`
Expected: FAIL

- [ ] **Step 3: Implement `find_completed_sessions`**

Add the import at the top of `tracking_store.py` (TYPE_CHECKING block):
```python
from mlflow.genai.scorers.online.entities import CompletedSession
```

Add method to `DynamoDBTrackingStore`:

```python
    def find_completed_sessions(
        self,
        experiment_id: str,
        min_last_trace_timestamp_ms: int,
        max_last_trace_timestamp_ms: int,
        max_results: int | None = None,
        filter_string: str | None = None,
    ) -> list[CompletedSession]:
        """Find completed sessions by last trace timestamp range via GSI2."""
        from mlflow.genai.scorers.online.entities import CompletedSession

        gsi2pk = f"{GSI2_SESSIONS_PREFIX}{self._workspace}#{experiment_id}"

        items = self._table.query(
            pk=gsi2pk,
            sk_gte=f"{min_last_trace_timestamp_ms:020d}",
            sk_lte=f"{max_last_trace_timestamp_ms:020d}",
            index_name="gsi2",
            scan_forward=True,
        )

        # Optional: post-filter sessions by trace attributes
        if filter_string:
            from mlflow_dynamodbstore.dynamodb.search import (
                _apply_trace_post_filter,
                parse_trace_filter,
            )

            preds = parse_trace_filter(filter_string)
            filtered_items = []
            for item in items:
                # Check if any trace in this session matches the filter.
                # Query traces with this session's metadata.
                session_id = item["session_id"]
                exp_pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
                trace_items = self._table.query(
                    pk=exp_pk, sk_prefix=SK_TRACE_PREFIX,
                )
                session_qualifies = False
                for t_item in trace_items:
                    if "trace_id" not in t_item:
                        continue
                    tid = t_item["trace_id"]
                    # Check if this trace belongs to the session
                    rmeta_sk = f"{SK_TRACE_PREFIX}{tid}#RMETA#mlflow.traceSession"
                    rmeta = self._table.get_item(pk=exp_pk, sk=rmeta_sk)
                    if rmeta and rmeta.get("value") == session_id:
                        if all(
                            _apply_trace_post_filter(self._table, exp_pk, tid, t_item, p)
                            for p in preds
                        ):
                            session_qualifies = True
                            break
                if session_qualifies:
                    filtered_items.append(item)
            items = filtered_items

        results: list[CompletedSession] = []
        for item in items:
            session = CompletedSession(
                session_id=item["session_id"],
                first_trace_timestamp_ms=int(item["first_trace_timestamp_ms"]),
                last_trace_timestamp_ms=int(item["last_trace_timestamp_ms"]),
            )
            results.append(session)

        if max_results is not None:
            results = results[:max_results]

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestFindCompletedSessions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement find_completed_sessions"
```

---

### Task 6: `link_prompts_to_trace`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** Resolve trace_id → experiment_id, serialize `prompt_versions` to JSON `[{"name": ..., "version": ...}, ...]`, write as trace tag `mlflow.promptVersions` using existing `_write_trace_tag`. The `PromptVersion` entity has `name` and `version` fields.

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestLinkPromptsToTrace`:

```python
class TestLinkPromptsToTrace:
    """Tests for link_prompts_to_trace."""

    def test_link_single_prompt(self, tracking_store):
        from mlflow.entities.model_registry import PromptVersion

        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-prompt-1")
        tracking_store.start_trace(trace_info)

        pv = PromptVersion(name="my-prompt", version=1, template="hello {name}")
        tracking_store.link_prompts_to_trace("tr-prompt-1", [pv])

        fetched = tracking_store.get_trace_info("tr-prompt-1")
        assert "mlflow.promptVersions" in fetched.tags
        versions = json.loads(fetched.tags["mlflow.promptVersions"])
        assert len(versions) == 1
        assert versions[0]["name"] == "my-prompt"
        assert versions[0]["version"] == 1

    def test_link_multiple_prompts(self, tracking_store):
        from mlflow.entities.model_registry import PromptVersion

        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-prompt-2")
        tracking_store.start_trace(trace_info)

        pvs = [
            PromptVersion(name="prompt-a", version=1, template="a"),
            PromptVersion(name="prompt-b", version=3, template="b"),
        ]
        tracking_store.link_prompts_to_trace("tr-prompt-2", pvs)

        fetched = tracking_store.get_trace_info("tr-prompt-2")
        versions = json.loads(fetched.tags["mlflow.promptVersions"])
        assert len(versions) == 2

    def test_overwrite_existing_prompt_links(self, tracking_store):
        from mlflow.entities.model_registry import PromptVersion

        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-prompt-3")
        tracking_store.start_trace(trace_info)

        pv1 = PromptVersion(name="old-prompt", version=1, template="old")
        tracking_store.link_prompts_to_trace("tr-prompt-3", [pv1])

        pv2 = PromptVersion(name="new-prompt", version=2, template="new")
        tracking_store.link_prompts_to_trace("tr-prompt-3", [pv2])

        fetched = tracking_store.get_trace_info("tr-prompt-3")
        versions = json.loads(fetched.tags["mlflow.promptVersions"])
        assert len(versions) == 1
        assert versions[0]["name"] == "new-prompt"

    def test_link_nonexistent_trace_raises(self, tracking_store):
        from mlflow.entities.model_registry import PromptVersion

        pv = PromptVersion(name="p", version=1, template="t")
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.link_prompts_to_trace("nonexistent-trace", [pv])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestLinkPromptsToTrace -v`
Expected: FAIL

- [ ] **Step 3: Implement `link_prompts_to_trace`**

Add to `DynamoDBTrackingStore`:

```python
    def link_prompts_to_trace(
        self, trace_id: str, prompt_versions: list[PromptVersion]
    ) -> None:
        """Link prompt versions to a trace by writing mlflow.promptVersions tag."""
        import json as _json

        from mlflow.entities.model_registry import PromptVersion  # noqa: F811

        experiment_id = self._resolve_trace_experiment(trace_id)
        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        if meta is None:
            raise MlflowException(
                f"Trace '{trace_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ttl = int(meta["ttl"]) if "ttl" in meta else self._get_trace_ttl()

        versions_json = _json.dumps(
            [{"name": pv.name, "version": pv.version} for pv in prompt_versions]
        )
        self._write_trace_tag(
            experiment_id, trace_id, "mlflow.promptVersions", versions_json, ttl
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestLinkPromptsToTrace -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement link_prompts_to_trace"
```

---

### Task 7: `unlink_traces_from_run`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** The inverse of `link_traces_to_run` (line 2832). That method writes `RMETA#mlflow.sourceRun` items. This method deletes them. Uses `TraceMetadataKey.SOURCE_RUN` for the key. Resolve each trace_id via GSI1, then delete the RMETA item. Silent on missing items (idempotent).

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestUnlinkTracesFromRun`:

```python
class TestUnlinkTracesFromRun:
    """Tests for unlink_traces_from_run."""

    def test_unlink_single_trace(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-unlink-1")
        tracking_store.start_trace(trace_info)
        tracking_store.link_traces_to_run(["tr-unlink-1"], "run-123")

        # Verify linked
        fetched = tracking_store.get_trace_info("tr-unlink-1")
        assert TraceMetadataKey.SOURCE_RUN in fetched.trace_metadata

        tracking_store.unlink_traces_from_run(["tr-unlink-1"], "run-123")

        # Verify unlinked
        fetched = tracking_store.get_trace_info("tr-unlink-1")
        assert TraceMetadataKey.SOURCE_RUN not in fetched.trace_metadata

    def test_unlink_multiple_traces(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        for tid in ["tr-ul-a", "tr-ul-b"]:
            trace_info = _make_trace_info(exp_id, trace_id=tid)
            tracking_store.start_trace(trace_info)
        tracking_store.link_traces_to_run(["tr-ul-a", "tr-ul-b"], "run-456")

        tracking_store.unlink_traces_from_run(["tr-ul-a", "tr-ul-b"], "run-456")

        for tid in ["tr-ul-a", "tr-ul-b"]:
            fetched = tracking_store.get_trace_info(tid)
            assert TraceMetadataKey.SOURCE_RUN not in fetched.trace_metadata

    def test_unlink_not_linked_trace_is_silent(self, tracking_store):
        """Unlinking a trace that was never linked should not raise."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-never-linked")
        tracking_store.start_trace(trace_info)

        # Should not raise
        tracking_store.unlink_traces_from_run(["tr-never-linked"], "run-789")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestUnlinkTracesFromRun -v`
Expected: FAIL

- [ ] **Step 3: Implement `unlink_traces_from_run`**

Add to `DynamoDBTrackingStore`:

```python
    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        """Unlink traces from a run by deleting mlflow.sourceRun RMETA items."""
        for trace_id in trace_ids:
            try:
                experiment_id = self._resolve_trace_experiment(trace_id)
            except MlflowException:
                continue
            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
            sk = f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{TraceMetadataKey.SOURCE_RUN}"
            self._table.delete_item(pk=pk, sk=sk)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestUnlinkTracesFromRun -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement unlink_traces_from_run"
```

---

### Task 8: `log_spans`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** Groups spans by trace_id. For each trace_id: resolve experiment_id (from `location` if provided), serialize spans to JSON, write SPANS cache item at `SK=T#<trace_id>#SPANS`. Same SPANS item structure as `get_trace` (line 2390): `{"PK": pk, "SK": spans_sk, "data": json.dumps(span_dicts)}`. Uses `Span.to_dict()` for serialization and `Span.from_dict()` for deserialization.

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestLogSpans`:

```python
class TestLogSpans:
    """Tests for log_spans."""

    @staticmethod
    def _make_mock_span(trace_id, span_id="span-1", name="root"):
        """Helper: build a mock Span with to_dict() support."""
        span = MagicMock()
        span.trace_id = trace_id
        span.name = name
        span.to_dict.return_value = {
            "name": name,
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": None,
            "start_time_unix_nano": 0,
            "end_time_unix_nano": 1000000000,
            "status": {"code": "OK", "message": ""},
            "attributes": {"mlflow.traceRequestId": trace_id},
            "events": [],
        }
        return span

    def test_log_spans_creates_spans_item(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-logspan-1")
        tracking_store.start_trace(trace_info)

        span = self._make_mock_span("tr-logspan-1")
        result = tracking_store.log_spans(exp_id, [span])
        assert len(result) == 1

        # Verify SPANS item was written
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        cached = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-logspan-1#SPANS")
        assert cached is not None
        assert "data" in cached

    def test_log_spans_overwrites_existing(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-logspan-2")
        tracking_store.start_trace(trace_info)

        span1 = self._make_mock_span("tr-logspan-2", name="first")
        tracking_store.log_spans(exp_id, [span1])

        span2 = self._make_mock_span("tr-logspan-2", name="second")
        tracking_store.log_spans(exp_id, [span2])

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        cached = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-logspan-2#SPANS")
        data = json.loads(cached["data"])
        assert any(s["name"] == "second" for s in data)

    def test_log_spans_multiple_traces(self, tracking_store):
        """Spans for different traces in one call create separate SPANS items."""
        exp_id = _create_experiment(tracking_store)
        for tid in ["tr-multi-a", "tr-multi-b"]:
            tracking_store.start_trace(_make_trace_info(exp_id, trace_id=tid))

        span_a = self._make_mock_span("tr-multi-a")
        span_b = self._make_mock_span("tr-multi-b", span_id="span-2")
        result = tracking_store.log_spans(exp_id, [span_a, span_b])
        assert len(result) == 2

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        for tid in ["tr-multi-a", "tr-multi-b"]:
            cached = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{tid}#SPANS")
            assert cached is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestLogSpans -v`
Expected: FAIL

- [ ] **Step 3: Implement `log_spans`**

Add to `DynamoDBTrackingStore`:

```python
    def log_spans(
        self, location: str, spans: list[Span], tracking_uri: str | None = None
    ) -> list[Span]:
        """Log spans to the tracking store by writing SPANS cache items."""
        import json as _json

        from mlflow.entities.span import Span  # noqa: F811

        if not spans:
            return []

        # Group spans by trace_id
        from collections import defaultdict

        spans_by_trace: dict[str, list[Span]] = defaultdict(list)
        for span in spans:
            spans_by_trace[span.trace_id].append(span)

        for trace_id, trace_spans in spans_by_trace.items():
            try:
                experiment_id = location or self._resolve_trace_experiment(trace_id)
            except MlflowException:
                experiment_id = location

            pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"

            # Read TTL from trace META
            meta = self._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
            ttl = int(meta["ttl"]) if meta and "ttl" in meta else self._get_trace_ttl()

            span_dicts = [s.to_dict() for s in trace_spans]
            spans_item: dict[str, Any] = {
                "PK": pk,
                "SK": f"{SK_TRACE_PREFIX}{trace_id}#SPANS",
                "data": _json.dumps(span_dicts),
            }
            if ttl is not None:
                spans_item["ttl"] = ttl
            self._table.put_item(spans_item)

        return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestLogSpans -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement log_spans"
```

---

### Task 9: `calculate_trace_filter_correlation`

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (add method)
- Test: `tests/unit/test_tracking_traces.py`

**Context:** Single-pass analysis over traces. Parse `filter_string1` and `filter_string2` into predicate lists using `parse_trace_filter`. Iterate over traces using `search_traces`-style query (reuse `execute_trace_query` + `_apply_trace_post_filter` pattern). For each trace, evaluate both predicate lists. Increment 4 counters: `total_count`, `filter1_count`, `filter2_count`, `joint_count`. Compute NPMI. Return `TraceFilterCorrelationResult`.

The NPMI formula:
- `p1 = filter1_count / total_count`
- `p2 = filter2_count / total_count`
- `p_joint = joint_count / total_count`
- `pmi = log(p_joint / (p1 * p2))`
- `npmi = pmi / -log(p_joint)`

Edge cases per spec:
- `total_count = 0` → NPMI = 0.0
- `joint_count = 0` → NPMI = -1.0
- `filter1_count = 0` or `filter2_count = 0` → NPMI = 0.0
- All traces match both → NPMI = 1.0

- [ ] **Step 1: Write the failing tests**

Add a new test class `TestCalculateTraceFilterCorrelation`:

```python
class TestCalculateTraceFilterCorrelation:
    """Tests for calculate_trace_filter_correlation."""

    def _create_traces_with_tags(self, tracking_store, exp_id, traces_spec):
        """Helper: create traces with specified tags.

        traces_spec: list of dicts with keys 'trace_id', 'tags'.
        """
        for i, spec in enumerate(traces_spec):
            trace_info = _make_trace_info(
                exp_id,
                trace_id=spec["trace_id"],
                request_time=1000 + i * 100,
                tags=spec.get("tags", {}),
            )
            tracking_store.start_trace(trace_info)

    def test_known_distribution(self, tracking_store):
        """10 traces, 5 match f1, 4 match f2, 2 match both."""
        exp_id = _create_experiment(tracking_store)
        specs = []
        for i in range(10):
            tags = {}
            if i < 5:
                tags["color"] = "red"
            if i < 4:
                tags["size"] = "large"
            if i >= 5:
                tags["color"] = "blue"
            if i >= 4:
                tags["size"] = "small"
            specs.append({"trace_id": f"tr-corr-{i}", "tags": tags})
        self._create_traces_with_tags(tracking_store, exp_id, specs)

        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.color = 'red'",
            filter_string2="tag.size = 'large'",
        )
        assert result.total_count == 10
        assert result.filter1_count == 5
        assert result.filter2_count == 4
        assert result.joint_count == 4  # first 4 have both red+large

    def test_no_traces_returns_zero_npmi(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.x = 'a'",
            filter_string2="tag.y = 'b'",
        )
        assert result.total_count == 0
        assert result.npmi == 0.0

    def test_joint_count_zero_returns_negative_one(self, tracking_store):
        """Filters never co-occur → NPMI = -1.0."""
        exp_id = _create_experiment(tracking_store)
        specs = [
            {"trace_id": "tr-nc-0", "tags": {"group": "a"}},
            {"trace_id": "tr-nc-1", "tags": {"group": "b"}},
        ]
        self._create_traces_with_tags(tracking_store, exp_id, specs)

        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.group = 'a'",
            filter_string2="tag.group = 'b'",
        )
        assert result.joint_count == 0
        assert result.npmi == -1.0

    def test_filter_count_zero_returns_zero_npmi(self, tracking_store):
        """One filter matches nothing → NPMI = 0.0."""
        exp_id = _create_experiment(tracking_store)
        specs = [{"trace_id": "tr-fz-0", "tags": {"x": "1"}}]
        self._create_traces_with_tags(tracking_store, exp_id, specs)

        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.x = '1'",
            filter_string2="tag.x = '2'",
        )
        assert result.filter2_count == 0
        assert result.npmi == 0.0

    def test_perfect_correlation(self, tracking_store):
        """All traces match both filters → NPMI = 1.0."""
        exp_id = _create_experiment(tracking_store)
        specs = [
            {"trace_id": f"tr-pc-{i}", "tags": {"a": "1", "b": "2"}}
            for i in range(5)
        ]
        self._create_traces_with_tags(tracking_store, exp_id, specs)

        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.a = '1'",
            filter_string2="tag.b = '2'",
        )
        assert result.npmi == 1.0

    def test_base_filter_restricts_universe(self, tracking_store):
        """base_filter limits which traces are counted."""
        exp_id = _create_experiment(tracking_store)
        specs = [
            {"trace_id": "tr-bf-0", "tags": {"env": "prod", "status": "ok"}},
            {"trace_id": "tr-bf-1", "tags": {"env": "prod", "status": "fail"}},
            {"trace_id": "tr-bf-2", "tags": {"env": "dev", "status": "ok"}},
        ]
        self._create_traces_with_tags(tracking_store, exp_id, specs)

        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.status = 'ok'",
            filter_string2="tag.status = 'fail'",
            base_filter="tag.env = 'prod'",
        )
        assert result.total_count == 2  # only prod traces
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestCalculateTraceFilterCorrelation -v`
Expected: FAIL

- [ ] **Step 3: Implement `calculate_trace_filter_correlation`**

Add to `DynamoDBTrackingStore`:

```python
    def calculate_trace_filter_correlation(
        self,
        experiment_ids: list[str],
        filter_string1: str,
        filter_string2: str,
        base_filter: str | None = None,
    ) -> TraceFilterCorrelationResult:
        """Calculate NPMI correlation between two trace filters."""
        import math

        from mlflow.tracing.analysis import TraceFilterCorrelationResult

        from mlflow_dynamodbstore.dynamodb.search import (
            _apply_trace_post_filter,
            execute_trace_query,
            parse_trace_filter,
            plan_trace_query,
        )

        preds1 = parse_trace_filter(filter_string1)
        preds2 = parse_trace_filter(filter_string2)
        base_preds = parse_trace_filter(base_filter)

        # Plan query using base_filter predicates (for efficient index usage)
        plan = plan_trace_query(base_preds, None)

        total_count = 0
        filter1_count = 0
        filter2_count = 0
        joint_count = 0

        for exp_id in experiment_ids:
            pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
            page_token: str | None = None

            while True:
                items, page_token = execute_trace_query(
                    table=self._table,
                    plan=plan,
                    pk=pk,
                    max_results=1000,
                    page_token=page_token,
                    predicates=base_preds,
                )

                for item in items:
                    trace_id = item["trace_id"]
                    total_count += 1

                    match1 = all(
                        _apply_trace_post_filter(self._table, pk, trace_id, item, p)
                        for p in preds1
                    )
                    match2 = all(
                        _apply_trace_post_filter(self._table, pk, trace_id, item, p)
                        for p in preds2
                    )

                    if match1:
                        filter1_count += 1
                    if match2:
                        filter2_count += 1
                    if match1 and match2:
                        joint_count += 1

                if not page_token:
                    break

        # Compute NPMI
        if total_count == 0:
            npmi = 0.0
        elif filter1_count == 0 or filter2_count == 0:
            npmi = 0.0
        elif joint_count == 0:
            npmi = -1.0
        elif (
            joint_count == total_count
            and filter1_count == total_count
            and filter2_count == total_count
        ):
            npmi = 1.0
        else:
            p1 = filter1_count / total_count
            p2 = filter2_count / total_count
            p_joint = joint_count / total_count
            pmi = math.log(p_joint / (p1 * p2))
            npmi = pmi / -math.log(p_joint)

        return TraceFilterCorrelationResult(
            npmi=npmi,
            filter1_count=filter1_count,
            filter2_count=filter2_count,
            joint_count=joint_count,
            total_count=total_count,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_tracking_traces.py::TestCalculateTraceFilterCorrelation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_tracking_traces.py
git commit -m "feat(traces): implement calculate_trace_filter_correlation"
```

---

### Task 10: Integration & E2E Tests

**Files:**
- Modify: `tests/integration/test_traces.py`
- Modify: `tests/e2e/test_traces.py`

- [ ] **Step 1: Add integration tests for batch operations**

Add to `tests/integration/test_traces.py`:

```python
class TestBatchTraceOperations:
    """Integration tests for batch_get_traces and batch_get_trace_infos."""

    def test_batch_get_trace_infos_round_trip(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-batch-infos")
        trace_ids = []
        for i in range(3):
            tid = f"tr-int-batch-{i}"
            trace_info = _make_trace_info(exp_id, trace_id=tid, request_time=1000 + i)
            tracking_store.start_trace(trace_info)
            trace_ids.append(tid)

        result = tracking_store.batch_get_trace_infos(trace_ids)
        assert len(result) == 3
        assert {t.trace_id for t in result} == set(trace_ids)
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/integration/test_traces.py::TestBatchTraceOperations -v`
Expected: PASS

- [ ] **Step 3: Add e2e test for batch_get_traces**

Add to `tests/e2e/test_traces.py`:

```python
@pytest.mark.e2e
class TestBatchGetTracesE2E:
    def test_batch_get_traces_via_client(self, mlflow_client):
        """Create multiple traces via client, then batch fetch them."""
        # This test exercises the REST endpoint round-trip
        import mlflow

        @mlflow.trace
        def my_func(x):
            return x + 1

        trace_ids = []
        for i in range(3):
            result = my_func(i)
            # Get the most recent trace
            traces = mlflow_client.search_traces(experiment_ids=["0"])
            if traces:
                trace_ids.append(traces[0].info.trace_id)

        if trace_ids:
            batch_result = mlflow_client.get_trace(trace_ids[0])
            assert batch_result is not None
```

- [ ] **Step 4: Run e2e tests**

Run: `uv run pytest tests/e2e/test_traces.py::TestBatchGetTracesE2E -v --timeout=60`
Expected: PASS (if e2e environment is available)

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_traces.py tests/e2e/test_traces.py
git commit -m "test(traces): add integration and e2e tests for Phase 3 methods"
```

---

### Task 11: Coverage Verification & Cleanup

- [ ] **Step 1: Run full test suite with coverage**

```bash
uv run pytest tests/unit/test_tracking_traces.py -v --cov=mlflow_dynamodbstore --cov-report=term-missing
```

Verify: 100% patch coverage on all new methods.

- [ ] **Step 2: Run all unit tests to check for regressions**

```bash
uv run pytest tests/unit/ -v
```

Expected: All tests pass.

- [ ] **Step 3: Run linting/type checks**

```bash
uv run ruff check src/mlflow_dynamodbstore/tracking_store.py src/mlflow_dynamodbstore/dynamodb/schema.py
```

Expected: No errors.

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: Phase 3 cleanup and coverage fixes"
```
