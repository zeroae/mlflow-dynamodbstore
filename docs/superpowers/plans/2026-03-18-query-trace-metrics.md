# Phase 4b: `query_trace_metrics` Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `query_trace_metrics` — the last `NotImplementedError` in the DynamoDB tracking store — enabling the experiment overview dashboard's trace analytics charts.

**Architecture:** Streaming aggregation with batched fetches. Trace META items are queried in pages (time-filtered via LSI1), filters applied per batch, view-specific items fetched for survivors, and values fed into O(1) accumulators. Results cached in DynamoDB (15-min TTL) for pagination. Three new item types (span items, trace metrics, span metrics) written at `log_spans` time. Assessment items gain denormalized top-level attributes.

**Tech Stack:** Python 3.11, boto3 (DynamoDB), moto (testing), MLflow SDK entities/constants

**Spec:** `docs/superpowers/specs/2026-03-17-query-trace-metrics-design.md`

---

## File Structure

```
src/mlflow_dynamodbstore/
├── dynamodb/
│   └── schema.py                          # MODIFY: add 5 new constants
├── tracking_store.py                      # MODIFY: log_spans, create/update_assessment, query_trace_metrics
└── trace_metrics/                         # CREATE: new package
    ├── __init__.py                         # CREATE: package init with public API
    ├── accumulators.py                     # CREATE: MetricAccumulator + percentile
    ├── extractors.py                       # CREATE: view-specific data extraction
    ├── filters.py                          # CREATE: filter parsing + application
    └── pagination.py                       # CREATE: DynamoDB cache + page tokens

tests/
├── unit/
│   ├── test_trace_metrics_accumulators.py  # CREATE: accumulator unit tests
│   ├── test_trace_metrics_write_path.py    # CREATE: log_spans + assessment denorm tests
│   └── test_trace_metrics_query.py         # CREATE: full query_trace_metrics tests
├── integration/
│   └── test_trace_metrics.py              # CREATE: REST round-trip tests
└── e2e/
    └── test_trace_metrics.py              # CREATE: SDK-level analytics tests
```

---

### Task 1: Schema Constants + Package Skeleton

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/schema.py`
- Create: `src/mlflow_dynamodbstore/trace_metrics/__init__.py`
- Create: `src/mlflow_dynamodbstore/trace_metrics/accumulators.py`
- Create: `src/mlflow_dynamodbstore/trace_metrics/extractors.py`
- Create: `src/mlflow_dynamodbstore/trace_metrics/filters.py`
- Create: `src/mlflow_dynamodbstore/trace_metrics/pagination.py`

- [ ] **Step 1: Add schema constants**

Add to `src/mlflow_dynamodbstore/dynamodb/schema.py` after existing constants:

```python
# Individual span items (within trace partition)
SK_SPAN_PREFIX = "#SPAN#"

# Trace-level metric items (token usage)
SK_TRACE_METRIC_PREFIX = "#TMETRIC#"

# Span-level metric items (costs)
SK_SPAN_METRIC_PREFIX = "#SMETRIC#"

# Query trace metrics cache
PK_TMCACHE_PREFIX = "TMCACHE#"
SK_TMCACHE_RESULT = "RESULT"
```

- [ ] **Step 2: Create package skeleton**

Create `src/mlflow_dynamodbstore/trace_metrics/__init__.py`:

```python
"""Trace metrics aggregation engine for query_trace_metrics."""
```

Create empty files with docstrings for `accumulators.py`, `extractors.py`, `filters.py`, `pagination.py`:

```python
# accumulators.py
"""Streaming metric accumulators with percentile support."""

# extractors.py
"""View-specific data extraction from DynamoDB items."""

# filters.py
"""Filter parsing and application for trace metrics queries."""

# pagination.py
"""DynamoDB cache and page token management for query_trace_metrics."""
```

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "from mlflow_dynamodbstore.trace_metrics import accumulators, extractors, filters, pagination; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/schema.py src/mlflow_dynamodbstore/trace_metrics/
git commit -m "feat(trace-metrics): add schema constants and package skeleton"
```

---

### Task 2: MetricAccumulator

Pure Python — no DynamoDB. Tested in isolation.

**Files:**
- Create: `src/mlflow_dynamodbstore/trace_metrics/accumulators.py`
- Create: `tests/unit/test_trace_metrics_accumulators.py`

- [ ] **Step 1: Write accumulator tests**

Create `tests/unit/test_trace_metrics_accumulators.py`:

```python
"""Unit tests for MetricAccumulator."""

import math

import pytest
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation

from mlflow_dynamodbstore.trace_metrics.accumulators import MetricAccumulator


class TestMetricAccumulator:
    def test_count(self):
        acc = MetricAccumulator()
        for v in [1.0, 2.0, 3.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.COUNT)]
        result = acc.finalize(aggs)
        assert result["COUNT"] == 3

    def test_sum(self):
        acc = MetricAccumulator()
        for v in [1.0, 2.0, 3.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.SUM)]
        result = acc.finalize(aggs)
        assert result["SUM"] == 6.0

    def test_avg(self):
        acc = MetricAccumulator()
        for v in [1.0, 2.0, 3.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.AVG)]
        result = acc.finalize(aggs)
        assert result["AVG"] == 2.0

    def test_avg_empty(self):
        acc = MetricAccumulator()
        aggs = [MetricAggregation(AggregationType.AVG)]
        result = acc.finalize(aggs)
        assert result["AVG"] == 0.0

    def test_min_max(self):
        acc = MetricAccumulator()
        for v in [3.0, 1.0, 2.0]:
            acc.add(v)
        aggs = [
            MetricAggregation(AggregationType.MIN),
            MetricAggregation(AggregationType.MAX),
        ]
        result = acc.finalize(aggs)
        assert result["MIN"] == 1.0
        assert result["MAX"] == 3.0

    def test_percentile_median(self):
        acc = MetricAccumulator(collect_values=True)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0)]
        result = acc.finalize(aggs)
        assert result["P50.0"] == 3.0

    def test_percentile_p90(self):
        acc = MetricAccumulator(collect_values=True)
        for v in range(1, 101):
            acc.add(float(v))
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=90.0)]
        result = acc.finalize(aggs)
        assert result["P90.0"] == pytest.approx(90.1, rel=1e-3)

    def test_percentile_p0_and_p100(self):
        acc = MetricAccumulator(collect_values=True)
        for v in [10.0, 20.0, 30.0]:
            acc.add(v)
        aggs = [
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=0.0),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=100.0),
        ]
        result = acc.finalize(aggs)
        assert result["P0.0"] == 10.0
        assert result["P100.0"] == 30.0

    def test_percentile_single_value(self):
        acc = MetricAccumulator(collect_values=True)
        acc.add(42.0)
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0)]
        result = acc.finalize(aggs)
        assert result["P50.0"] == 42.0

    def test_percentile_without_collect_values(self):
        """Percentile on accumulator without collect_values returns NaN."""
        acc = MetricAccumulator(collect_values=False)
        acc.add(1.0)
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0)]
        result = acc.finalize(aggs)
        assert math.isnan(result["P50.0"])

    def test_multiple_aggregations(self):
        acc = MetricAccumulator(collect_values=True)
        for v in [10.0, 20.0, 30.0]:
            acc.add(v)
        aggs = [
            MetricAggregation(AggregationType.COUNT),
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.AVG),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0),
        ]
        result = acc.finalize(aggs)
        assert result["COUNT"] == 3
        assert result["SUM"] == 60.0
        assert result["AVG"] == 20.0
        assert result["P50.0"] == 20.0

    def test_multiple_groups_independent(self):
        """Separate accumulators for different dimension groups stay independent."""
        groups: dict[tuple, MetricAccumulator] = {}
        for dim, val in [("a", 1.0), ("b", 10.0), ("a", 2.0), ("b", 20.0)]:
            key = (dim,)
            if key not in groups:
                groups[key] = MetricAccumulator()
            groups[key].add(val)

        aggs = [MetricAggregation(AggregationType.SUM)]
        assert groups[("a",)].finalize(aggs)["SUM"] == 3.0
        assert groups[("b",)].finalize(aggs)["SUM"] == 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_trace_metrics_accumulators.py -v`
Expected: FAIL — `MetricAccumulator` not implemented

- [ ] **Step 3: Implement MetricAccumulator**

Write `src/mlflow_dynamodbstore/trace_metrics/accumulators.py`:

```python
"""Streaming metric accumulators with percentile support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mlflow.entities.trace_metrics import AggregationType, MetricAggregation


@dataclass
class MetricAccumulator:
    """O(1) streaming accumulator for metric aggregations.

    Tracks count, sum, min, max incrementally. Optionally collects raw values
    for percentile computation (linear interpolation matching percentile_cont).

    Args:
        collect_values: If True, collect raw values for percentile computation.
    """

    collect_values: bool = False
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    _values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        """Add a value to the accumulator."""
        self.count += 1
        self.sum += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        if self.collect_values:
            self._values.append(value)

    def finalize(self, aggregations: list[MetricAggregation]) -> dict[str, float]:
        """Compute final aggregation values.

        Returns:
            Dict mapping aggregation label (e.g., "COUNT", "P50.0") to value.
        """
        result: dict[str, float] = {}
        for agg in aggregations:
            match agg.aggregation_type:
                case AggregationType.COUNT:
                    result[str(agg)] = float(self.count)
                case AggregationType.SUM:
                    result[str(agg)] = self.sum
                case AggregationType.AVG:
                    result[str(agg)] = self.sum / self.count if self.count else 0.0
                case AggregationType.MIN:
                    result[str(agg)] = self.min if self.count else 0.0
                case AggregationType.MAX:
                    result[str(agg)] = self.max if self.count else 0.0
                case AggregationType.PERCENTILE:
                    result[str(agg)] = self._percentile(agg.percentile_value or 0.0)
        return result

    def _percentile(self, p: float) -> float:
        """Compute percentile using linear interpolation (percentile_cont)."""
        if not self._values:
            return float("nan")
        sorted_vals = sorted(self._values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        # percentile_cont: rank = p/100 * (n - 1)
        rank = (p / 100.0) * (n - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return sorted_vals[lower]
        fraction = rank - lower
        return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trace_metrics_accumulators.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/trace_metrics/accumulators.py tests/unit/test_trace_metrics_accumulators.py
git commit -m "feat(trace-metrics): implement MetricAccumulator with percentile support"
```

---

### Task 3: Write Path — `log_spans` Changes

Move META denormalization from read path to write path. Add individual span items, trace metric items (token usage), and span metric items (costs).

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (log_spans method, ~line 2572)
- Create: `tests/unit/test_trace_metrics_write_path.py`

- [ ] **Step 1: Write tests for log_spans write path**

Create `tests/unit/test_trace_metrics_write_path.py`:

```python
"""Unit tests for trace metrics write path (log_spans + assessment denorm)."""

import json
import time

import pytest
from mlflow.entities import (
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
)
from mlflow.entities.trace_location import MlflowExperimentLocation

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_TRACE_PREFIX,
)


def _make_trace_info(experiment_id: str, trace_id: str = "tr-test") -> TraceInfo:
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=int(time.time() * 1000),
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={},
        tags={},
    )


class _FakeSpan:
    """Minimal span-like object for testing log_spans."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        name: str = "test-span",
        span_type: str = "CHAIN",
        status: str = "OK",
        start_time_ns: int = 1_000_000_000,
        end_time_ns: int = 2_000_000_000,
        parent_id: str | None = None,
        attributes: dict | None = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.span_type = span_type
        self.status = status
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.parent_id = parent_id
        self._attributes = attributes or {}

    def to_dict(self) -> dict:
        d = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "span_type": self.span_type,
            "status": self.status,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "parent_id": self.parent_id,
            "attributes": self._attributes,
        }
        return d


class TestLogSpansWritePath:
    """Test that log_spans creates individual span items."""

    def test_creates_individual_span_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("span-items-exp")
        trace_id = "tr-span-items"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [
            _FakeSpan(trace_id, "span-1", name="ChatModel", span_type="LLM"),
            _FakeSpan(trace_id, "span-2", name="Retriever", span_type="RETRIEVER"),
        ]
        tracking_store.log_spans(exp_id, spans)

        # Verify individual span items exist
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        span_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#SPAN#"
        )
        assert len(span_items) == 2
        names = {item["name"] for item in span_items}
        assert names == {"ChatModel", "Retriever"}
        types = {item["type"] for item in span_items}
        assert types == {"LLM", "RETRIEVER"}

    def test_span_item_has_duration_ms(self, tracking_store):
        exp_id = tracking_store.create_experiment("span-dur-exp")
        trace_id = "tr-span-dur"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [
            _FakeSpan(
                trace_id,
                "span-1",
                start_time_ns=1_000_000_000,
                end_time_ns=1_500_000_000,
            ),
        ]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#SPAN#"
        )
        assert len(items) == 1
        assert items[0]["duration_ms"] == 500

    def test_span_item_extracts_model_attributes(self, tracking_store):
        exp_id = tracking_store.create_experiment("span-model-exp")
        trace_id = "tr-span-model"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [
            _FakeSpan(
                trace_id,
                "span-1",
                attributes={
                    "mlflow.llm.model": "gpt-4",
                    "mlflow.llm.provider": "openai",
                },
            ),
        ]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#SPAN#"
        )
        assert items[0]["model_name"] == "gpt-4"
        assert items[0]["model_provider"] == "openai"

    def test_denormalizes_meta_on_write(self, tracking_store):
        """META span_types/span_names/span_statuses written at log_spans time."""
        exp_id = tracking_store.create_experiment("meta-denorm-exp")
        trace_id = "tr-meta-denorm"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [
            _FakeSpan(trace_id, "s1", name="ChatModel", span_type="LLM", status="OK"),
            _FakeSpan(trace_id, "s2", name="Retriever", span_type="RETRIEVER", status="OK"),
        ]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        assert meta["span_types"] == {"LLM", "RETRIEVER"}
        assert meta["span_names"] == {"ChatModel", "Retriever"}
        assert meta["span_statuses"] == {"OK"}


class TestLogSpansTraceMetrics:
    """Test that log_spans creates trace metric items (token usage)."""

    def test_creates_trace_metric_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("tmetric-exp")
        trace_id = "tr-tmetric"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [
            _FakeSpan(
                trace_id,
                "s1",
                attributes={
                    "mlflow.chat.tokenUsage": json.dumps(
                        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
                    ),
                },
            ),
            _FakeSpan(
                trace_id,
                "s2",
                attributes={
                    "mlflow.chat.tokenUsage": json.dumps(
                        {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300}
                    ),
                },
            ),
        ]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        metric_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#TMETRIC#"
        )
        metrics = {item["key"]: item["value"] for item in metric_items}
        assert metrics["input_tokens"] == 300
        assert metrics["output_tokens"] == 150
        assert metrics["total_tokens"] == 450

    def test_no_trace_metrics_without_token_usage(self, tracking_store):
        exp_id = tracking_store.create_experiment("no-tmetric-exp")
        trace_id = "tr-no-tmetric"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [_FakeSpan(trace_id, "s1")]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        metric_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#TMETRIC#"
        )
        assert len(metric_items) == 0


class TestLogSpansSpanMetrics:
    """Test that log_spans creates span metric items (costs)."""

    def test_creates_span_metric_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("smetric-exp")
        trace_id = "tr-smetric"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [
            _FakeSpan(
                trace_id,
                "s1",
                attributes={
                    "mlflow.llm.cost": json.dumps(
                        {"input_cost": 0.01, "output_cost": 0.02, "total_cost": 0.03}
                    ),
                },
            ),
        ]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        metric_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#SMETRIC#"
        )
        metrics = {item["key"]: item["value"] for item in metric_items}
        assert metrics["input_cost"] == pytest.approx(0.01)
        assert metrics["output_cost"] == pytest.approx(0.02)
        assert metrics["total_cost"] == pytest.approx(0.03)

    def test_no_span_metrics_without_costs(self, tracking_store):
        exp_id = tracking_store.create_experiment("no-smetric-exp")
        trace_id = "tr-no-smetric"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        spans = [_FakeSpan(trace_id, "s1")]
        tracking_store.log_spans(exp_id, spans)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        metric_items = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#SMETRIC#"
        )
        assert len(metric_items) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_trace_metrics_write_path.py -v -k "LogSpans"`
Expected: FAIL — span items, metrics, and META denorm not written by log_spans

- [ ] **Step 3: Implement log_spans changes**

Modify `src/mlflow_dynamodbstore/tracking_store.py` `log_spans` method (~line 2572). Replace the current implementation with:

```python
def log_spans(
    self, location: str, spans: list[Any], tracking_uri: str | None = None
) -> list[Any]:
    """Log spans to the tracking store.

    Writes:
    1. SPANS cache item (JSON blob of all span dicts)
    2. Individual span items (T#<trace_id>#SPAN#<span_id>)
    3. Trace metric items (T#<trace_id>#TMETRIC#<key>) — aggregated token usage
    4. Span metric items (T#<trace_id>#SMETRIC#<span_id>#<key>) — per-span costs
    5. META denormalization (span_types, span_names, span_statuses)
    6. FTS items for span names
    """
    import json as _json
    from collections import defaultdict

    from mlflow_dynamodbstore.dynamodb.schema import (
        SK_SPAN_METRIC_PREFIX,
        SK_SPAN_PREFIX,
        SK_TRACE_METRIC_PREFIX,
    )

    if not spans:
        return []

    # Group spans by trace_id
    spans_by_trace: dict[str, list[Any]] = defaultdict(list)
    for span in spans:
        spans_by_trace[span.trace_id].append(span)

    for trace_id, trace_spans in spans_by_trace.items():
        try:
            experiment_id = location or self._resolve_trace_experiment(trace_id)
        except MlflowException:
            if not location:
                continue
            experiment_id = location

        pk = f"{PK_EXPERIMENT_PREFIX}{experiment_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}"

        # Read TTL from trace META
        meta = self._table.get_item(pk=pk, sk=sk)
        ttl = int(meta["ttl"]) if meta and "ttl" in meta else self._get_trace_ttl()

        span_dicts = [s.to_dict() for s in trace_spans]

        # 1. Write SPANS cache item (existing behavior)
        spans_item: dict[str, Any] = {
            "PK": pk,
            "SK": f"{sk}#SPANS",
            "data": _json.dumps(span_dicts),
        }
        if ttl is not None:
            spans_item["ttl"] = ttl
        self._table.put_item(spans_item)

        # 2-4. Build individual span items, trace metrics, span metrics
        # NOTE: Read from span objects directly (not to_dict()) for reliable
        # field access. The to_dict() format varies between V3/X-Ray formats,
        # but the span object properties are stable.
        batch_items: list[dict[str, Any]] = []
        token_totals: dict[str, float] = {}
        span_types: set[str] = set()
        span_names: set[str] = set()
        span_statuses: set[str] = set()

        for span_obj, sd in zip(trace_spans, span_dicts):
            # Read from span object properties (stable API)
            span_id = getattr(span_obj, "span_id", sd.get("span_id", ""))
            name = getattr(span_obj, "name", sd.get("name", ""))
            span_type = getattr(span_obj, "span_type", sd.get("span_type", ""))
            status = getattr(span_obj, "status", sd.get("status", ""))
            # Handle status that may be a StatusCode object or dict
            if hasattr(status, "status_code"):
                status = status.status_code
            elif isinstance(status, dict):
                status = status.get("code", str(status))
            else:
                status = str(status)
            start_ns = int(getattr(span_obj, "start_time_ns", sd.get("start_time_ns", 0)))
            end_ns = int(getattr(span_obj, "end_time_ns", sd.get("end_time_ns", 0)))
            duration_ms = (end_ns - start_ns) // 1_000_000
            # Attributes: try span object, fall back to dict
            attributes = sd.get("attributes", {})

            # Collect for META denormalization
            if span_type:
                span_types.add(span_type)
            if name:
                span_names.add(name)
            if status:
                span_statuses.add(status)

            # Individual span item
            span_item: dict[str, Any] = {
                "PK": pk,
                "SK": f"{sk}{SK_SPAN_PREFIX}{span_id}",
                "name": name,
                "type": span_type,
                "status": status,
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "duration_ms": duration_ms,
            }
            # Extract model attributes
            model_name = attributes.get("mlflow.llm.model")
            model_provider = attributes.get("mlflow.llm.provider")
            if model_name:
                span_item["model_name"] = (
                    _json.loads(model_name) if isinstance(model_name, str)
                    and model_name.startswith('"') else model_name
                )
            if model_provider:
                span_item["model_provider"] = (
                    _json.loads(model_provider) if isinstance(model_provider, str)
                    and model_provider.startswith('"') else model_provider
                )
            if ttl is not None:
                span_item["ttl"] = ttl
            batch_items.append(span_item)

            # Span metric items (costs)
            cost_attr = attributes.get("mlflow.llm.cost")
            if cost_attr:
                try:
                    cost_dict = _json.loads(cost_attr) if isinstance(cost_attr, str) else cost_attr
                    for cost_key, cost_value in cost_dict.items():
                        if cost_value is not None:
                            smetric_item: dict[str, Any] = {
                                "PK": pk,
                                "SK": f"{sk}{SK_SPAN_METRIC_PREFIX}{span_id}#{cost_key}",
                                "span_id": span_id,
                                "key": cost_key,
                                "value": float(cost_value),
                            }
                            if ttl is not None:
                                smetric_item["ttl"] = ttl
                            batch_items.append(smetric_item)
                except (ValueError, TypeError):
                    pass

            # Accumulate token usage for trace-level metrics
            usage_attr = attributes.get("mlflow.chat.tokenUsage")
            if usage_attr:
                try:
                    usage = _json.loads(usage_attr) if isinstance(usage_attr, str) else usage_attr
                    for token_key in ("input_tokens", "output_tokens", "total_tokens"):
                        val = usage.get(token_key)
                        if val is not None:
                            token_totals[token_key] = token_totals.get(token_key, 0) + float(val)
                except (ValueError, TypeError):
                    pass

        # Trace metric items (aggregated token usage)
        for token_key, total in token_totals.items():
            tmetric_item: dict[str, Any] = {
                "PK": pk,
                "SK": f"{sk}{SK_TRACE_METRIC_PREFIX}{token_key}",
                "key": token_key,
                "value": total,
            }
            if ttl is not None:
                tmetric_item["ttl"] = ttl
            batch_items.append(tmetric_item)

        # Batch write all new items
        if batch_items:
            self._table.batch_write(batch_items)

        # 5. META denormalization
        updates: dict[str, Any] = {}
        if span_types:
            updates["span_types"] = span_types
        if span_statuses:
            updates["span_statuses"] = span_statuses
        if span_names:
            updates["span_names"] = span_names
        if updates:
            self._table.update_item(pk=pk, sk=sk, updates=updates)

        # 6. FTS items for span names
        if span_names:
            span_names_text = " ".join(sorted(span_names))
            fts_items = fts_items_for_text(
                pk=pk,
                entity_type="T",
                entity_id=trace_id,
                field="spans",
                text=span_names_text,
            )
            if ttl is not None:
                for item in fts_items:
                    item["ttl"] = ttl
            self._table.batch_write(fts_items)

    return spans
```

Also update `_get_trace_with_spans` to skip denormalization if META already has span sets (idempotent check). Find the denormalization block (~line 2528-2565) and wrap it with:

```python
# Denormalize span attributes on META (skip if already done by log_spans)
if span_dicts and not meta_already_has_span_sets:
    # ... existing denormalization code ...
```

Where `meta_already_has_span_sets` checks:
```python
meta_item = self._table.get_item(pk=pk, sk=sk)
meta_already_has_span_sets = bool(meta_item and meta_item.get("span_types"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trace_metrics_write_path.py -v -k "LogSpans"`
Expected: All PASS

- [ ] **Step 5: Run existing trace tests to verify no regressions**

Run: `uv run pytest tests/unit/test_tracking_traces.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_trace_metrics_write_path.py
git commit -m "feat(trace-metrics): add span items, trace/span metrics, and META denorm to log_spans"
```

---

### Task 4: Assessment Denormalization

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py` (create_assessment ~line 3344, update_assessment ~line 3412)
- Modify: `tests/unit/test_trace_metrics_write_path.py` (add assessment tests)

- [ ] **Step 1: Write assessment denormalization tests**

Add to `tests/unit/test_trace_metrics_write_path.py`:

```python
from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback, Expectation


def _make_feedback(trace_id: str, name: str = "quality", value: str = "good") -> Feedback:
    return Feedback(
        name=name,
        source=AssessmentSource(source_type="HUMAN", source_id="user1"),
        trace_id=trace_id,
        feedback=value,
    )


class TestAssessmentDenormalization:
    """Test that create/update_assessment denormalize top-level attributes."""

    def test_create_assessment_denormalizes(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-denorm-exp")
        trace_id = "tr-assess-denorm"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        assessment = _make_feedback(trace_id, name="quality", value="good")
        created = tracking_store.create_assessment(assessment)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)

        assert item["name"] == "quality"
        assert item["assessment_type"] == "feedback"
        assert "created_timestamp" in item

    def test_numeric_value_from_bool_true(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-bool-exp")
        trace_id = "tr-assess-bool"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        assessment = _make_feedback(trace_id, value="yes")
        created = tracking_store.create_assessment(assessment)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item.get("numeric_value") == 1.0

    def test_numeric_value_from_non_numeric(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-str-exp")
        trace_id = "tr-assess-str"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        assessment = _make_feedback(trace_id, value="some text")
        created = tracking_store.create_assessment(assessment)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert "numeric_value" not in item

    def test_update_assessment_updates_denormalized(self, tracking_store):
        exp_id = tracking_store.create_experiment("assess-update-exp")
        trace_id = "tr-assess-update"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))

        assessment = _make_feedback(trace_id, name="quality", value="good")
        created = tracking_store.create_assessment(assessment)

        tracking_store.update_assessment(
            trace_id=trace_id,
            assessment_id=created.assessment_id,
            name="renamed",
            feedback="excellent",
        )

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{created.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item["name"] == "renamed"
        assert item["assessment_type"] == "feedback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_trace_metrics_write_path.py -v -k "Assessment"`
Expected: FAIL — denormalized attributes not on assessment items

- [ ] **Step 3: Implement assessment denormalization**

Add a helper method to `tracking_store.py`:

```python
@staticmethod
def _parse_assessment_numeric_value(assess_dict: dict) -> float | None:
    """Parse numeric value from assessment dict for denormalization.

    Handles: float, int, true/false, "yes"/"no". Returns None for non-numeric.
    """
    import json as _json

    # Determine value source: feedback or expectation
    fb = assess_dict.get("feedback", {})
    ex = assess_dict.get("expectation", {})
    raw = fb.get("value") if fb else ex.get("value")
    if raw is None:
        return None

    # Handle JSON-encoded values
    val_str = str(raw)
    if val_str in ("True", "true", "yes"):
        return 1.0
    if val_str in ("False", "false", "no"):
        return 0.0
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return None


def _denormalize_assessment_item(self, item: dict, assess_dict: dict) -> None:
    """Add denormalized top-level attributes to an assessment item."""
    item["name"] = assess_dict.get("assessment_name", "")
    item["assessment_type"] = "feedback" if "feedback" in assess_dict else "expectation"
    numeric_val = self._parse_assessment_numeric_value(assess_dict)
    if numeric_val is not None:
        item["numeric_value"] = numeric_val

    # Parse created_timestamp from proto timestamp
    create_time = assess_dict.get("create_time", {})
    if isinstance(create_time, dict):
        seconds = int(create_time.get("seconds", 0))
        nanos = int(create_time.get("nanos", 0))
        item["created_timestamp"] = seconds * 1000 + nanos // 1_000_000
    elif isinstance(create_time, (int, float)):
        item["created_timestamp"] = int(create_time)
```

Modify `create_assessment` (~line 3379) — add after building the item dict, before `self._table.put_item(item)`:

```python
self._denormalize_assessment_item(item, assess_dict)
```

Modify `update_assessment` (~line 3462) — add after `item["data"] = assess_dict`, before `self._table.put_item(item)`:

```python
self._denormalize_assessment_item(item, assess_dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trace_metrics_write_path.py -v -k "Assessment"`
Expected: All PASS

- [ ] **Step 5: Run existing assessment tests for regressions**

Run: `uv run pytest tests/unit/test_tracking_traces.py -v -k "assessment or Assessment"`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_trace_metrics_write_path.py
git commit -m "feat(trace-metrics): denormalize assessment attributes for query_trace_metrics"
```

---

### Task 5: Filters Module

**Files:**
- Create: `src/mlflow_dynamodbstore/trace_metrics/filters.py`
- Create: `tests/unit/test_trace_metrics_filters.py` (inline in existing test file or separate)

- [ ] **Step 1: Write filter tests**

Add to `tests/unit/test_trace_metrics_write_path.py` or create `tests/unit/test_trace_metrics_filters.py`:

```python
"""Unit tests for trace metrics filter parsing and application."""

import pytest
from mlflow.entities.trace_metrics import MetricViewType

from mlflow_dynamodbstore.trace_metrics.filters import apply_trace_metric_filters


class TestTraceMetricFilters:
    def test_trace_status_filter_pass(self):
        meta = {"state": "OK"}
        result = apply_trace_metric_filters(
            meta_item=meta,
            filters=["trace.status = 'OK'"],
            view_type=MetricViewType.TRACES,
        )
        assert result is True

    def test_trace_status_filter_fail(self):
        meta = {"state": "ERROR"}
        result = apply_trace_metric_filters(
            meta_item=meta,
            filters=["trace.status = 'OK'"],
            view_type=MetricViewType.TRACES,
        )
        assert result is False

    def test_span_filter_rejected_for_traces_view(self):
        meta = {"state": "OK"}
        with pytest.raises(Exception, match="only supported for"):
            apply_trace_metric_filters(
                meta_item=meta,
                filters=["span.name = 'ChatModel'"],
                view_type=MetricViewType.TRACES,
            )

    def test_assessment_filter_rejected_for_spans_view(self):
        meta = {"state": "OK"}
        with pytest.raises(Exception, match="only supported for"):
            apply_trace_metric_filters(
                meta_item=meta,
                filters=["assessment.name = 'score'"],
                view_type=MetricViewType.SPANS,
            )

    def test_empty_filters_passes(self):
        meta = {"state": "OK"}
        result = apply_trace_metric_filters(
            meta_item=meta,
            filters=None,
            view_type=MetricViewType.TRACES,
        )
        assert result is True


class TestSpanItemFilters:
    def test_span_name_filter(self):
        from mlflow_dynamodbstore.trace_metrics.filters import filter_span_items

        span_items = [
            {"name": "ChatModel", "type": "LLM", "status": "OK"},
            {"name": "Retriever", "type": "RETRIEVER", "status": "OK"},
        ]
        result = filter_span_items(span_items, ["span.name = 'ChatModel'"])
        assert len(result) == 1
        assert result[0]["name"] == "ChatModel"

    def test_span_type_filter(self):
        from mlflow_dynamodbstore.trace_metrics.filters import filter_span_items

        span_items = [
            {"name": "a", "type": "LLM", "status": "OK"},
            {"name": "b", "type": "CHAIN", "status": "OK"},
        ]
        result = filter_span_items(span_items, ["span.type = 'LLM'"])
        assert len(result) == 1
        assert result[0]["name"] == "a"


class TestAssessmentItemFilters:
    def test_assessment_name_filter(self):
        from mlflow_dynamodbstore.trace_metrics.filters import filter_assessment_items

        items = [
            {"name": "quality", "assessment_type": "feedback"},
            {"name": "accuracy", "assessment_type": "expectation"},
        ]
        result = filter_assessment_items(items, ["assessment.name = 'quality'"])
        assert len(result) == 1
        assert result[0]["name"] == "quality"

    def test_assessment_type_filter(self):
        from mlflow_dynamodbstore.trace_metrics.filters import filter_assessment_items

        items = [
            {"name": "q1", "assessment_type": "feedback"},
            {"name": "q2", "assessment_type": "expectation"},
        ]
        result = filter_assessment_items(items, ["assessment.type = 'feedback'"])
        assert len(result) == 1
        assert result[0]["assessment_type"] == "feedback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_trace_metrics_filters.py -v`
Expected: FAIL — module not implemented

- [ ] **Step 3: Implement filters module**

Write `src/mlflow_dynamodbstore/trace_metrics/filters.py`:

```python
"""Filter parsing and application for trace metrics queries."""

from __future__ import annotations

from mlflow.entities.trace_metrics import MetricViewType
from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchTraceMetricsUtils


def apply_trace_metric_filters(
    meta_item: dict,
    filters: list[str] | None,
    view_type: MetricViewType,
    tag_items: list[dict] | None = None,
    metadata_items: list[dict] | None = None,
) -> bool:
    """Apply trace-level filters against a META item.

    Returns True if the trace passes all filters, False otherwise.
    Raises MlflowException if a filter is invalid for the view type.
    """
    if not filters:
        return True

    for filter_string in filters:
        parsed = SearchTraceMetricsUtils.parse_search_filter(filter_string)

        if parsed.view_type == "trace":
            if parsed.entity == "status":
                if meta_item.get("state") != parsed.value:
                    return False
            elif parsed.entity == "tag":
                tags = tag_items or []
                if not any(t["key"] == parsed.key and t["value"] == parsed.value for t in tags):
                    return False
            elif parsed.entity == "metadata":
                metadata = metadata_items or []
                if not any(
                    m["key"] == parsed.key and m["value"] == parsed.value for m in metadata
                ):
                    return False
        elif parsed.view_type == "span":
            if view_type != MetricViewType.SPANS:
                raise MlflowException.invalid_parameter_value(
                    f"Filtering by span is only supported for {MetricViewType.SPANS} view "
                    f"type, got {view_type}",
                )
            # Span filters handled at span-item level, not META level.
            # Pre-filter uses META denormalized sets (handled by caller).
            pass
        elif parsed.view_type == "assessment":
            if view_type != MetricViewType.ASSESSMENTS:
                raise MlflowException.invalid_parameter_value(
                    f"Filtering by assessment is only supported for "
                    f"{MetricViewType.ASSESSMENTS} view type, got {view_type}",
                )
            # Assessment filters handled at assessment-item level.
            pass

    return True


def _extract_span_filters(filters: list[str] | None) -> list[dict]:
    """Extract span-specific filters from filter list."""
    if not filters:
        return []
    result = []
    for f in filters:
        parsed = SearchTraceMetricsUtils.parse_search_filter(f)
        if parsed.view_type == "span":
            result.append({"entity": parsed.entity, "value": parsed.value})
    return result


def filter_span_items(
    span_items: list[dict],
    filters: list[str] | None,
) -> list[dict]:
    """Filter individual span items by span-level filters."""
    span_filters = _extract_span_filters(filters)
    if not span_filters:
        return span_items

    entity_to_field = {"name": "name", "status": "status", "type": "type"}

    result = []
    for item in span_items:
        if all(
            item.get(entity_to_field.get(sf["entity"], sf["entity"])) == sf["value"]
            for sf in span_filters
        ):
            result.append(item)
    return result


def _extract_assessment_filters(filters: list[str] | None) -> list[dict]:
    """Extract assessment-specific filters from filter list."""
    if not filters:
        return []
    result = []
    for f in filters:
        parsed = SearchTraceMetricsUtils.parse_search_filter(f)
        if parsed.view_type == "assessment":
            result.append({"entity": parsed.entity, "value": parsed.value})
    return result


def filter_assessment_items(
    assessment_items: list[dict],
    filters: list[str] | None,
) -> list[dict]:
    """Filter assessment items by assessment-level filters."""
    assess_filters = _extract_assessment_filters(filters)
    if not assess_filters:
        return assessment_items

    entity_to_field = {"name": "name", "type": "assessment_type"}

    result = []
    for item in assessment_items:
        if all(
            item.get(entity_to_field.get(af["entity"], af["entity"])) == af["value"]
            for af in assess_filters
        ):
            result.append(item)
    return result


def meta_prefilter_spans(
    meta_item: dict,
    filters: list[str] | None,
) -> bool:
    """Fast pre-filter: check META denormalized sets before fetching span items.

    Returns True if trace might have matching spans, False if definitely not.
    """
    span_filters = _extract_span_filters(filters)
    if not span_filters:
        return True

    field_map = {"name": "span_names", "type": "span_types", "status": "span_statuses"}

    for sf in span_filters:
        meta_field = field_map.get(sf["entity"])
        if not meta_field:
            continue
        values = meta_item.get(meta_field)
        if not values or sf["value"] not in values:
            return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trace_metrics_filters.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/trace_metrics/filters.py tests/unit/test_trace_metrics_filters.py
git commit -m "feat(trace-metrics): implement filter parsing and application module"
```

---

### Task 6: Extractors Module

View-specific data extraction: takes DynamoDB items and yields `(dimension_key, value)` tuples.

**Files:**
- Create: `src/mlflow_dynamodbstore/trace_metrics/extractors.py`
- Add tests inline with query tests (Task 8)

- [ ] **Step 1: Implement extractors**

Write `src/mlflow_dynamodbstore/trace_metrics/extractors.py`:

```python
"""View-specific data extraction from DynamoDB items."""

from __future__ import annotations

from datetime import datetime, timezone

from mlflow.entities.trace_metrics import MetricViewType
from mlflow.tracing.constant import (
    AssessmentMetricKey,
    SpanMetricKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
)

TIME_BUCKET_LABEL = "time_bucket"


def compute_time_bucket(timestamp_ms: int, interval_seconds: int) -> str:
    """Floor timestamp to nearest bucket and return ISO 8601 UTC string."""
    bucket_ms = interval_seconds * 1000
    bucket_start_ms = (timestamp_ms // bucket_ms) * bucket_ms
    dt = datetime.fromtimestamp(bucket_start_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def get_timestamp_for_view(
    view_type: MetricViewType,
    item: dict,
    meta_item: dict | None = None,
) -> int:
    """Get the relevant timestamp (ms) for time bucketing based on view type."""
    if view_type == MetricViewType.TRACES:
        return int(meta_item.get("request_time", 0) if meta_item else 0)
    elif view_type == MetricViewType.SPANS:
        return int(item.get("start_time_ns", 0)) // 1_000_000
    elif view_type == MetricViewType.ASSESSMENTS:
        return int(item.get("created_timestamp", 0))
    return 0


def build_dimension_key(
    dimensions: list[str] | None,
    view_type: MetricViewType,
    item: dict,
    meta_item: dict | None = None,
    trace_tags: dict[str, str] | None = None,
    time_bucket: str | None = None,
) -> tuple:
    """Build a dimension key tuple for grouping.

    Returns a tuple of dimension values in order: (time_bucket, dim1, dim2, ...).
    """
    parts: list[str | None] = []
    if time_bucket is not None:
        parts.append(time_bucket)

    for dim in dimensions or []:
        val = _extract_dimension_value(dim, view_type, item, meta_item, trace_tags)
        parts.append(val)

    return tuple(parts)


def _extract_dimension_value(
    dimension: str,
    view_type: MetricViewType,
    item: dict,
    meta_item: dict | None = None,
    trace_tags: dict[str, str] | None = None,
) -> str | None:
    """Extract a single dimension value from an item."""
    if view_type == MetricViewType.TRACES:
        if dimension == TraceMetricDimensionKey.TRACE_NAME:
            tags = trace_tags or {}
            return tags.get("mlflow.traceName")
        elif dimension == TraceMetricDimensionKey.TRACE_STATUS:
            return meta_item.get("state") if meta_item else None
    elif view_type == MetricViewType.SPANS:
        field_map = {
            "span_name": "name",
            "span_type": "type",
            "span_status": "status",
            "span_model_name": "model_name",
            "span_model_provider": "model_provider",
        }
        return item.get(field_map.get(dimension, dimension))
    elif view_type == MetricViewType.ASSESSMENTS:
        field_map = {
            "assessment_name": "name",
            "assessment_value": "assessment_value_str",
        }
        if dimension == "assessment_value":
            # For grouping, use the string representation
            fb = item.get("data", {}).get("feedback", {})
            ex = item.get("data", {}).get("expectation", {})
            raw = fb.get("value") if fb else ex.get("value")
            return str(raw) if raw is not None else None
        return item.get(field_map.get(dimension, dimension))
    return None


def extract_metric_value(
    metric_name: str,
    view_type: MetricViewType,
    item: dict,
    meta_item: dict | None = None,
    trace_metric_items: list[dict] | None = None,
    span_metric_items: list[dict] | None = None,
) -> float | None:
    """Extract the metric value to aggregate from an item.

    Returns None if the metric value is not available.
    """
    if view_type == MetricViewType.TRACES:
        if metric_name == TraceMetricKey.TRACE_COUNT:
            return 1.0  # Each trace counts as 1
        elif metric_name == TraceMetricKey.LATENCY:
            return float(meta_item.get("execution_duration", 0)) if meta_item else None
        elif metric_name in TraceMetricKey.token_usage_keys():
            # Look up from trace metric items
            if trace_metric_items:
                for mi in trace_metric_items:
                    if mi["key"] == metric_name:
                        return float(mi["value"])
            return None

    elif view_type == MetricViewType.SPANS:
        if metric_name == SpanMetricKey.SPAN_COUNT:
            return 1.0  # Each span counts as 1
        elif metric_name == SpanMetricKey.LATENCY:
            return float(item.get("duration_ms", 0))
        elif metric_name in SpanMetricKey.cost_keys():
            # Look up from span metric items for this span
            span_id = item.get("span_id") or item.get("SK", "").split("#SPAN#")[-1]
            if span_metric_items:
                for mi in span_metric_items:
                    if mi.get("span_id") == span_id and mi["key"] == metric_name:
                        return float(mi["value"])
            return None

    elif view_type == MetricViewType.ASSESSMENTS:
        if metric_name == AssessmentMetricKey.ASSESSMENT_COUNT:
            return 1.0  # Each assessment counts as 1
        elif metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
            return item.get("numeric_value")

    return None
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from mlflow_dynamodbstore.trace_metrics.extractors import extract_metric_value, build_dimension_key; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/trace_metrics/extractors.py
git commit -m "feat(trace-metrics): implement view-specific data extractors"
```

---

### Task 7: Pagination Module (DynamoDB Cache)

**Files:**
- Create: `src/mlflow_dynamodbstore/trace_metrics/pagination.py`
- Tests integrated in Task 8

- [ ] **Step 1: Implement pagination module**

Write `src/mlflow_dynamodbstore/trace_metrics/pagination.py`:

```python
"""DynamoDB cache and page token management for query_trace_metrics."""

from __future__ import annotations

import hashlib
import json
import time
from base64 import b64decode, b64encode
from typing import Any

from mlflow.entities.trace_metrics import MetricAggregation, MetricDataPoint, MetricViewType

CACHE_TTL_SECONDS = 900  # 15 minutes


def compute_query_hash(
    experiment_ids: list[str],
    view_type: MetricViewType,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None,
    filters: list[str] | None,
    time_interval_seconds: int | None,
    start_time_ms: int | None,
    end_time_ms: int | None,
) -> str:
    """Compute deterministic hash of query parameters."""
    key_parts = {
        "experiment_ids": sorted(experiment_ids),
        "view_type": str(view_type),
        "metric_name": metric_name,
        "aggregations": sorted(str(a) for a in aggregations),
        "dimensions": sorted(dimensions) if dimensions else [],
        "filters": sorted(filters) if filters else [],
        "time_interval_seconds": time_interval_seconds,
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
    }
    canonical = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def encode_page_token(query_hash: str, offset: int) -> str:
    """Encode pagination state as a base64 page token."""
    data = json.dumps({"query_hash": query_hash, "offset": offset})
    return b64encode(data.encode()).decode()


def decode_page_token(token: str) -> dict[str, Any]:
    """Decode a page token into query_hash and offset."""
    data = json.loads(b64decode(token).decode())
    return data


def cache_put(
    table: Any,
    query_hash: str,
    data_points: list[MetricDataPoint],
) -> None:
    """Store aggregation results in DynamoDB cache."""
    from mlflow_dynamodbstore.dynamodb.schema import PK_TMCACHE_PREFIX, SK_TMCACHE_RESULT

    serialized = [
        {
            "metric_name": dp.metric_name,
            "dimensions": dp.dimensions,
            "values": dp.values,
        }
        for dp in data_points
    ]
    table.put_item(
        {
            "PK": f"{PK_TMCACHE_PREFIX}{query_hash}",
            "SK": SK_TMCACHE_RESULT,
            "data": json.dumps(serialized),
            "ttl": int(time.time()) + CACHE_TTL_SECONDS,
        }
    )


def cache_get(
    table: Any,
    query_hash: str,
) -> list[MetricDataPoint] | None:
    """Retrieve cached aggregation results from DynamoDB.

    Returns None on cache miss or expired entry.
    """
    from mlflow_dynamodbstore.dynamodb.schema import PK_TMCACHE_PREFIX, SK_TMCACHE_RESULT

    item = table.get_item(
        pk=f"{PK_TMCACHE_PREFIX}{query_hash}",
        sk=SK_TMCACHE_RESULT,
    )
    if item is None:
        return None

    # Check TTL (moto may not auto-expire)
    if item.get("ttl") and int(item["ttl"]) < int(time.time()):
        return None

    data = json.loads(item["data"])
    return [
        MetricDataPoint(
            metric_name=d["metric_name"],
            dimensions=d["dimensions"],
            values=d["values"],
        )
        for d in data
    ]
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from mlflow_dynamodbstore.trace_metrics.pagination import compute_query_hash, encode_page_token, decode_page_token; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/trace_metrics/pagination.py
git commit -m "feat(trace-metrics): implement DynamoDB cache and pagination module"
```

---

### Task 8: `query_trace_metrics` Orchestrator

The main method that ties everything together.

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py`
- Create: `tests/unit/test_trace_metrics_query.py`

- [ ] **Step 1: Write orchestrator tests**

Create `tests/unit/test_trace_metrics_query.py`:

```python
"""Unit tests for query_trace_metrics orchestrator."""

import json
import time

import pytest
from mlflow.entities import (
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
)
from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricViewType,
)


def _make_trace_info(experiment_id: str, trace_id: str, **overrides) -> TraceInfo:
    defaults = dict(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=int(time.time() * 1000),
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={},
        tags={},
    )
    defaults.update(overrides)
    return TraceInfo(**defaults)


class _FakeSpan:
    def __init__(self, trace_id, span_id, name="span", span_type="CHAIN",
                 status="OK", start_time_ns=1_000_000_000,
                 end_time_ns=2_000_000_000, parent_id=None, attributes=None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.span_type = span_type
        self.status = status
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.parent_id = parent_id
        self._attributes = attributes or {}

    def to_dict(self):
        return {
            "trace_id": self.trace_id, "span_id": self.span_id,
            "name": self.name, "span_type": self.span_type,
            "status": self.status, "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns, "parent_id": self.parent_id,
            "attributes": self._attributes,
        }


class TestQueryTraceMetricsTraces:
    """TRACES view tests."""

    def test_trace_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-count")
        for i in range(3):
            tracking_store.start_trace(_make_trace_info(exp_id, f"tr-{i}"))
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0

    def test_trace_latency_avg(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-latency")
        for i, dur in enumerate([100, 200, 300]):
            tracking_store.start_trace(
                _make_trace_info(exp_id, f"tr-lat-{i}", execution_duration=dur)
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-lat-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="latency",
            aggregations=[MetricAggregation(AggregationType.AVG)],
        )
        assert len(result) == 1
        assert result[0].values["AVG"] == 200.0

    def test_trace_token_metrics(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-tokens")
        trace_id = "tr-tokens"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [
            _FakeSpan(trace_id, "s1", attributes={
                "mlflow.chat.tokenUsage": json.dumps(
                    {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
                ),
            }),
        ])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="input_tokens",
            aggregations=[MetricAggregation(AggregationType.SUM)],
        )
        assert len(result) == 1
        assert result[0].values["SUM"] == 100.0

    def test_trace_count_with_status_dimension(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-dim-status")
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-ok", state=TraceState.OK))
        tracking_store.log_spans(exp_id, [_FakeSpan("tr-ok", "s1")])
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-err", state=TraceState.ERROR))
        tracking_store.log_spans(exp_id, [_FakeSpan("tr-err", "s2")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            dimensions=["trace_status"],
        )
        assert len(result) == 2
        values = {dp.dimensions["trace_status"]: dp.values["COUNT"] for dp in result}
        assert values["OK"] == 1.0
        assert values["ERROR"] == 1.0

    def test_time_bucketing(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-timebucket")
        base = 1700000000000  # fixed timestamp
        for i in range(3):
            tracking_store.start_trace(
                _make_trace_info(exp_id, f"tr-tb-{i}", request_time=base + i * 60000)
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-tb-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            time_interval_seconds=300,  # 5 min bucket
            start_time_ms=base,
            end_time_ms=base + 300000,
        )
        # All 3 traces should fall in the same 5-min bucket
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0
        assert "time_bucket" in result[0].dimensions

    def test_time_interval_requires_start_end(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-notime")
        with pytest.raises(Exception):
            tracking_store.query_trace_metrics(
                experiment_ids=[exp_id],
                view_type=MetricViewType.TRACES,
                metric_name="trace_count",
                aggregations=[MetricAggregation(AggregationType.COUNT)],
                time_interval_seconds=60,
            )

    def test_trace_status_filter(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-filter")
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-f-ok", state=TraceState.OK))
        tracking_store.log_spans(exp_id, [_FakeSpan("tr-f-ok", "s1")])
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-f-err", state=TraceState.ERROR))
        tracking_store.log_spans(exp_id, [_FakeSpan("tr-f-err", "s2")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            filters=["trace.status = 'OK'"],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 1.0


class TestQueryTraceMetricsSpans:
    """SPANS view tests."""

    def test_span_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-span-count")
        trace_id = "tr-span-cnt"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [
            _FakeSpan(trace_id, "s1", name="A", span_type="LLM"),
            _FakeSpan(trace_id, "s2", name="B", span_type="CHAIN"),
        ])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name="span_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 2.0

    def test_span_latency(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-span-lat")
        trace_id = "tr-span-lat"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [
            _FakeSpan(trace_id, "s1", start_time_ns=0, end_time_ns=100_000_000),
            _FakeSpan(trace_id, "s2", start_time_ns=0, end_time_ns=200_000_000),
        ])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name="latency",
            aggregations=[MetricAggregation(AggregationType.AVG)],
        )
        assert result[0].values["AVG"] == 150.0

    def test_span_name_filter(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-span-filter")
        trace_id = "tr-span-flt"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [
            _FakeSpan(trace_id, "s1", name="ChatModel", span_type="LLM"),
            _FakeSpan(trace_id, "s2", name="Retriever", span_type="RETRIEVER"),
        ])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name="span_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            filters=["span.name = 'ChatModel'"],
        )
        assert result[0].values["COUNT"] == 1.0

    def test_span_type_dimension(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-span-dim")
        trace_id = "tr-span-dim"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [
            _FakeSpan(trace_id, "s1", name="A", span_type="LLM"),
            _FakeSpan(trace_id, "s2", name="B", span_type="LLM"),
            _FakeSpan(trace_id, "s3", name="C", span_type="CHAIN"),
        ])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name="span_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            dimensions=["span_type"],
        )
        values = {dp.dimensions["span_type"]: dp.values["COUNT"] for dp in result}
        assert values["LLM"] == 2.0
        assert values["CHAIN"] == 1.0


class TestQueryTraceMetricsAssessments:
    """ASSESSMENTS view tests."""

    def test_assessment_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-assess-count")
        trace_id = "tr-assess-cnt"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [_FakeSpan(trace_id, "s1")])

        for name in ["quality", "accuracy"]:
            tracking_store.create_assessment(Feedback(
                name=name,
                source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                trace_id=trace_id,
                feedback="good",
            ))

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name="assessment_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 2.0

    def test_assessment_value_avg(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-assess-val")
        trace_id = "tr-assess-val"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [_FakeSpan(trace_id, "s1")])

        for val in ["1.0", "2.0", "3.0"]:
            tracking_store.create_assessment(Feedback(
                name="score",
                source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                trace_id=trace_id,
                feedback=val,
            ))

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name="assessment_value",
            aggregations=[MetricAggregation(AggregationType.AVG)],
        )
        assert len(result) == 1
        assert result[0].values["AVG"] == 2.0


class TestQueryTraceMetricsPagination:
    """Pagination with DynamoDB cache."""

    def test_max_results_limits_output(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-maxresults")
        for i in range(5):
            tracking_store.start_trace(
                _make_trace_info(exp_id, f"tr-mr-{i}", state=TraceState.OK)
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-mr-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            dimensions=["trace_status"],
            max_results=1,
        )
        # Only 1 status group (OK), so max_results=1 should return exactly 1
        assert len(result) <= 1

    def test_pagination_returns_all_results(self, tracking_store):
        exp_id = tracking_store.create_experiment("qtm-paginate")
        base = 1700000000000
        # Create traces across 3 time buckets
        for i in range(6):
            tracking_store.start_trace(
                _make_trace_info(exp_id, f"tr-pg-{i}", request_time=base + i * 120000)
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-pg-{i}", f"s-{i}")])

        # First page
        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            time_interval_seconds=120,
            start_time_ms=base,
            end_time_ms=base + 720000,
            max_results=2,
        )
        assert len(result) <= 2
        if result.token:
            # Second page
            result2 = tracking_store.query_trace_metrics(
                experiment_ids=[exp_id],
                view_type=MetricViewType.TRACES,
                metric_name="trace_count",
                aggregations=[MetricAggregation(AggregationType.COUNT)],
                time_interval_seconds=120,
                start_time_ms=base,
                end_time_ms=base + 720000,
                max_results=2,
                page_token=result.token,
            )
            assert len(result2) >= 1


class TestQueryTraceMetricsMultiExperiment:
    def test_merges_across_experiments(self, tracking_store):
        exp1 = tracking_store.create_experiment("qtm-multi-1")
        exp2 = tracking_store.create_experiment("qtm-multi-2")
        tracking_store.start_trace(_make_trace_info(exp1, "tr-m1"))
        tracking_store.log_spans(exp1, [_FakeSpan("tr-m1", "s1")])
        tracking_store.start_trace(_make_trace_info(exp2, "tr-m2"))
        tracking_store.log_spans(exp2, [_FakeSpan("tr-m2", "s2")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp1, exp2],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
        )
        assert result[0].values["COUNT"] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_trace_metrics_query.py -v`
Expected: FAIL — `query_trace_metrics` not implemented

- [ ] **Step 3: Implement query_trace_metrics**

Add to `src/mlflow_dynamodbstore/tracking_store.py`:

```python
def query_trace_metrics(
    self,
    experiment_ids: list[str],
    view_type: MetricViewType,
    metric_name: str,
    aggregations: list[MetricAggregation],
    dimensions: list[str] | None = None,
    filters: list[str] | None = None,
    time_interval_seconds: int | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    max_results: int = 1000,
    page_token: str | None = None,
) -> PagedList:
    """Query trace metrics with streaming aggregation."""
    from mlflow.store.tracking.utils.sql_trace_metrics_utils import (
        validate_query_trace_metrics_params,
    )

    from mlflow_dynamodbstore.dynamodb.schema import (
        SK_SPAN_METRIC_PREFIX,
        SK_SPAN_PREFIX,
        SK_TRACE_METRIC_PREFIX,
    )
    from mlflow_dynamodbstore.trace_metrics.accumulators import MetricAccumulator
    from mlflow_dynamodbstore.trace_metrics.extractors import (
        TIME_BUCKET_LABEL,
        build_dimension_key,
        compute_time_bucket,
        extract_metric_value,
        get_timestamp_for_view,
    )
    from mlflow_dynamodbstore.trace_metrics.filters import (
        apply_trace_metric_filters,
        filter_assessment_items,
        filter_span_items,
        meta_prefilter_spans,
    )
    from mlflow_dynamodbstore.trace_metrics.pagination import (
        cache_get,
        cache_put,
        compute_query_hash,
        decode_page_token,
        encode_page_token,
    )

    # 1. Validate
    validate_query_trace_metrics_params(view_type, metric_name, aggregations, dimensions)
    if time_interval_seconds and (start_time_ms is None or end_time_ms is None):
        raise MlflowException.invalid_parameter_value(
            "start_time_ms and end_time_ms are required if time_interval_seconds is set"
        )

    # 2. Check cache on pagination
    query_hash = compute_query_hash(
        experiment_ids, view_type, metric_name, aggregations,
        dimensions, filters, time_interval_seconds, start_time_ms, end_time_ms,
    )

    if page_token:
        token_data = decode_page_token(page_token)
        cached = cache_get(self._table, token_data["query_hash"])
        if cached is not None:
            offset = token_data["offset"]
            page = cached[offset : offset + max_results]
            next_token = (
                encode_page_token(token_data["query_hash"], offset + max_results)
                if offset + max_results < len(cached)
                else None
            )
            return PagedList(page, next_token)
        # Cache miss — recompute below
        query_hash = token_data["query_hash"]

    # 3. Stream and aggregate
    has_percentile = any(
        a.aggregation_type == AggregationType.PERCENTILE for a in aggregations
    )
    accumulators: dict[tuple, MetricAccumulator] = {}

    for exp_id in experiment_ids:
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

        # Query trace META items with time range via LSI1
        # table.query() uses: index_name, sk_gte, sk_lte (not "index" or "sk_between")
        if start_time_ms is not None or end_time_ms is not None:
            query_kwargs: dict[str, Any] = {
                "pk": pk,
                "index_name": "lsi1",
            }
            if start_time_ms is not None:
                query_kwargs["sk_gte"] = f"{start_time_ms:020d}"
            if end_time_ms is not None:
                query_kwargs["sk_lte"] = f"{end_time_ms:020d}"
            meta_items = self._table.query(**query_kwargs)
        else:
            # No time filter — query all trace META items
            meta_items = self._table.query(pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}")

        # Filter to trace META items only.
        # META items have "request_time" attribute; sub-items (tags, spans, etc.) do not.
        meta_items = [
            item for item in meta_items
            if "request_time" in item
        ]

        for meta in meta_items:
            trace_id = meta["SK"][len(SK_TRACE_PREFIX):]

            # 4. Apply trace-level filters
            tag_items = None
            metadata_items = None
            if filters:
                # Fetch tags/metadata if needed for filters
                tag_items = self._table.query(
                    pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#TAG#"
                )
                metadata_items = self._table.query(
                    pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#RMETA#"
                )
            if not apply_trace_metric_filters(
                meta, filters, view_type, tag_items, metadata_items
            ):
                continue

            # 5. Fetch view-specific items and aggregate
            if view_type == MetricViewType.TRACES:
                trace_tags = (
                    {t["key"]: t["value"] for t in tag_items}
                    if tag_items
                    else {
                        t["key"]: t["value"]
                        for t in self._table.query(
                            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#TAG#"
                        )
                    }
                )
                trace_metric_items = self._table.query(
                    pk=pk,
                    sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}{SK_TRACE_METRIC_PREFIX}",
                )
                value = extract_metric_value(
                    metric_name, view_type, meta,
                    meta_item=meta,
                    trace_metric_items=trace_metric_items,
                )
                if value is None and metric_name != TraceMetricKey.TRACE_COUNT:
                    continue

                time_bucket = None
                if time_interval_seconds:
                    ts = get_timestamp_for_view(view_type, meta, meta)
                    time_bucket = compute_time_bucket(ts, time_interval_seconds)

                dim_key = build_dimension_key(
                    dimensions, view_type, meta,
                    meta_item=meta, trace_tags=trace_tags,
                    time_bucket=time_bucket,
                )
                if dim_key not in accumulators:
                    accumulators[dim_key] = MetricAccumulator(
                        collect_values=has_percentile
                    )
                if value is not None:
                    accumulators[dim_key].add(value)

            elif view_type == MetricViewType.SPANS:
                # Pre-filter using META denormalized sets
                if not meta_prefilter_spans(meta, filters):
                    continue

                span_items = self._table.query(
                    pk=pk,
                    sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}{SK_SPAN_PREFIX}",
                )
                span_items = filter_span_items(span_items, filters)

                span_metric_items = (
                    self._table.query(
                        pk=pk,
                        sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}{SK_SPAN_METRIC_PREFIX}",
                    )
                    if metric_name in SpanMetricKey.cost_keys()
                    else []
                )

                for span_item in span_items:
                    value = extract_metric_value(
                        metric_name, view_type, span_item,
                        span_metric_items=span_metric_items,
                    )
                    if value is None and metric_name not in (
                        SpanMetricKey.SPAN_COUNT,
                    ):
                        continue

                    time_bucket = None
                    if time_interval_seconds:
                        ts = get_timestamp_for_view(view_type, span_item)
                        time_bucket = compute_time_bucket(ts, time_interval_seconds)

                    dim_key = build_dimension_key(
                        dimensions, view_type, span_item,
                        time_bucket=time_bucket,
                    )
                    if dim_key not in accumulators:
                        accumulators[dim_key] = MetricAccumulator(
                            collect_values=has_percentile
                        )
                    if value is not None:
                        accumulators[dim_key].add(value)

            elif view_type == MetricViewType.ASSESSMENTS:
                assess_items = self._table.query(
                    pk=pk,
                    sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#",
                )
                assess_items = filter_assessment_items(assess_items, filters)

                for assess_item in assess_items:
                    value = extract_metric_value(
                        metric_name, view_type, assess_item,
                    )
                    if value is None and metric_name != AssessmentMetricKey.ASSESSMENT_COUNT:
                        continue

                    time_bucket = None
                    if time_interval_seconds:
                        ts = get_timestamp_for_view(view_type, assess_item)
                        time_bucket = compute_time_bucket(ts, time_interval_seconds)

                    dim_key = build_dimension_key(
                        dimensions, view_type, assess_item,
                        time_bucket=time_bucket,
                    )
                    if dim_key not in accumulators:
                        accumulators[dim_key] = MetricAccumulator(
                            collect_values=has_percentile
                        )
                    if value is not None:
                        accumulators[dim_key].add(value)

    # 7. Finalize
    data_points = []
    for dim_key, acc in accumulators.items():
        values = acc.finalize(aggregations)
        if not values:
            continue

        dims: dict[str, str] = {}
        idx = 0
        if time_interval_seconds:
            dims[TIME_BUCKET_LABEL] = dim_key[idx] if idx < len(dim_key) else ""
            idx += 1
        for dim_name in dimensions or []:
            val = dim_key[idx] if idx < len(dim_key) else None
            if val is not None:
                dims[dim_name] = str(val)
            idx += 1

        # Skip data points with None dimensions
        if any(v is None for v in dim_key):
            continue

        data_points.append(
            MetricDataPoint(
                metric_name=metric_name,
                dimensions=dims,
                values=values,
            )
        )

    # Sort by dimension keys
    data_points.sort(key=lambda dp: tuple(dp.dimensions.values()))

    # 8. Cache and paginate
    cache_put(self._table, query_hash, data_points)

    page = data_points[:max_results]
    next_token = (
        encode_page_token(query_hash, max_results)
        if max_results < len(data_points)
        else None
    )
    return PagedList(page, next_token)
```

Add the required imports at the top of `tracking_store.py`:

```python
from mlflow.entities.trace_metrics import (
    AggregationType,
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.tracing.constant import (
    AssessmentMetricKey,
    SpanMetricKey,
    TraceMetricKey,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_trace_metrics_query.py -v`
Expected: All PASS

- [ ] **Step 5: Run all unit tests for regressions**

Run: `uv run pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py tests/unit/test_trace_metrics_query.py
git commit -m "feat(trace-metrics): implement query_trace_metrics orchestrator"
```

---

### Task 9: Integration + E2E Tests

**Files:**
- Create: `tests/integration/test_trace_metrics.py`
- Create: `tests/e2e/test_trace_metrics.py`

- [ ] **Step 1: Write integration tests**

Create `tests/integration/test_trace_metrics.py`:

```python
"""Integration tests for query_trace_metrics via REST."""

import json
import time

from mlflow.entities import TraceInfo, TraceLocation, TraceLocationType, TraceState
from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType


def _make_trace_info(experiment_id, trace_id, **overrides):
    defaults = dict(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=int(time.time() * 1000),
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={},
        tags={},
    )
    defaults.update(overrides)
    return TraceInfo(**defaults)


class _FakeSpan:
    def __init__(self, trace_id, span_id, name="span", span_type="CHAIN",
                 status="OK", start_time_ns=1_000_000_000,
                 end_time_ns=2_000_000_000, parent_id=None, attributes=None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.span_type = span_type
        self.status = status
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.parent_id = parent_id
        self._attributes = attributes or {}

    def to_dict(self):
        return {
            "trace_id": self.trace_id, "span_id": self.span_id,
            "name": self.name, "span_type": self.span_type,
            "status": self.status, "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns, "parent_id": self.parent_id,
            "attributes": self._attributes,
        }


class TestQueryTraceMetricsIntegration:
    def test_traces_view_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("integ-qtm-count")
        for i in range(3):
            tracking_store.start_trace(_make_trace_info(exp_id, f"tr-i-{i}"))
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-i-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0
        assert result[0].metric_name == "trace_count"

    def test_spans_view_with_dimensions(self, tracking_store):
        exp_id = tracking_store.create_experiment("integ-qtm-spans")
        trace_id = "tr-integ-spans"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [
            _FakeSpan(trace_id, "s1", name="ChatModel", span_type="LLM"),
            _FakeSpan(trace_id, "s2", name="Retriever", span_type="RETRIEVER"),
            _FakeSpan(trace_id, "s3", name="Pipeline", span_type="CHAIN"),
        ])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name="span_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            dimensions=["span_type"],
        )
        types = {dp.dimensions["span_type"]: dp.values["COUNT"] for dp in result}
        assert types["LLM"] == 1.0
        assert types["RETRIEVER"] == 1.0
        assert types["CHAIN"] == 1.0

    def test_assessments_view(self, tracking_store):
        exp_id = tracking_store.create_experiment("integ-qtm-assess")
        trace_id = "tr-integ-assess"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [_FakeSpan(trace_id, "s1")])

        for val in ["1.0", "3.0", "5.0"]:
            tracking_store.create_assessment(Feedback(
                name="score",
                source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                trace_id=trace_id,
                feedback=val,
            ))

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name="assessment_value",
            aggregations=[MetricAggregation(AggregationType.AVG)],
        )
        assert len(result) == 1
        assert result[0].values["AVG"] == 3.0

    def test_pagination_via_cache(self, tracking_store):
        exp_id = tracking_store.create_experiment("integ-qtm-page")
        base = 1700000000000
        for i in range(4):
            tracking_store.start_trace(
                _make_trace_info(exp_id, f"tr-pg-{i}", request_time=base + i * 300000)
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-pg-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="trace_count",
            aggregations=[MetricAggregation(AggregationType.COUNT)],
            time_interval_seconds=300,
            start_time_ms=base,
            end_time_ms=base + 1200000,
            max_results=2,
        )
        # Should get first 2 time buckets
        assert len(result) <= 2
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/integration/test_trace_metrics.py -v`
Expected: All PASS

- [ ] **Step 3: Write E2E tests**

Create `tests/e2e/test_trace_metrics.py`:

```python
"""E2E tests for query_trace_metrics via MLflow SDK."""

import uuid

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import SpanType

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestQueryTraceMetricsE2E:
    def test_trace_count_via_sdk(self, mlflow_server):
        """Create traces via SDK, query metrics via REST."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-count-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="e2e-func")
        def my_func(x):
            return x * 2

        for i in range(3):
            my_func(i)

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)

        result = client.query_trace_metrics(
            experiment_ids=[exp.experiment_id],
            view_type="TRACES",
            metric_name="trace_count",
            aggregations=[{"aggregation_type": "COUNT"}],
        )
        assert len(result) >= 1

    def test_span_count_with_hierarchy(self, mlflow_server):
        """Create hierarchical trace, query span metrics."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-spans-{_uid()}"
        mlflow.set_experiment(exp_name)

        root = mlflow.start_span_no_context(
            name="pipeline", span_type=SpanType.CHAIN,
            inputs={"query": "test"},
        )
        child = mlflow.start_span_no_context(
            name="llm_call", span_type=SpanType.LLM,
            parent_span=root, inputs={"prompt": "test"},
        )
        child.set_outputs({"response": "ok"})
        child.end()
        root.set_outputs({"result": "ok"})
        root.end()

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)

        result = client.query_trace_metrics(
            experiment_ids=[exp.experiment_id],
            view_type="SPANS",
            metric_name="span_count",
            aggregations=[{"aggregation_type": "COUNT"}],
        )
        assert len(result) >= 1
```

- [ ] **Step 4: Run E2E tests**

Run: `uv run pytest tests/e2e/test_trace_metrics.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/compatibility`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_trace_metrics.py tests/e2e/test_trace_metrics.py
git commit -m "test(trace-metrics): add integration and e2e tests for query_trace_metrics"
```

---

## Implementation Notes

### Key Imports Reference

```python
# MLflow entities
from mlflow.entities.trace_metrics import (
    AggregationType, MetricAggregation, MetricDataPoint, MetricViewType,
)
from mlflow.tracing.constant import (
    TraceMetricKey, TraceMetricDimensionKey,
    SpanMetricKey, SpanMetricDimensionKey,
    AssessmentMetricKey, AssessmentMetricDimensionKey,
    SpanAttributeKey, TraceMetadataKey, TokenUsageKey, CostKey,
)
from mlflow.utils.search_utils import SearchTraceMetricsUtils
from mlflow.store.tracking.utils.sql_trace_metrics_utils import (
    validate_query_trace_metrics_params,
)

# Schema
from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX, SK_TRACE_PREFIX,
    SK_SPAN_PREFIX, SK_TRACE_METRIC_PREFIX, SK_SPAN_METRIC_PREFIX,
    PK_TMCACHE_PREFIX, SK_TMCACHE_RESULT,
)
```

### DynamoDB Query Patterns

```python
# LSI1 query with time range (lsi1sk is zero-padded string)
items = table.query(pk=pk, index="lsi1", sk_between=(f"{start:020d}", f"{end:020d}"))

# Prefix query for sub-items
span_items = table.query(pk=pk, sk_prefix=f"T#{trace_id}#SPAN#")
tmetric_items = table.query(pk=pk, sk_prefix=f"T#{trace_id}#TMETRIC#")
smetric_items = table.query(pk=pk, sk_prefix=f"T#{trace_id}#SMETRIC#")
assess_items = table.query(pk=pk, sk_prefix=f"T#{trace_id}#ASSESS#")

# Cache item (direct key access)
cache = table.get_item(pk=f"TMCACHE#{hash}", sk="RESULT")
```

### Span Attribute Keys

```python
SpanAttributeKey.CHAT_USAGE = "mlflow.chat.tokenUsage"  # JSON str with token counts
SpanAttributeKey.LLM_COST = "mlflow.llm.cost"           # JSON str with cost values
SpanAttributeKey.MODEL = "mlflow.llm.model"              # model name
SpanAttributeKey.MODEL_PROVIDER = "mlflow.llm.provider"  # provider name
```

### Assessment Dict Structure

```python
# From Assessment.to_dictionary() (via MessageToDict):
{
    "assessment_name": "quality",
    "feedback": {"value": "good"},     # OR
    "expectation": {"value": "expected output"},
    "create_time": {"seconds": 1710000000, "nanos": 0},
    "last_update_time": {"seconds": 1710000000, "nanos": 0},
    # ... other fields
}
```
