# Phase 4b: `query_trace_metrics` — Trace Analytics Engine

## Context

Phases 1–3 implemented datasets, logged models, scorers, core gaps, and advanced traces. Phase 4a closed remaining simple gaps (`log_outputs`, `log_spans_async`, `get_metric_history_bulk_interval_from_steps`). Phase 4b implements `query_trace_metrics` — the **last remaining `NotImplementedError`** in the DynamoDB tracking store. This method powers the experiment overview dashboard's trace analytics charts (latency, token counts, error rates, cost breakdowns over time).

## Scope

1 store method: `query_trace_metrics`

Supporting changes:
- Schema additions (3 new item types + assessment denormalization)
- Write path changes in `log_spans` and `create_assessment`/`update_assessment`
- New `trace_metrics/` package for aggregation logic

### What `query_trace_metrics` Does

Given experiment IDs, a view type (TRACES/SPANS/ASSESSMENTS), a metric name, and aggregation types, it returns aggregated `MetricDataPoint` objects grouped by optional dimensions (e.g., trace name, span type) and/or time buckets.

**Signature** (from `AbstractStore`):
```python
def query_trace_metrics(
    self,
    experiment_ids: list[str],
    view_type: MetricViewType,         # TRACES, SPANS, ASSESSMENTS
    metric_name: str,                   # e.g., "latency", "trace_count", "input_tokens"
    aggregations: list[MetricAggregation],  # COUNT, SUM, AVG, MIN, MAX, PERCENTILE
    dimensions: list[str] | None = None,    # e.g., ["trace_name", "trace_status"]
    filters: list[str] | None = None,       # e.g., ["trace.status = 'OK'"]
    time_interval_seconds: int | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    max_results: int = 1000,
    page_token: str | None = None,
) -> PagedList[list[MetricDataPoint]]
```

**Return type**: `MetricDataPoint(metric_name, dimensions: dict[str, str], values: dict[str, float])`

### Supported Metrics by View Type

**TRACES view:**

| Metric | Aggregations | Dimensions |
|--------|-------------|------------|
| `trace_count` | COUNT | trace_name, trace_status |
| `latency` | AVG, PERCENTILE | trace_name |
| `input_tokens` | SUM, AVG, PERCENTILE | trace_name |
| `output_tokens` | SUM, AVG, PERCENTILE | trace_name |
| `total_tokens` | SUM, AVG, PERCENTILE | trace_name |

**SPANS view:**

| Metric | Aggregations | Dimensions |
|--------|-------------|------------|
| `span_count` | COUNT | span_name, span_type, span_status, span_model_name, span_model_provider |
| `latency` | AVG, PERCENTILE | span_name, span_status |
| `input_cost` | SUM, AVG, PERCENTILE | span_model_name, span_model_provider |
| `output_cost` | SUM, AVG, PERCENTILE | span_model_name, span_model_provider |
| `total_cost` | SUM, AVG, PERCENTILE | span_model_name, span_model_provider |

**ASSESSMENTS view:**

| Metric | Aggregations | Dimensions |
|--------|-------------|------------|
| `assessment_count` | COUNT | assessment_name, assessment_value |
| `assessment_value` | AVG, PERCENTILE | assessment_name |

### Filters

Exact match only (no LIKE patterns). Parsed by MLflow's `SearchTraceMetricsUtils.parse_search_filter()`:

- **Trace**: `trace.status = 'OK'`, `trace.metadata.key = 'value'`, `trace.tag.key = 'value'`
- **Span**: `span.name = 'ChatModel'`, `span.status = 'ERROR'`, `span.type = 'LLM'`
- **Assessment**: `assessment.name = 'score'`, `assessment.type = 'feedback'`

## DynamoDB Item Design

### New Item Types

**Individual Span Items** — `T#<trace_id>#SPAN#<span_id>`

Written at `log_spans` time. One item per span, with denormalized queryable fields.

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Span | `EXP#<exp_id>` | `T#<trace_id>#SPAN#<span_id>` | name, type, status, start_time_ns, end_time_ns, duration_ms, model_name (optional), model_provider (optional), ttl (optional) |

- `duration_ms`: precomputed `(end_time_ns - start_time_ns) // 1_000_000`
- `model_name` / `model_provider`: extracted from span attributes (keys `SpanAttributeKey.MODEL` = `mlflow.llm.model`, `SpanAttributeKey.MODEL_PROVIDER` = `mlflow.llm.provider`)

**Trace Metric Items** — `T#<trace_id>#TMETRIC#<key>`

Written at `log_spans` time. Aggregated token usage across all spans in the trace.

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Trace Metric | `EXP#<exp_id>` | `T#<trace_id>#TMETRIC#<key>` | key, value (float), ttl (optional) |

- `key`: one of `input_tokens`, `output_tokens`, `total_tokens`
- Token usage extracted from `SpanAttributeKey.CHAT_USAGE` span attributes, summed across all spans

**Span Metric Items** — `T#<trace_id>#SMETRIC#<span_id>#<key>`

Written at `log_spans` time. Per-span cost metrics.

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Span Metric | `EXP#<exp_id>` | `T#<trace_id>#SMETRIC#<span_id>#<key>` | span_id, key, value (float), ttl (optional) |

- `key`: one of `input_cost`, `output_cost`, `total_cost`
- Cost extracted from `SpanAttributeKey.LLM_COST` span attributes (JSON string → dict)

**Query Metrics Cache Items** — `TMCACHE#<query_hash>` / `RESULT`

Caches aggregation results to avoid recomputation on pagination.

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Cache | `TMCACHE#<query_hash>` | `RESULT` | data (JSON-serialized list of MetricDataPoint dicts), ttl |

- `query_hash`: deterministic hash of all query params (sorted experiment_ids, view_type, metric_name, sorted aggregations, sorted dimensions, sorted filters, time_interval_seconds, start_time_ms, end_time_ms). Lists are sorted before hashing to ensure determinism regardless of input order.
- `ttl`: current epoch seconds + 900 (15 minutes)
- **Partition key divergence**: `TMCACHE#` is the first PK prefix that does not use `EXP#`. This is intentional — cache items span multiple experiments and have their own lifecycle (TTL-based auto-expiry). Table-level TTL must be enabled on the `ttl` attribute (already required for trace TTL support).
- **Cache consistency**: Traces added/deleted during the 15-minute window produce stale pages. This is acceptable for dashboard analytics — the SQL store doesn't implement pagination at all, and chart data tolerates slight staleness.

### Assessment Item Denormalization

Existing `T#<trace_id>#ASSESS#<assessment_id>` items gain top-level attributes:

| New Attribute | Source | Notes |
|---------------|--------|-------|
| `name` | `data["assessment_name"]` | Assessment name string |
| `assessment_type` | Derived: `"feedback"` if `data` has `feedback` key, else `"expectation"` | Matches `Assessment.from_dictionary()` logic |
| `numeric_value` | Parsed from `data["feedback"]["value"]` or `data["expectation"]["value"]` | JSON-encoded value; handles: numeric, `"true"`→1.0, `"false"`→0.0, `"yes"`→1.0, `"no"`→0.0, non-numeric→None |
| `created_timestamp` | `data["create_time"]` | Proto timestamp (converted to milliseconds) |

Written in `create_assessment` and `update_assessment`.

### No New GSIs or LSIs

All new items live in the experiment partition under the trace SK prefix. Queried via SK prefix patterns. Time range filtering uses existing LSI1.

## Schema Constants

New constants to add to `dynamodb/schema.py`:

```python
SK_SPAN_PREFIX = "#SPAN#"
SK_TRACE_METRIC_PREFIX = "#TMETRIC#"
SK_SPAN_METRIC_PREFIX = "#SMETRIC#"
PK_TMCACHE_PREFIX = "TMCACHE#"
SK_TMCACHE_RESULT = "RESULT"
```

## Access Patterns

Every method maps to Query, GetItem, BatchWrite, or PutItem — never Scan.

| # | Access Pattern | Caller | Operation | Key Condition | Notes |
|---|---------------|--------|-----------|---------------|-------|
| AP1 | Write individual span | `log_spans` | BatchWrite | `PK=EXP#<exp_id>, SK=T#<trace_id>#SPAN#<span_id>` | One per span |
| AP2 | Write trace metric | `log_spans` | BatchWrite | `PK=EXP#<exp_id>, SK=T#<trace_id>#TMETRIC#<key>` | Up to 3 token keys (input_tokens, output_tokens, total_tokens) |
| AP3 | Write span metric | `log_spans` | BatchWrite | `PK=EXP#<exp_id>, SK=T#<trace_id>#SMETRIC#<span_id>#<key>` | Up to 3 cost keys per span |
| AP4 | Query trace METAs by time range | `query_trace_metrics` | Query (LSI1) | `PK=EXP#<exp_id>, lsi1sk BETWEEN f"{start_ms:020d}" AND f"{end_ms:020d}"` | Paginated via `query_page()`; lsi1sk is zero-padded string |
| AP5 | Fetch spans for a trace | `query_trace_metrics` (SPANS) | Query | `PK=EXP#<exp_id>, SK begins_with T#<trace_id>#SPAN#` | After META pre-filter |
| AP6 | Fetch trace metrics for a trace | `query_trace_metrics` (TRACES) | Query | `PK=EXP#<exp_id>, SK begins_with T#<trace_id>#TMETRIC#` | Token usage |
| AP7 | Fetch span metrics for a trace | `query_trace_metrics` (SPANS) | Query | `PK=EXP#<exp_id>, SK begins_with T#<trace_id>#SMETRIC#` | Span costs |
| AP8 | Fetch assessments for a trace | `query_trace_metrics` (ASSESSMENTS) | Query | `PK=EXP#<exp_id>, SK begins_with T#<trace_id>#ASSESS#` | Already exists |
| AP9 | Cache put | `query_trace_metrics` | PutItem | `PK=TMCACHE#<hash>, SK=RESULT` | TTL = 15 min |
| AP10 | Cache get | `query_trace_metrics` | GetItem | `PK=TMCACHE#<hash>, SK=RESULT` | Direct lookup |

### Scan Risk Assessment

**All 10 access patterns are efficient.** No table scans. AP1-AP3 are batch writes with known keys. AP4 uses LSI1 for time range (most selective filter). AP5-AP8 are scoped to a single trace's sub-items. AP9-AP10 are direct key operations.

## Architecture: Streaming Aggregation (Approach C)

### Why Streaming

Rather than collecting all matching traces into memory before aggregating, we process traces in batches:

- **Constant memory** for COUNT/SUM/AVG/MIN/MAX accumulators (O(1) per dimension group)
- **Natural DynamoDB alignment** — `query_page()` returns paginated results; streaming maps directly to this
- **Immediate work** — aggregation starts on first batch, no waiting for full scan
- **Percentile exception** — percentile computation requires all values, so raw values are collected into a list. This is unavoidable regardless of approach.

### Query Flow

```
1. VALIDATE
   └─ validate_query_trace_metrics_params() — reuse MLflow's validator
   └─ Additional: if time_interval_seconds is set, require start_time_ms and end_time_ms

2. CHECK CACHE (if page_token provided)
   └─ GetItem TMCACHE#<hash>/RESULT → if hit, slice at offset, return page

3. STREAM TRACE META ITEMS
   └─ For each experiment_id:
      └─ query_page(LSI1, time range) → batch of META items
         │
         ├─ 4. APPLY FILTERS per batch
         │     ├─ Trace filters: status (META.state), tags (sub-items), metadata (sub-items)
         │     ├─ Span pre-filter: check META.span_names/span_types/span_statuses sets
         │     └─ Assessment filters: query assessment sub-items
         │
         ├─ 5. FETCH VIEW-SPECIFIC ITEMS for surviving traces
         │     ├─ TRACES: META.execution_duration + TMETRIC items
         │     ├─ SPANS: SPAN items + SMETRIC items
         │     └─ ASSESSMENTS: ASSESS items (denormalized)
         │
         └─ 6. UPDATE ACCUMULATORS
               ├─ Compute dimension key (time bucket + dimension values)
               └─ Feed values into MetricAccumulator per group

7. FINALIZE
   ├─ Compute AVG = sum/count, percentiles via linear interpolation
   ├─ Convert to MetricDataPoint objects
   ├─ Sort by dimension keys (time bucket first)
   └─ Store full result in cache (TMCACHE, TTL=15min)

8. PAGINATE
   └─ Return first max_results, encode next page token with query_hash + offset
```

### Time Bucketing

Matches SQL store behavior:
```python
bucket_ms = time_interval_seconds * 1000
bucket_start = (timestamp_ms // bucket_ms) * bucket_ms
bucket_label = datetime.fromtimestamp(bucket_start / 1000, tz=timezone.utc).isoformat()
```

Time bucket is the first dimension in the dimension key tuple.

### Accumulator Design

```python
@dataclass
class MetricAccumulator:
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    values: list[float] | None = None  # only allocated if percentile requested

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        if self.values is not None:
            self.values.append(value)

    def finalize(self, aggregations: list[MetricAggregation]) -> dict[str, float]:
        result = {}
        for agg in aggregations:
            match agg.aggregation_type:
                case AggregationType.COUNT:
                    result[str(agg)] = self.count
                case AggregationType.SUM:
                    result[str(agg)] = self.sum
                case AggregationType.AVG:
                    result[str(agg)] = self.sum / self.count if self.count else 0.0
                case AggregationType.MIN:
                    result[str(agg)] = self.min
                case AggregationType.MAX:
                    result[str(agg)] = self.max
                case AggregationType.PERCENTILE:
                    result[str(agg)] = self._percentile(agg.percentile_value)
        return result
```

Accumulators keyed by `tuple` of dimension values, stored in `dict[tuple, MetricAccumulator]`.

**Note on MIN/MAX**: MLflow's `validate_query_trace_metrics_params()` currently does not include MIN/MAX in supported aggregation configs, so they won't reach the accumulator in practice. The accumulator tracks them anyway (O(1) cost) for forward compatibility — if MLflow adds MIN/MAX support later, no accumulator changes needed.

Percentile uses linear interpolation matching `percentile_cont` (PostgreSQL/SQLite behavior).

### META Pre-filter for Span Queries

For SPANS view with span-level filters, the denormalized sets on trace META items (`span_names`, `span_types`, `span_statuses`) serve as a fast pre-filter. If a trace's `span_types` set doesn't contain `"LLM"`, skip fetching its span items entirely. This avoids unnecessary DynamoDB reads.

### Pagination with DynamoDB Cache

- **First call** (no page_token): compute all aggregations → store full result in `TMCACHE#<hash>/RESULT` with 15-min TTL → return first `max_results` data points + page token
- **Subsequent calls** (with page_token): token contains `{"query_hash": "<hash>", "offset": N}` → GetItem from cache → slice at offset → return next page + token (or None if last page)
- **Cache miss on pagination**: recompute aggregations, re-cache, return requested page

## Write Path Changes

### `log_spans` Additions

In addition to the existing span blob write (`T#<trace_id>#SPANS`), `log_spans` now also writes:

**Note**: META denormalization of `span_types`, `span_names`, `span_statuses` currently happens in the read path (`_get_trace_with_spans`). This Phase moves that denormalization into `log_spans` (the write path) so that `query_trace_metrics` can rely on these sets being present on META items without requiring a prior `get_trace` call. The read-path denormalization becomes a no-op if sets are already present.

1. **Individual span items** — for each span:
   - Extract `name`, `type`, `status`, `start_time_ns`, `end_time_ns`
   - Compute `duration_ms = (end_time_ns - start_time_ns) // 1_000_000`
   - Extract `model_name` and `model_provider` from span attributes (if present)
   - Create `T#<trace_id>#SPAN#<span_id>` item

2. **Span metric items** — for each span with `SpanAttributeKey.LLM_COST` attribute:
   - Parse cost JSON string → dict
   - For each cost key/value pair, create `T#<trace_id>#SMETRIC#<span_id>#<key>` item

3. **Trace metric items** — aggregate token usage across all spans:
   - For each span, extract `SpanAttributeKey.CHAT_USAGE` from attributes
   - Sum `input_tokens`, `output_tokens`, `total_tokens` across all spans
   - Create `T#<trace_id>#TMETRIC#<key>` for each non-null token key

All new items written via existing `batch_write()` (boto3 `batch_writer` handles 25-item chunking automatically). TTL propagates from the trace.

**Write amplification**: For a trace with N spans where M have cost attributes, `log_spans` writes: 1 SPANS blob + N span items + up to 3M span metric items + up to 3 trace metric items + META update. Example: 20 spans, 5 with costs → 1 + 20 + 15 + 3 = 39 items (2 batch calls). Individual span items are small (~200 bytes each), so WCU impact is modest.

### `create_assessment` / `update_assessment` Additions

When writing assessment items, denormalize top-level attributes:
- Extract `name`, `assessment_type` from assessment entity
- Parse `numeric_value` from value field (handle float, int, bool true→1.0/false→0.0, string "yes"→1.0/"no"→0.0)
- Store `created_timestamp` from `creation_time`

## File Structure

```
src/mlflow_dynamodbstore/
├── dynamodb/
│   └── schema.py              # Add SK_SPAN_PREFIX, SK_TRACE_METRIC_PREFIX,
│                               # SK_SPAN_METRIC_PREFIX, PK_TMCACHE_PREFIX, SK_TMCACHE_RESULT
├── tracking_store.py           # Add query_trace_metrics orchestrator +
│                               # write path changes in log_spans, create/update_assessment
└── trace_metrics/              # NEW package
    ├── __init__.py
    ├── accumulators.py         # MetricAccumulator + percentile computation
    ├── extractors.py           # View-specific data extraction (traces/spans/assessments)
    ├── filters.py              # Filter parsing + application for trace metrics queries
    └── pagination.py           # DynamoDB cache read/write + page token encode/decode
```

**Why a separate package**: `tracking_store.py` is already ~4400 lines. The aggregation logic is self-contained and benefits from isolation. `tracking_store.query_trace_metrics()` becomes a thin orchestrator that delegates to these modules.

## Testing Strategy

### Unit Tests (moto, direct store)

**Write path:**
- `log_spans` creates individual span items with correct attributes
- `log_spans` creates trace metric items from token usage in span attributes
- `log_spans` creates span metric items from cost attributes
- `log_spans` handles spans without token/cost attributes (no metric items created)
- `create_assessment` denormalizes name, type, numeric_value, created_timestamp onto item
- `update_assessment` updates denormalized attributes correctly
- `log_spans` moves META denormalization (span_types/names/statuses) from read path to write path

**Accumulators:**
- COUNT, SUM, AVG, MIN, MAX produce correct results
- Percentile with linear interpolation matches `percentile_cont` behavior
- Empty accumulator returns sensible defaults
- Multiple dimension groups tracked independently

**Extractors:**
- TRACES view: extracts trace_count (from META), latency (from META execution_duration), token metrics (from TMETRIC items)
- SPANS view: extracts span_count, latency (from duration_ms), costs (from SMETRIC items)
- ASSESSMENTS view: extracts assessment_count, assessment_value (numeric parsing including true/false/yes/no)
- Time bucketing produces correct ISO 8601 UTC strings

**Filters:**
- Trace status filter
- Trace tag filter (exact match)
- Trace metadata filter (exact match)
- Span name/type/status filters (exact match against span items)
- Assessment name/type filters
- META pre-filter skips traces that can't match span filters

**Pagination:**
- Cache miss → compute + store + return first page
- Cache hit → return correct offset slice
- Page token encodes query_hash + offset
- TTL is set correctly (15 min)
- Cache item auto-expires

**Full query_trace_metrics:**
- Each view type × each metric name × each aggregation type
- Time bucketing with dimensions
- Multiple experiment IDs merged
- max_results respected
- Filters reduce result set
- Traces with no spans logged (empty span data)
- Invalid metric names / unsupported aggregation-dimension combos rejected by validator
- Span filters on non-SPANS view type rejected
- time_interval_seconds without start/end raises error

### Integration Tests (moto server, REST)

- Round-trip via MlflowClient for each view type
- Pagination across multiple pages

### E2E Tests (full server)

- Create traces with spans (decorated functions), query metrics via client
- Verify time bucketing returns correct chart data
- Span cost and trace token aggregation

### Coverage

100% patch coverage on new code.

## Out of Scope

- Gateway endpoint management (11 methods) — separate system, not tracking store
- `get_online_trace_details` — Databricks-only
- True async DynamoDB I/O via `aioboto3` — future optimization
- FTS for span names — `query_trace_metrics` uses exact match filters only
