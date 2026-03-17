# Phase 4: Remaining Store Parity — DynamoDB Implementation

## Context

Phases 1–3 implemented datasets, logged models, scorers, core gaps, and advanced traces. Phase 4 closes the remaining gap between the DynamoDB tracking store and the SQLAlchemy store, achieving full `AbstractStore` parity.

**Scope revision from original plan**: The original Phase 4 ("Async/batch") and Phase 5 ("Metric history + remaining") are merged into a single phase. Investigation revealed that async methods (`log_batch_async`, `log_metric_async`, `log_param_async`, `set_tag_async`, `end_async_logging`, `flush_async_logging`, `shut_down_async_logging`) work correctly via inheritance — `AbstractStore.__init__` creates `_async_logging_queue` which wraps the synchronous `log_batch()` in a thread pool. No override needed.

## Scope

5 store methods:

| Method | Category | Priority |
|--------|----------|----------|
| `log_outputs` | Run association | Required — raises `NotImplementedError` |
| `log_spans_async` | Async trace | Required — raises `NotImplementedError` |
| `query_trace_metrics` | Trace analytics | Required — raises `MlflowNotImplementedException` |
| `get_metric_history_bulk_interval_from_steps` | Performance override | Recommended — default loads full history then filters in-memory |
| `get_metric_history_bulk_interval` | Performance override | Recommended — delegates to above |

### Methods Confirmed Working via Inheritance (No Override Needed)

| Method | Mechanism |
|--------|-----------|
| `log_batch_async` | `AbstractStore._async_logging_queue` wraps `log_batch()` in thread pool |
| `log_metric_async` | Delegates to `log_batch_async(metrics=[metric])` |
| `log_param_async` | Delegates to `log_batch_async(params=[param])` |
| `set_tag_async` | Delegates to `log_batch_async(tags=[tag])` |
| `end_async_logging` | Queue lifecycle — no store-specific logic |
| `flush_async_logging` | Queue lifecycle — no store-specific logic |
| `shut_down_async_logging` | Queue lifecycle — no store-specific logic |
| `search_runs` | Delegates to `_search_runs()` (already implemented) |
| `log_metric` | Delegates to `log_batch()` (already implemented) |
| `log_param` | Delegates to `log_batch()` (already implemented) |
| `set_model_versions_tags` | Delegates to `get_logged_model()` + `set_logged_model_tags()` (Phase 1b) |

### Explicitly Out of Scope

| Method | Reason |
|--------|--------|
| `get_online_trace_details` | Databricks-only — no OSS store implements it |
| Gateway endpoint methods (11) | Separate system — not part of tracking store parity |

## DynamoDB Item Design

### Run Output Items (new)

`log_outputs` creates association items linking runs to logged model outputs. These follow the existing run sub-item pattern (`R#<ulid>#...`).

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Run Output | `EXP#<exp_id>` | `R#<run_ulid>#OUTPUT#<output_uuid>` | source_type (`RUN_OUTPUT`), source_id (run_id), destination_type (`MODEL_OUTPUT`), destination_id (model_id), step |

- `output_uuid`: ULID — unique identifier for the association item.
- No GSI/LSI projections needed — outputs are only read when loading a run's full details, which queries `SK begins_with R#<run_ulid>#`.

### No New Items for Other Methods

- `log_spans_async`: Delegates to `log_spans` (Phase 3) — no new items.
- `query_trace_metrics`: Reads existing trace META items — no new items.
- `get_metric_history_bulk_interval*`: Reads existing metric history items (`R#<ulid>#MHIST#...`) — no new items.

## Access Patterns

Every store method maps to Query, GetItem, or UpdateItem — never Scan.

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP1 | Write run output association | `log_outputs` | BatchWriteItem | Table | `PK=EXP#<exp_id>, SK=R#<run_ulid>#OUTPUT#<output_uuid>` per model | O(N) where N = models count |
| AP2 | Async span write | `log_spans_async` | (delegates to `log_spans`) | — | — | Same as Phase 3 AP7 |
| AP3 | Aggregate trace metrics by time | `query_trace_metrics` | Paginated Query | Table + LSI1 | `PK=EXP#<exp_id>`, LSI1 range on timestamp | Single pass over trace META items within time range |
| AP4 | Get metric history for specific steps | `get_metric_history_bulk_interval_from_steps` | Query | Table | `PK=EXP#<exp_id>, SK begins_with R#<run_ulid>#MHIST#<metric_key>#` | One query per run; in-memory step filter (bounded by metric history size) |
| AP5 | Get metric history across runs (sampled) | `get_metric_history_bulk_interval` | (delegates to AP4) | — | — | Collects steps from AP4, samples, then re-queries AP4 per run |

### Scan Risk Assessment

**All 5 access patterns are efficient.** No table scans. AP1 is batch writes with known keys. AP2 delegates to Phase 3. AP3 uses LSI1 for time-range queries within experiment partitions. AP4 queries within a run's metric history partition (bounded by step count). AP5 delegates to AP4.

## Store Methods

### `log_outputs(run_id, models)`

Associates logged model outputs with a run, creating input/output tracking items.

1. Resolve `run_id` → `experiment_id` via `_resolve_run_experiment()`.
2. Verify run is active (not deleted).
3. For each `LoggedModelOutput` in `models`:
   a. Generate `output_uuid` = ULID.
   b. Create item: `PK=EXP#<exp_id>`, `SK=R#<run_ulid>#OUTPUT#<output_uuid>`.
   c. Attributes: `source_type="RUN_OUTPUT"`, `source_id=run_id`, `destination_type="MODEL_OUTPUT"`, `destination_id=model.model_id`, `step=model.step`.
4. Batch write all items.

**Note**: The SQLAlchemy store uses an `SqlInput` table with `source_type="RUN_OUTPUT"` and `destination_type="MODEL_OUTPUT"`. The DynamoDB implementation uses the same field names for compatibility but stores them as run sub-items rather than a separate table.

### `log_spans_async(location, spans)`

Asynchronous version of `log_spans` (Phase 3).

```python
async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
    return self.log_spans(location, spans)
```

This matches the SQLAlchemy store's implementation, which also delegates to the synchronous version. True async DynamoDB support (via `aioboto3`) is a future optimization, not a parity requirement.

### `query_trace_metrics(experiment_ids, view_type, metric_name, aggregations, dimensions, filters, time_interval_seconds, start_time_ms, end_time_ms, max_results, page_token)`

Computes aggregated trace metrics (e.g., average latency over time windows) for the experiment overview dashboard.

1. Validate parameters using MLflow's `validate_query_trace_metrics_params(view_type, metric_name, aggregations, dimensions)`.
2. If `time_interval_seconds` is set: require `start_time_ms` and `end_time_ms`, raise `MlflowException` if missing.
3. For each `experiment_id` in `experiment_ids`:
   a. Query trace META items within time range using LSI1 (`lsi1sk` = timestamp for traces).
      - Key condition: `PK=EXP#<exp_id>`, `SK begins_with T#`, filter for META items only.
      - Time filter: `lsi1sk BETWEEN start_time_ms AND end_time_ms` (if specified).
   b. Collect trace META items with their denormalized attributes (request metadata, tags).
4. Build in-memory trace data structures matching what `query_metrics()` expects.
5. Extract metric values from trace attributes:
   - `view_type=TRACE`: metric comes from trace-level attributes (latency, token count, etc.).
   - Standard trace metrics: `latency` (execution_time_ms), `token_count` (from metadata), etc.
6. Apply `filters` as predicate functions against trace attributes.
7. Group by `dimensions` (e.g., by tag value) and `time_interval_seconds` (time buckets).
8. Compute `aggregations` (COUNT, AVG, MIN, MAX, SUM, P50, P90, P95, P99) per group.
9. Return `PagedList[list[MetricDataPoint]]` with up to `max_results` data points.

**Implementation approach**: Rather than reimplementing the full SQL `query_metrics()` aggregation engine, use a single-pass streaming approach:

```python
# Pseudocode for aggregation
buckets = defaultdict(lambda: defaultdict(list))  # {time_bucket: {dimension_key: [values]}}

for trace_meta in all_trace_metas:
    value = extract_metric(trace_meta, metric_name)
    if value is None:
        continue
    if not passes_filters(trace_meta, filters):
        continue

    time_bucket = compute_bucket(trace_meta.timestamp_ms, time_interval_seconds, start_time_ms)
    dim_key = extract_dimensions(trace_meta, dimensions)
    buckets[time_bucket][dim_key].append(value)

data_points = []
for time_bucket, dim_groups in sorted(buckets.items()):
    for dim_key, values in dim_groups.items():
        data_points.append(compute_aggregations(values, aggregations, time_bucket, dim_key))
```

**Pagination**: Cursor-based, encoding the last time bucket processed. Not strictly needed for initial implementation since trace metric queries are typically bounded by time range.

### `get_metric_history_bulk_interval_from_steps(run_id, metric_key, steps, max_results)`

Performance-optimized override of the `AbstractStore` default, which loads the entire metric history then filters in-memory.

1. Resolve `run_id` → `experiment_id`.
2. Query metric history items: `PK=EXP#<exp_id>`, `SK begins_with R#<run_ulid>#MHIST#<metric_key>#`.
3. Convert `steps` to a set for O(1) lookup.
4. Filter results to only include items where `step` is in the steps set.
5. Sort by `(step, timestamp)`.
6. Apply `max_results` limit.
7. Return `list[MetricWithRunId]`.

**Optimization vs default**: The default implementation calls `get_metric_history()` which loads ALL history items, then filters by step. This override queries the same prefix but applies the step filter during iteration rather than loading into a PagedList first. For runs with thousands of metric history entries but only a few requested steps, this avoids constructing unnecessary `Metric` objects.

**Future optimization**: If metric history items used step in the SK (they do: `#MHIST#<key>#<padded_step>#<timestamp>`), a more aggressive optimization could issue one Query per step with an exact SK prefix. However, the number of requested steps in `get_metric_history_bulk_interval` is bounded by `max_results` (typically 200-1000), and the single-prefix-query + filter approach is simpler and sufficient.

```python
def get_metric_history_bulk_interval_from_steps(
    self, run_id: str, metric_key: str, steps: list[int], max_results: int
) -> list[MetricWithRunId]:
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

### `get_metric_history_bulk_interval(run_ids, metric_key, max_results, start_step, end_step)`

No override needed — the `AbstractStore` default implementation delegates to `get_metric_history_bulk_interval_from_steps()` (which we override above) and `get_metric_history()` (already implemented). The default's step-sampling algorithm works correctly with our overridden `_from_steps` method.

## Schema Constants

New constant to add to `dynamodb/schema.py`:

```python
# Run output association (within EXP# partition, under R# prefix)
SK_OUTPUT_PREFIX = "#OUTPUT#"
```

No new partition keys, GSIs, or LSIs needed.

## Testing Strategy

### Unit Tests (moto, direct store)

**log_outputs:**
- Log single model output to a run — verify item created with correct attributes
- Log multiple model outputs — batch write creates all items
- Log output to deleted run raises error
- Log output to non-existent run raises error
- Verify output items are returned when querying run sub-items (`SK begins_with R#<ulid>#`)

**log_spans_async:**
- Async call delegates to sync `log_spans` — verify spans are written
- Verify return type is `list[Span]`
- Verify spans are retrievable after async write

**query_trace_metrics:**
- COUNT aggregation over traces in time range
- AVG latency across traces with time_interval_seconds bucketing
- Multiple aggregations (COUNT + AVG) in single call
- Dimension grouping (e.g., by tag value)
- Filter application reduces trace count
- Empty experiment returns empty result
- Missing start_time_ms/end_time_ms with time_interval_seconds raises error
- max_results limits output
- Multiple experiment_ids — results aggregated across experiments

**get_metric_history_bulk_interval_from_steps:**
- Returns metrics only for requested steps
- Steps not present in history are silently skipped
- Results sorted by (step, timestamp)
- max_results limits output
- Empty steps list returns empty result

**get_metric_history_bulk_interval:**
- Returns sampled metrics across multiple runs
- start_step/end_step bounds the range
- max_results controls sampling density
- Single run with dense history — sampling reduces output
- Preserves min/max steps per run

### Integration Tests (moto server, REST)

- `MlflowClient` round-trip for `log_outputs`
- Trace metric query via REST endpoint
- Metric history bulk interval via REST endpoint

### E2E Tests (full server)

- Experiment overview dashboard loads trace metrics without errors
- Metric chart bulk loading returns sampled data

### Coverage

100% patch coverage on new code.

## Out of Scope

- Gateway endpoint management (11 methods) — separate system, not tracking store
- `get_online_trace_details` — Databricks-only
- True async DynamoDB I/O via `aioboto3` — future optimization
- Async queue tuning (thread pool size, buffering seconds) — works with MLflow defaults
