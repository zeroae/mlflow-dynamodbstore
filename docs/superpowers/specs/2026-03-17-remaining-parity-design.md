# Phase 4a: Remaining Store Parity — DynamoDB Implementation

## Context

Phases 1–3 implemented datasets, logged models, scorers, core gaps, and advanced traces. Phase 4a closes the remaining simple gaps. Phase 4b (separate spec) will tackle `query_trace_metrics` — the in-memory trace analytics engine.

**Scope revision from original plan**: The original Phase 4 ("Async/batch") and Phase 5 ("Metric history + remaining") are merged. Investigation revealed that async methods (`log_batch_async`, `log_metric_async`, `log_param_async`, `set_tag_async`, `end_async_logging`, `flush_async_logging`, `shut_down_async_logging`) work correctly via inheritance — `AbstractStore.__init__` creates `_async_logging_queue` which wraps the synchronous `log_batch()` in a thread pool. No override needed.

## Scope

3 store methods:

| Method | Category | Priority |
|--------|----------|----------|
| `log_outputs` | Run association | Required — raises `NotImplementedError` |
| `log_spans_async` | Async trace | Required — raises `NotImplementedError` |
| `get_metric_history_bulk_interval_from_steps` | Performance override | Recommended — default loads full history then filters in-memory |

### Deferred to Phase 4b (separate spec)

| Method | Reason |
|--------|--------|
| `query_trace_metrics` | Complex in-memory analytics engine — requires dedicated spec for trace/span/assessment metric extraction, time bucketing, aggregations (COUNT/AVG/MIN/MAX/SUM/percentiles), dimension grouping, and filter application |

### No Override Needed

| Method | Mechanism |
|--------|-----------|
| `get_metric_history_bulk_interval` | Default delegates to `_from_steps` (overridden above) + `get_metric_history` (already implemented) |

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
| AP3 | Get metric history for specific steps | `get_metric_history_bulk_interval_from_steps` | Query | Table | `PK=EXP#<exp_id>, SK begins_with R#<run_ulid>#MHIST#<metric_key>#` | One query per run; in-memory step filter (bounded by metric history size) |

### Scan Risk Assessment

**All 3 access patterns are efficient.** No table scans. AP1 is batch writes with known keys. AP2 delegates to Phase 3. AP3 queries within a run's metric history partition (bounded by step count).

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

**get_metric_history_bulk_interval_from_steps:**
- Returns metrics only for requested steps
- Steps not present in history are silently skipped
- Results sorted by (step, timestamp)
- max_results limits output
- Empty steps list returns empty result

**get_metric_history_bulk_interval (via inheritance):**
- Returns sampled metrics across multiple runs (exercises our `_from_steps` override)
- start_step/end_step bounds the range
- max_results controls sampling density

### Integration Tests (moto server, REST)

- `MlflowClient` round-trip for `log_outputs`
- Metric history bulk interval via REST endpoint

### E2E Tests (full server)

- Metric chart bulk loading returns sampled data

### Coverage

100% patch coverage on new code.

## Out of Scope

- `query_trace_metrics` — Phase 4b (separate spec for trace analytics engine)
- Gateway endpoint management (11 methods) — separate system, not tracking store
- `get_online_trace_details` — Databricks-only
- True async DynamoDB I/O via `aioboto3` — future optimization
- Async queue tuning (thread pool size, buffering seconds) — works with MLflow defaults
