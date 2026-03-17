# Phase 3: Advanced Traces — DynamoDB Implementation

## Context

Phase 1 fixed UI 500 errors (datasets, logged models, scorers). Phase 2 closed core tracking gaps (`delete_experiment_tag`, `supports_workspaces`). Phase 3 implements the remaining trace-related methods required for full parity with the SQLAlchemy store: bulk reads, session discovery, prompt linking, span logging, and trace filter correlation analysis.

This spec picks up `calculate_trace_filter_correlation` deferred from Phase 1c (scorers).

## Scope

7 store methods + 1 modification to `start_trace`:

| Method | Category | Called By |
|--------|----------|-----------|
| `batch_get_traces` | Bulk read | Evaluation pipelines, `mlflow.search_traces` internals |
| `batch_get_trace_infos` | Bulk read | Online scoring, batch assessment |
| `find_completed_sessions` | Session discovery | Online scoring scheduler |
| `link_prompts_to_trace` | Association | `mlflow.genai.prompts`, demo generator |
| `unlink_traces_from_run` | Association | Run cleanup |
| `log_spans` | Write | Trace export (V3 API) |
| `calculate_trace_filter_correlation` | Analysis | Scorer configuration UI |

**`start_trace` modification**: Upsert a Session Tracker item whenever trace metadata contains `mlflow.traceSession`.

**Not in scope:**
- `get_online_trace_details` — Databricks-only; no OSS store implements it. The existing `NotImplementedError` is correct.
- Async logging methods (`log_batch_async`, `log_metric_async`, etc.) — Phase 4.
- Metric history bulk queries — Phase 5.

## DynamoDB Item Design

### New Item Type

| Item | PK | SK | GSI2 PK | GSI2 SK | Attributes |
|------|----|----|---------|---------|------------|
| Session Tracker | `EXP#<exp_id>` | `SESS#<session_id>` | `SESSIONS#<workspace>#<exp_id>` | `<last_trace_timestamp_ms>` | session_id, trace_count, last_trace_timestamp_ms, first_trace_timestamp_ms |

**Maintenance**: `start_trace` upserts the Session Tracker item whenever trace metadata contains `mlflow.traceSession`. Uses a single `UpdateItem` call with:

```
UpdateExpression:
  ADD trace_count :one
  SET first_trace_timestamp_ms = if_not_exists(first_trace_timestamp_ms, :ts),
      last_trace_timestamp_ms = if_not_exists(last_trace_timestamp_ms, :zero),
      last_trace_timestamp_ms = if_greater(:ts, last_trace_timestamp_ms, :ts, last_trace_timestamp_ms),
      #gsi2pk = :gsi2pk, #gsi2sk = :ts,
      session_id = :sid
```

Because DynamoDB does not support conditional SET within an UpdateExpression, the `last_trace_timestamp_ms` update uses a two-step approach: initialize to zero via `if_not_exists`, then always overwrite with the max. In practice, since traces within a session are typically ingested in chronological order, the simple approach of always setting `last_trace_timestamp_ms = :ts` and `GSI2 SK = :ts` is sufficient. Out-of-order ingestion would result in a slightly inaccurate `last_trace_timestamp_ms`, which is acceptable for session completion discovery (the time window query is approximate by nature).

Simplified actual expression:

```
UpdateExpression:
  ADD trace_count :one
  SET first_trace_timestamp_ms = if_not_exists(first_trace_timestamp_ms, :ts),
      last_trace_timestamp_ms = :ts,
      session_id = if_not_exists(session_id, :sid),
      #gsi2pk = :gsi2pk, #gsi2sk = :ts
```

**Workspace resolution**: The `start_trace` method has access to `self._workspace` (set during store initialization, default `"default"`). This is used for the GSI2 PK value `SESSIONS#<workspace>#<exp_id>`.

TTL: Inherits the trace TTL policy (same `_get_trace_ttl()`).

### Schema Constants

New constants to add to `dynamodb/schema.py`:

```python
# Session tracking (within EXP# partition)
SK_SESSION_PREFIX = "SESS#"

# GSI prefixes for sessions
GSI2_SESSIONS_PREFIX = "SESSIONS#"
```

No new partition key prefixes — sessions reuse the existing `EXP#<exp_id>` partition family. No provisioner changes needed — GSI2 already exists with `ALL` projection.

### Existing Items Used

| Method | Items Read/Written |
|--------|--------------------|
| `batch_get_traces` | Trace META, TAG, RMETA, SPANS (parallel reads) |
| `batch_get_trace_infos` | Trace META, TAG, RMETA (parallel reads) |
| `find_completed_sessions` | Session Tracker via GSI2 range query |
| `link_prompts_to_trace` | Trace TAG (write `mlflow.promptVersions`) |
| `unlink_traces_from_run` | Trace TAG (delete run-link tags) |
| `log_spans` | SPANS cache item `T#<trace_id>#SPANS` (overwrite) |
| `calculate_trace_filter_correlation` | Delegates to `search_traces` (single pass) |

## Access Patterns

Every store method maps to Query, GetItem, or UpdateItem — never Scan.

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP1 | Resolve trace_id → experiment_id (batch) | `batch_get_traces`, `batch_get_trace_infos` | Query (parallel) | GSI1 | `PK=TRACE#<trace_id>` per ID | One Query per trace_id, parallelized. Reuses existing `_resolve_trace_experiment` pattern. Bounded by input list size. |
| AP2 | Get trace META + TAG + RMETA (batch) | `batch_get_trace_infos` | Query per trace | Table | `PK=EXP#<exp_id>, SK begins_with T#<trace_id>` | Same as existing `get_trace_info`. One query per trace fetches META + all sub-items. |
| AP3 | Get trace with SPANS (batch) | `batch_get_traces` | Query per trace | Table | `PK=EXP#<exp_id>, SK begins_with T#<trace_id>` | Same as AP2 but also includes SPANS cache item in results. |
| AP4 | Find sessions in time window | `find_completed_sessions` | Query | GSI2 | `PK=SESSIONS#<workspace>#<exp_id>`, SK between `<min_ts>` and `<max_ts>` | Range query on GSI2. Returns session tracker items directly. |
| AP5 | Write prompt link tag | `link_prompts_to_trace` | PutItem | Table | `PK=EXP#<exp_id>, SK=T#<trace_id>#TAG#mlflow.promptVersions` | JSON-serialized list of `{name, version}` dicts. Overwrites existing tag. |
| AP6 | Remove run-link tags | `unlink_traces_from_run` | DeleteItem per trace | Table | `PK=EXP#<exp_id>, SK=T#<trace_id>#TAG#mlflow.sourceRun` | Resolve each trace_id via GSI1, then delete the tag item. |
| AP7 | Write/overwrite spans | `log_spans` | PutItem | Table | `PK=EXP#<exp_id>, SK=T#<trace_id>#SPANS` | Single item, overwrite. Spans serialized as JSON. |
| AP8 | Count traces for correlation | `calculate_trace_filter_correlation` | 1× paginated Query | Table + LSIs | `base_filter` key conditions | Single pass. Parse filter1/filter2 into predicate functions, evaluate both per trace, increment 4 counters. 3× fewer reads than naive 3-query approach. |
| AP9 | Upsert session tracker | `start_trace` (modified) | UpdateItem | Table | `PK=EXP#<exp_id>, SK=SESS#<session_id>` | Conditional upsert on every trace with session metadata. |

### Scan Risk Assessment

**All 9 access patterns are efficient.** No table scans. AP1-AP3 use GetItem/Query with known keys. AP4 uses GSI2 range query bounded by time window. AP5-AP7 are single-item writes/deletes. AP8 is a single paginated query (same cost as one `search_traces` call). AP9 is a single UpdateItem.

## Store Methods

### `batch_get_traces(trace_ids, location=None)`

1. Resolve each trace_id → experiment_id via GSI1 (AP1). If `location` is provided, skip resolution and use it as experiment_id.
2. For each (experiment_id, trace_id): query all items with `SK begins_with T#<trace_id>` (AP3) — returns META, TAGs, RMETAs, and SPANS.
3. Reconstruct `TraceInfo` from META/TAG/RMETA items (reuse existing `_build_trace_info`).
4. Deserialize SPANS cache item into `list[Span]`.
5. Return `list[Trace]`.

### `batch_get_trace_infos(trace_ids, location=None)`

1. Same as `batch_get_traces` steps 1-3 (AP1 + AP2), but skip SPANS.
2. Return `list[TraceInfo]`.

### `find_completed_sessions(experiment_id, min_last_trace_timestamp_ms, max_last_trace_timestamp_ms, max_results=None, filter_string=None)`

1. Query GSI2: `PK=SESSIONS#<workspace>#<experiment_id>`, SK between `min_ts` and `max_ts` (AP4). Returns Session Tracker items directly.
2. If `filter_string` provided: for each session returned in step 1, query traces belonging to that session by querying `PK=EXP#<exp_id>` with `SK begins_with T#` on LSI1 (time range) and post-filtering for traces whose `RMETA#mlflow.traceSession` matches the session_id. Apply the parsed filter predicates to these traces. A session qualifies if at least one of its traces matches the filter. This is bounded by `session_count × avg_traces_per_session`, which is small in practice (online scoring configs target specific session patterns).
3. Order by `(last_trace_timestamp_ms ASC, session_id ASC)` — GSI2 SK ordering provides this naturally.
4. Apply `max_results` limit.
5. Return `list[CompletedSession]`.

### `link_prompts_to_trace(trace_id, prompt_versions)`

1. Resolve trace_id → experiment_id via GSI1.
2. Serialize `prompt_versions` to JSON: `[{"name": ..., "version": ...}, ...]`.
3. Write trace tag: `key=mlflow.promptVersions`, `value=<json>` (AP5). Uses existing `_write_trace_tag`.

### `unlink_traces_from_run(trace_ids, run_id)`

1. For each trace_id: resolve experiment_id via GSI1.
2. Delete the `mlflow.sourceRun` tag item (AP6). Uses existing `delete_trace_tag` logic.

### `log_spans(location, spans, tracking_uri=None)`

1. Group spans by trace_id.
2. For each trace_id: resolve experiment_id (from `location` if provided, else GSI1).
3. Serialize spans to JSON.
4. PutItem to `SK=T#<trace_id>#SPANS` (AP7). Overwrites any existing cached spans.
5. Return `list[Span]` — the input spans passed through.

### `calculate_trace_filter_correlation(experiment_ids, filter_string1, filter_string2, base_filter=None)`

1. Parse `filter_string1` and `filter_string2` into predicate functions (extract from existing `search_traces` filter logic).
2. Single paginated pass over traces in `experiment_ids`, applying `base_filter` as the query filter (AP8).
3. For each trace: evaluate both predicates, increment `total_count`, `filter1_count`, `filter2_count`, `joint_count`.
4. Compute NPMI with edge case handling:
   - If `total_count = 0`: return NPMI = 0.0 (no data).
   - If `joint_count = 0`: return NPMI = -1.0 (filters never co-occur; by NPMI convention).
   - If `filter1_count = 0` or `filter2_count = 0`: return NPMI = 0.0 (undefined PMI, treat as no association).
   - If `joint_count = total_count` and `filter1_count = total_count` and `filter2_count = total_count`: return NPMI = 1.0 (perfect correlation).
   - Otherwise: `pmi = log(joint / (f1 * f2 / total))`, `npmi = pmi / -log(joint / total)`.
5. Return `TraceFilterCorrelationResult(npmi, filter1_count, filter2_count, joint_count, total_count)`.

### `start_trace` modification (session tracker upsert)

1. After existing trace creation logic, check if `trace_info.trace_metadata` contains `mlflow.traceSession`.
2. If present: UpdateItem on `SK=SESS#<session_id>` (AP9) with `ADD trace_count :1`, conditional set for timestamps.
3. Apply same TTL as trace items.

## Entity Mapping

| MLflow Entity | DynamoDB Source |
|---------------|----------------|
| `Trace.info` | Existing `_build_trace_info` from META + TAG + RMETA items |
| `Trace.data.spans` | SPANS cache item, JSON deserialized |
| `CompletedSession.session_id` | Session Tracker `session_id` attribute |
| `CompletedSession.last_trace_timestamp_ms` | Session Tracker `last_trace_timestamp_ms` attribute |
| `CompletedSession.trace_count` | Session Tracker `trace_count` attribute |
| `TraceFilterCorrelationResult.npmi` | Computed in-memory |
| `TraceFilterCorrelationResult.filter1_count` | Counter from single-pass scan |
| `TraceFilterCorrelationResult.filter2_count` | Counter from single-pass scan |
| `TraceFilterCorrelationResult.joint_count` | Counter from single-pass scan |
| `TraceFilterCorrelationResult.total_count` | Counter from single-pass scan |

## Testing Strategy

### Unit Tests (moto, direct store)

**batch_get_traces / batch_get_trace_infos:**
- Batch fetch 1 trace, 3 traces, 10 traces — verify all returned
- Fetch with `location` parameter skips GSI1 resolution
- Non-existent trace_id in batch — excluded from results (no error)
- Empty list input — returns empty list
- Trace with spans vs trace without spans (batch_get_traces returns spans, batch_get_trace_infos does not)
- Duplicate trace_ids in input — deduplicated in output (one result per unique trace_id)

**find_completed_sessions:**
- Create traces with session metadata, verify session tracker items are created by `start_trace`
- Session tracker: trace_count increments, timestamps update correctly
- Query sessions in time window — returns only sessions within range
- Sessions outside window excluded
- `max_results` limits output
- `filter_string` filters sessions by trace attributes
- Empty experiment — returns empty list
- Session TTL matches trace TTL

**link_prompts_to_trace:**
- Link single prompt version — tag written with correct JSON
- Link multiple prompt versions — all serialized in tag
- Overwrite existing prompt links — tag replaced
- Non-existent trace raises RESOURCE_DOES_NOT_EXIST

**unlink_traces_from_run:**
- Unlink single trace — `mlflow.sourceRun` tag deleted
- Unlink multiple traces — all tags deleted
- Unlink trace not linked to run — silent (idempotent)

**log_spans:**
- Log spans for a trace — SPANS item created
- Log spans overwrites existing SPANS item
- Multiple traces in one call — each gets its own SPANS item

**calculate_trace_filter_correlation:**
- Known distribution: 10 traces, 5 match filter1, 4 match filter2, 2 match both — verify NPMI calculation
- No traces match either filter — NPMI = 0.0
- `joint_count = 0` (filters never co-occur) — NPMI = -1.0
- `filter1_count = 0` or `filter2_count = 0` — NPMI = 0.0
- All traces match both filters — perfect correlation
- `base_filter` restricts the counting universe
- Empty experiment — returns zero counts

**start_trace session tracker:**
- Trace with session metadata creates/updates SESS# item
- Trace without session metadata does not write SESS# item
- Multiple traces same session — trace_count increments, timestamps update
- Session tracker has TTL

### Integration Tests (moto server, REST)

- `MlflowClient.get_trace()` for batch operations round-trip
- Session discovery via REST endpoint (if exposed)

### E2E Tests (full server)

- Demo generation succeeds with prompt linking (exercises `link_prompts_to_trace`)
- `batch_get_traces` via client after creating multiple traces

### Coverage

100% patch coverage on new code.

## Out of Scope

- `get_online_trace_details` — Databricks-only; no OSS store implements it
- Async logging methods — Phase 4
- Metric history bulk queries — Phase 5
- Session tracker backfill for existing data — separate migration script if needed
