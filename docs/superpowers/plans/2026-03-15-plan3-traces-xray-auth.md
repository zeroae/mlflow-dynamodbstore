# Plan 3: Traces, X-Ray & Auth — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trace CRUD with X-Ray span proxy (lazy caching + span attribute indexing), OTel annotation processor, and the full DynamoDB-backed auth plugin with ~50 duck-typed methods.

**Architecture:** Traces stored in DynamoDB (META, tags, request metadata, assessments). Spans live in X-Ray and are lazy-cached to DynamoDB on first `get_trace` call, with span attributes (types, statuses, names, models) denormalized onto trace META and FTS-indexed. Auth uses the USER# partition with typed permission SKs and GSI4 for reverse queries.

**Tech Stack:** Same as Plans 1-2, plus `werkzeug` (for password hashing, already a transitive dep of mlflow/flask).

**Spec:** `docs/superpowers/specs/2026-03-15-mlflow-dynamodbstore-design.md`

**Depends on:** Plans 1 + 2 (foundation, CRUD, search, FTS)

**What Plan 2 provides (used by Plan 3):**
- `dynamodb/search.py`: `FilterPredicate`, `QueryPlan`, `parse_run_filter()`, `plan_run_query()`, `execute_query()` — extend for traces
- `dynamodb/fts.py`: `tokenize_words()`, `tokenize_trigrams()`, `fts_items_for_text()`, `fts_diff()` — use for trace tags, metadata, assessments, span names/content
- `dynamodb/config.py`: `ConfigReader` — use for TTL policy, denormalize patterns, FTS trigram fields
- `dynamodb/pagination.py`: `encode_page_token()`, `decode_page_token()` — use for search_traces
- Tag denormalization pattern: `_denormalize_tag()`, `_remove_denormalized_tag()` on tracking_store — replicate for trace tags
- NAME_REV pattern and GSI5 queries — not needed for traces (trace names not cross-partition searchable)

**Only remaining Plan 1/2 stub:** `link_traces_to_run()` — implement in this plan

---

## File Structure (new/modified files only)

```
src/mlflow_dynamodbstore/
├── tracking_store.py                   # MODIFY: Add trace methods (start_trace, get_trace, search_traces, etc.)
├── dynamodb/
│   └── search.py                       # MODIFY: Add parse_trace_filter(), plan_trace_query()
├── auth/
│   ├── __init__.py                     # NEW
│   ├── store.py                        # NEW: DynamoDBAuthStore (~50 methods)
│   ├── app.py                          # NEW: create_app(Flask) → Flask
│   └── client.py                       # NEW: DynamoDBAuthClient (REST client)
├── xray/
│   ├── __init__.py                     # NEW
│   ├── client.py                       # NEW: X-Ray API wrapper (GetTraceSummaries, BatchGetTraces)
│   ├── filter_translator.py            # NEW: MLflow span.* filter → X-Ray filter expression
│   ├── span_converter.py              # NEW: X-Ray segments → MLflow Span objects
│   └── annotation_config.py            # NEW: Configurable mlflow attr → X-Ray annotation mapping
├── otel/
│   ├── __init__.py                     # NEW
│   └── annotation_processor.py         # NEW: OTel SpanProcessor for X-Ray annotations

tests/
├── unit/
│   ├── test_tracking_traces.py         # NEW: Trace CRUD tests
│   ├── xray/
│   │   ├── test_client.py              # NEW: X-Ray client tests (mocked)
│   │   ├── test_filter_translator.py   # NEW: Filter translation tests
│   │   └── test_span_converter.py      # NEW: Span conversion tests
│   ├── auth/
│   │   ├── test_store.py               # NEW: Auth store tests
│   │   └── test_app.py                 # NEW: Auth app integration
│   └── otel/
│       └── test_annotation_processor.py # NEW: OTel processor tests
├── integration/
│   ├── test_traces.py                  # NEW: Trace lifecycle integration
│   └── test_auth.py                    # NEW: Auth integration
```

---

## Chunk 1: Trace CRUD (DynamoDB-only, no X-Ray)

### Task 1: Trace metadata CRUD

Extend `tracking_store.py` with: `start_trace`, `get_trace_info`, `set_trace_tag`, `delete_trace_tag`, `link_traces_to_run`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tracking_traces.py
import pytest
from mlflow.entities import TraceInfo, TraceStatus

class TestTraceCRUD:
    def test_start_trace(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123",
            experiment_id=exp_id,
            timestamp_ms=1709251200000,
            execution_time_ms=500,
            status=TraceStatus.OK,
            request_metadata={},
            tags={},
        )
        tracking_store.start_trace(trace_info)
        result = tracking_store.get_trace_info("tr-abc123")
        assert result.trace_id == "tr-abc123"
        assert result.experiment_id == exp_id

    def test_trace_has_ttl(self, tracking_store, table):
        """Verify trace META has ttl from CONFIG#TTL_POLICY.trace_retention_days."""
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK, request_metadata={}, tags={},
        )
        tracking_store.start_trace(trace_info)
        meta = table.get_item(f"EXP#{exp_id}", "T#tr-abc123")
        assert "ttl" in meta
        # TTL should be ~30 days from now (default trace_retention_days)

    def test_trace_lsi_attributes(self, tracking_store, table):
        """Verify all LSI attributes set on trace META."""
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK, request_metadata={},
            tags={"mlflow.traceName": "my-trace"},
        )
        tracking_store.start_trace(trace_info)
        meta = table.get_item(f"EXP#{exp_id}", "T#tr-abc123")
        assert meta.get("lsi1sk") == 1709251200000  # timestamp_ms
        assert "lsi3sk" in meta  # status#timestamp_ms
        assert meta.get("lsi4sk") == "my-trace"  # lower(trace_name)
        assert meta.get("lsi5sk") == 500  # execution_time_ms

    def test_trace_gsi1_entry(self, tracking_store, table):
        """Verify GSI1 reverse lookup: TRACE#<id> → EXP#<id>."""
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK, request_metadata={}, tags={},
        )
        tracking_store.start_trace(trace_info)
        # get_trace_info should resolve via GSI1
        result = tracking_store.get_trace_info("tr-abc123")
        assert result.experiment_id == exp_id

    def test_trace_client_request_ptr(self, tracking_store, table):
        """Verify CLIENTPTR materialized item for client_request_id."""
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK,
            request_metadata={"client_request_id": "req-xyz"},
            tags={},
        )
        tracking_store.start_trace(trace_info)
        # CLIENTPTR item should exist with GSI1 attributes
        ptr = table.get_item(f"EXP#{exp_id}", "T#tr-abc123#CLIENTPTR")
        assert ptr is not None
        assert ptr.get("gsi1pk") == "CLIENT#req-xyz"

    def test_set_trace_tag_with_denormalization(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK, request_metadata={}, tags={},
        )
        tracking_store.start_trace(trace_info)
        tracking_store.set_trace_tag("tr-abc123", "mlflow.traceName", "test-trace")
        # Tag item should exist
        tag = table.get_item(f"EXP#{exp_id}", "T#tr-abc123#TAG#mlflow.traceName")
        assert tag is not None
        # Should be denormalized (mlflow.* pattern)
        meta = table.get_item(f"EXP#{exp_id}", "T#tr-abc123")
        assert meta.get("tags", {}).get("mlflow.traceName") == "test-trace"

    def test_delete_trace_tag(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK, request_metadata={}, tags={},
        )
        tracking_store.start_trace(trace_info)
        tracking_store.set_trace_tag("tr-abc123", "mlflow.traceName", "test")
        tracking_store.delete_trace_tag("tr-abc123", "mlflow.traceName")
        tag = table.get_item(f"EXP#{exp_id}", "T#tr-abc123#TAG#mlflow.traceName")
        assert tag is None
        meta = table.get_item(f"EXP#{exp_id}", "T#tr-abc123")
        assert "mlflow.traceName" not in meta.get("tags", {})

    def test_link_traces_to_run(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        trace_info = TraceInfo(
            trace_id="tr-abc123", experiment_id=exp_id,
            timestamp_ms=1709251200000, execution_time_ms=500,
            status=TraceStatus.OK, request_metadata={}, tags={},
        )
        tracking_store.start_trace(trace_info)
        tracking_store.link_traces_to_run(["tr-abc123"], run.info.run_id)
        # Verify trace request metadata has mlflow.sourceRun
        info = tracking_store.get_trace_info("tr-abc123")
        assert info.request_metadata.get("mlflow.sourceRun") == run.info.run_id
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement trace methods**

`start_trace`: Write trace META with:
- TTL from `ConfigReader.get_ttl_policy().trace_retention_days`
- LSI attributes: `lsi1sk=timestamp_ms`, `lsi2sk=end_time_ms`, `lsi3sk=<status>#<timestamp_ms>`, `lsi4sk=lower(trace_name)`, `lsi5sk=execution_time_ms`
- GSI1 entry: `gsi1pk=TRACE#<trace_id>`, `gsi1sk=EXP#<exp_id>`
- CLIENTPTR item if `client_request_id` in request_metadata
- `tags: {}` map for denormalization
- Write request metadata items (`T#<trace_id>#RMETA#<key>`) with same TTL
- Write FTS items for trace tag values (using `fts_items_for_text()` from Plan 2)

`get_trace_info`: Resolve trace_id → experiment_id via GSI1 (with cache), read trace META.

`set_trace_tag` / `delete_trace_tag`: Write/delete tag item + denormalization (using `ConfigReader.should_denormalize()`) + FTS maintenance (using `fts_items_for_text()` / `fts_diff()`). All trace child items inherit the trace's TTL.

`link_traces_to_run`: Write/update request metadata item `T#<trace_id>#RMETA#mlflow.sourceRun` with run_id value.

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add trace metadata CRUD to tracking store"
```

### Task 2: Assessments

Extend with: `create_assessment`, `update_assessment`, `delete_assessment`, `get_assessment`.

- [ ] **Step 1: Write failing tests**

Test create (ULID ID, FTS tokens for value using `fts_items_for_text()`), update (FTS diff using `fts_diff()`), delete (FTS + FTS_REV cleanup via reverse index query), get. Verify assessment items inherit trace TTL.

- [ ] **Step 2: Implement assessment methods**

Assessment items: `T#<trace_id>#ASSESS#<assessment_id>`. ID generated via `generate_ulid()`. FTS items for assessment value text. All items get same TTL as parent trace.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add assessment CRUD with FTS indexing"
```

### Task 3: search_traces (DynamoDB-only)

Add `parse_trace_filter()` and `plan_trace_query()` to `search.py`, then implement `search_traces` in `tracking_store.py`.

- [ ] **Step 1: Write failing tests**

```python
class TestSearchTraces:
    def test_search_by_timestamp(self, tracking_store):
        """LSI1 query."""
        ...

    def test_search_by_status(self, tracking_store):
        """LSI3 composite: status#timestamp_ms."""
        ...

    def test_search_by_execution_time(self, tracking_store):
        """LSI5 duration sort."""
        ...

    def test_search_by_trace_name(self, tracking_store):
        """LSI4 begins_with for prefix ILIKE."""
        ...

    def test_search_by_tag(self, tracking_store):
        """Denormalized tag → FilterExpression."""
        ...

    def test_search_by_metadata(self, tracking_store):
        """Request metadata filter."""
        ...

    def test_search_fts_keyword(self, tracking_store):
        """FTS word-level search on tag/metadata/assessment text."""
        ...

    def test_search_assessment_filter(self, tracking_store):
        """Client-side: query assessments per trace, filter."""
        ...

    def test_search_pagination(self, tracking_store):
        """Pagination with page_token."""
        ...
```

- [ ] **Step 2: Add `parse_trace_filter()` and `plan_trace_query()` to search.py**

Trace search uses:
- LSI1: `timestamp_ms` (default sort)
- LSI2: `end_time_ms`
- LSI3: `<status>#<timestamp_ms>` (status filter + time sort)
- LSI4: `lower(trace_name)` (name prefix ILIKE)
- LSI5: `execution_time_ms` (duration sort)
- FTS: word/trigram search on trace tags, metadata, assessments
- Post-filter: assessment-based filters (`feedback.<key>`), `RLIKE` regex

- [ ] **Step 3: Implement `search_traces` in tracking_store.py using parse → plan → execute pipeline**
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add search_traces with DynamoDB query support"
```

### Task 4: delete_traces

Implement physical trace deletion with full cleanup.

- [ ] **Step 1: Write failing tests**

Test that delete_traces removes: trace META, tags, request metadata, assessments, CLIENTPTR, span cache (if exists), and all FTS/FTS_REV items (via reverse index query `SK begins_with("FTS_REV#T#<trace_id>#")`).

- [ ] **Step 2: Implement using FTS_REV reverse index**

```python
def delete_traces(self, experiment_id, trace_ids):
    for trace_id in trace_ids:
        pk = f"EXP#{experiment_id}"
        # 1. Query all trace sub-items
        trace_items = self._table.query(pk=pk, sk_prefix=f"T#{trace_id}")
        # 2. Query FTS_REV for this trace
        fts_rev_items = self._table.query(pk=pk, sk_prefix=f"FTS_REV#T#{trace_id}#")
        # 3. Derive forward FTS SKs from reverse items
        fts_forward_sks = [derive_forward_sk(item) for item in fts_rev_items]
        # 4. Batch delete everything
        all_items = trace_items + fts_rev_items + fts_forward_items
        self._table.batch_delete(all_items)
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add delete_traces with FTS reverse index cleanup"
```

---

## Chunk 2: X-Ray Integration

### Task 5: X-Ray client wrapper

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/client.py`
- Create: `tests/unit/xray/test_client.py`

- [ ] **Step 1: Write failing tests**

Mock `boto3.client('xray')` to test: `get_trace_summaries(filter_expression, start_time, end_time)`, `batch_get_traces(trace_ids)`. Test time window chunking (6-hour max per query). Test pagination of GetTraceSummaries.

- [ ] **Step 2: Implement XRayClient**

Wrapper around `boto3.client('xray')`. Handles time window chunking, pagination of `GetTraceSummaries`, and `BatchGetTraces` with 5-ID batches.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add X-Ray client wrapper"
```

### Task 6: X-Ray filter translator

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/filter_translator.py`
- Create: `tests/unit/xray/test_filter_translator.py`

- [ ] **Step 1: Write failing tests**

```python
from mlflow_dynamodbstore.xray.filter_translator import translate_span_filters
from mlflow_dynamodbstore.xray.annotation_config import DEFAULT_ANNOTATION_CONFIG

class TestFilterTranslator:
    def test_span_type_equals(self):
        xray_filter, remaining = translate_span_filters(
            [FilterPredicate("span_attribute", "mlflow.spanType", "=", "LLM")],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert 'annotation.mlflow_spanType = "LLM"' in xray_filter

    def test_span_name_equals(self):
        xray_filter, remaining = translate_span_filters(
            [FilterPredicate("span_attribute", "name", "=", "ChatModel")],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert 'annotation.mlflow_spanName = "ChatModel"' in xray_filter

    def test_span_like_not_translatable(self):
        xray_filter, remaining = translate_span_filters(
            [
                FilterPredicate("span_attribute", "mlflow.spanType", "=", "LLM"),
                FilterPredicate("span_attribute", "name", "LIKE", "%chat%"),
            ],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert len(remaining) == 1  # LIKE stays as post-filter
```

- [ ] **Step 2: Implement filter translator**

Takes list of span FilterPredicates + annotation config. Returns (xray_filter_expression, remaining_predicates). Only `=` comparisons on mapped annotations are pushable to X-Ray.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add MLflow span filter → X-Ray filter translator"
```

### Task 7: X-Ray span converter

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/span_converter.py`
- Create: `tests/unit/xray/test_span_converter.py`

- [ ] **Step 1: Write failing tests**

Test conversion of X-Ray segment JSON → MLflow Span objects. Test annotation→attribute mapping per spec table (segment ID → span_id, mlflow_spanType → span_type, etc.). Test parent-child relationships. Test timing conversion (X-Ray seconds → MLflow nanoseconds).

- [ ] **Step 2: Implement span converter**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add X-Ray segment → MLflow Span converter"
```

### Task 8: Annotation config

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/annotation_config.py`
- Create: `tests/unit/xray/test_annotation_config.py`

- [ ] **Step 1: Write failing tests**

Test `DEFAULT_ANNOTATION_CONFIG` maps mlflow.spanType → mlflow_spanType, etc. Test custom mappings. Test `get_xray_annotation_name(mlflow_attr)` lookup.

- [ ] **Step 2: Implement**

Default mapping per spec. Configurable via `[tool.mlflow-dynamodbstore.xray]` in pyproject.toml or env vars.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add configurable X-Ray annotation mapping"
```

### Task 9: get_trace with span proxy + lazy caching + span indexing

Modify `tracking_store.py`: implement `get_trace` to fetch spans from X-Ray, cache to DynamoDB, and index span attributes.

- [ ] **Step 1: Write failing tests**

```python
from unittest.mock import patch, MagicMock

class TestGetTraceWithSpans:
    def test_cache_miss_fetches_from_xray(self, tracking_store):
        """First get_trace → X-Ray fetch → cache write → return spans."""
        # Create trace in DynamoDB
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        # ... start_trace ...
        # Mock XRayClient to return spans
        with patch.object(tracking_store, '_xray_client') as mock_xray:
            mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_SEGMENTS]
            trace = tracking_store.get_trace("tr-abc123")
            assert trace.spans is not None
            mock_xray.batch_get_traces.assert_called_once()

    def test_cache_hit_skips_xray(self, tracking_store):
        """Second get_trace → cache hit → no X-Ray call."""
        # ... start_trace, first get_trace (populates cache) ...
        with patch.object(tracking_store, '_xray_client') as mock_xray:
            trace = tracking_store.get_trace("tr-abc123")
            mock_xray.batch_get_traces.assert_not_called()

    def test_span_attributes_denormalized_on_cache(self, tracking_store, table):
        """On cache, span_types/statuses/names/models sets written to META."""
        # ... start_trace, get_trace ...
        meta = table.get_item(f"EXP#{exp_id}", "T#tr-abc123")
        assert "span_types" in meta  # set of all span types
        assert "span_names" in meta  # set of all span names

    def test_span_name_fts_on_cache(self, tracking_store, table):
        """On cache, FTS items written for span names."""
        # ... start_trace, get_trace ...
        fts_items = table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#W#")
        # Should include tokens from span names
        assert len(fts_items) > 0

    def test_xray_expired_returns_trace_without_spans(self, tracking_store):
        """If X-Ray returns nothing (>30 days), return metadata only."""
        with patch.object(tracking_store, '_xray_client') as mock_xray:
            mock_xray.batch_get_traces.return_value = []
            trace = tracking_store.get_trace("tr-abc123")
            assert trace.spans == [] or trace.spans is None
```

Note: X-Ray calls must be mocked (moto doesn't support X-Ray). Use `unittest.mock.patch`.

- [ ] **Step 2: Implement get_trace with X-Ray proxy**

Flow per spec:
1. Read trace META, tags, metadata, assessments from DynamoDB
2. Check for cached spans: `T#<trace_id>#SPANS`
3. If cached → use them
4. If not cached → call `XRayClient.batch_get_traces(trace_id)`
5. Convert X-Ray segments → MLflow Spans via `span_converter.py`
6. Cache to DynamoDB: `T#<trace_id>#SPANS` (JSON blob, same TTL as trace)
7. Denormalize span attributes on META: `span_types`, `span_statuses`, `span_models`, `span_names` (string sets)
8. Write FTS items for span names and span content (using `fts_items_for_text()`)
9. Return complete Trace

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add get_trace with X-Ray span proxy, lazy caching, and span indexing"
```

### Task 10: search_traces hybrid (DynamoDB + X-Ray)

Modify `search_traces`: for `span.*` filters, query both DynamoDB (cached traces with denormalized span attrs) and X-Ray (uncached), union results.

- [ ] **Step 1: Write failing tests**

Test: span.type filter returns:
- Cached traces via FilterExpression `contains(span_types, 'LLM')` on META
- Uncached traces via X-Ray `GetTraceSummaries` (mocked)
- Results unioned and deduplicated

Test: span.name LIKE '%chat%' on cached traces uses FTS items.

- [ ] **Step 2: Implement hybrid search**

Split span predicates from non-span predicates. For non-span: use existing DynamoDB plan_trace_query. For span predicates on cached traces: add FilterExpressions for denormalized sets or FTS queries. For uncached traces: translate to X-Ray filter, query, union.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add hybrid DynamoDB + X-Ray search for span filters"
```

### Task 11: OTel annotation processor

**Files:**
- Create: `src/mlflow_dynamodbstore/otel/annotation_processor.py`
- Create: `tests/unit/otel/test_annotation_processor.py`

- [ ] **Step 1: Write failing tests**

Test that the SpanProcessor maps `mlflow.spanType` → `mlflow_spanType` annotation on OTel spans before export. Test configurable mapping. Test that unmapped attributes pass through unchanged.

- [ ] **Step 2: Implement AnnotationSpanProcessor(SpanProcessor)**

Intercepts `on_end` to add X-Ray annotations from configurable MLflow attribute mappings. Uses `annotation_config.py` for the mapping table.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add OTel SpanProcessor for X-Ray annotation mapping"
```

---

## Chunk 3: Auth Plugin

### Task 12: DynamoDBAuthStore — user methods

**Files:**
- Create: `src/mlflow_dynamodbstore/auth/__init__.py`
- Create: `src/mlflow_dynamodbstore/auth/store.py`
- Create: `tests/unit/auth/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest

class TestAuthStoreUsers:
    def test_create_user(self, auth_store):
        user = auth_store.create_user("alice", "password123")
        assert user.username == "alice"
        assert user.is_admin is False

    def test_authenticate_user(self, auth_store):
        auth_store.create_user("alice", "password123")
        assert auth_store.authenticate_user("alice", "password123") is True
        assert auth_store.authenticate_user("alice", "wrong") is False

    def test_get_user(self, auth_store):
        auth_store.create_user("alice", "password123")
        user = auth_store.get_user("alice")
        assert user.username == "alice"

    def test_list_users(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.create_user("bob", "pass2")
        users = auth_store.list_users()
        assert len(users) == 2

    def test_update_user_password(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.update_user("alice", password="pass2")
        assert auth_store.authenticate_user("alice", "pass2") is True

    def test_delete_user_cascades_permissions(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        auth_store.delete_user("alice")
        assert auth_store.has_user("alice") is False
        # All permissions should be gone too

    def test_create_duplicate_user_raises(self, auth_store):
        auth_store.create_user("alice", "pass1")
        with pytest.raises(Exception):
            auth_store.create_user("alice", "pass2")
```

- [ ] **Step 2: Implement user CRUD**

`USER#<username>` partition with `U#META` SK. Password hashing via `werkzeug.security.generate_password_hash` / `check_password_hash`. Strongly consistent reads (`ConsistentRead=True`) for `authenticate_user` and `get_user`. `list_users` via GSI2 (`AUTH_USERS`). `delete_user` queries all items in `USER#<username>` partition and batch deletes.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add DynamoDBAuthStore user CRUD"
```

### Task 13: DynamoDBAuthStore — experiment permissions

- [ ] **Step 1: Write failing tests**

Test CRUD: `create_experiment_permission`, `get_experiment_permission`, `list_experiment_permissions(username)`, `update_experiment_permission`, `delete_experiment_permission`. Test GSI4 reverse query: list all users with access to experiment X.

- [ ] **Step 2: Implement**

SK: `U#PERM#EXP#<experiment_id>`. GSI4: `gsi4pk=PERM#EXP#<experiment_id>`, `gsi4sk=USER#<username>`.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add experiment permissions to auth store"
```

### Task 14: DynamoDBAuthStore — model + workspace + scorer permissions

Implement remaining permission types following the same pattern.

- [ ] **Step 1: Write failing tests**

Model permissions: CRUD + `delete_registered_model_permissions(name)` (bulk via GSI4) + `rename_registered_model_permissions(old, new)` (GSI4 query → delete old → write new).
Workspace permissions: CRUD + `list_workspace_permissions(workspace)` (GSI4) + `list_accessible_workspace_names(username)` (query `USER#<username>` SK prefix).
Scorer permissions: CRUD + `delete_scorer_permissions_for_scorer(exp_id, scorer_name)` (GSI4).

- [ ] **Step 2: Implement all permission methods**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add model, workspace, and scorer permissions to auth store"
```

### Task 15: Auth app plugin (create_app)

**Files:**
- Create: `src/mlflow_dynamodbstore/auth/app.py`
- Create: `tests/unit/auth/test_app.py`

- [ ] **Step 1: Write failing tests**

Test `create_app(flask_app)` returns Flask app with DynamoDBAuthStore. Test admin user created from config. Test auth middleware registered.

- [ ] **Step 2: Implement create_app**

1. Parse DynamoDB URI from `MLFLOW_BACKEND_STORE_URI` or config
2. Initialize `DynamoDBAuthStore` with same table
3. Create admin user from `MLFLOW_AUTH_ADMIN_USERNAME`/`MLFLOW_AUTH_ADMIN_PASSWORD` env vars (default: admin/password1234)
4. Replace MLflow's auth module-level `store` singleton with our instance
5. Reuse MLflow's `_before_request`/`_after_request` hooks and permission validators

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add dynamodb-auth Flask app plugin"
```

### Task 16: Auth client

**Files:**
- Create: `src/mlflow_dynamodbstore/auth/client.py`

- [ ] **Step 1: Implement DynamoDBAuthClient**

REST client following MLflow's `AuthServiceClient` pattern. Makes HTTP requests to the MLflow server's auth endpoints.

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: add DynamoDBAuthClient REST client"
```

---

## Chunk 4: Integration Tests + Verification

### Task 17: Trace integration tests

**Files:**
- Create: `tests/integration/test_traces.py`

- [ ] **Step 1: Write integration tests via moto server**

Full trace lifecycle: start_trace → set_trace_tag → create_assessment → update_assessment → search_traces (by status, by timestamp, by FTS keyword) → delete_traces. Verify:
- TTL attributes on all trace items
- FTS + FTS_REV items written and cleaned up
- CLIENTPTR item created and cleaned up
- Tag denormalization on trace META
- Pagination

Note: X-Ray proxy cannot be tested via moto (X-Ray not supported). Test DynamoDB-only paths. X-Ray integration tested via mocks in unit tests.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "test: add trace integration tests"
```

### Task 18: Auth integration tests

**Files:**
- Create: `tests/integration/test_auth.py`

- [ ] **Step 1: Write integration tests via moto server**

Full auth lifecycle: create user → create experiment permission → authenticate → verify access → create model permission → rename model permissions → delete workspace permissions cascade → delete user cascade. Verify GSI4 reverse queries.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "test: add auth integration tests"
```

### Task 19: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --cov=mlflow_dynamodbstore --cov-report=term-missing
```

- [ ] **Step 2: Run linters**

```bash
uv run ruff check src/ tests/
uv run mypy src/mlflow_dynamodbstore/
```

- [ ] **Step 3: Run MLflow compatibility tests**

```bash
uv run pytest tests/compatibility/ -v --tb=short
```

Check if trace-related and auth-related compatibility tests now pass.

- [ ] **Step 4: Verify auth app plugin end-to-end (manual smoke test)**

```bash
uv run mlflow server \
  --app-name dynamodb-auth \
  --backend-store-uri dynamodb://us-east-1/test-table \
  --default-artifact-root /tmp/artifacts \
  --port 5001 &
sleep 3
# Test auth
curl -u admin:password1234 http://localhost:5001/api/2.0/mlflow/experiments/list
kill %1
```

- [ ] **Step 5: Commit any fixes**

```bash
git commit -m "chore: fix lint and type errors from Plan 3 verification"
```
