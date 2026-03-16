# Plan 3: Traces, X-Ray & Auth — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trace CRUD with X-Ray span proxy (lazy caching + span attribute indexing), OTel annotation processor, and the full DynamoDB-backed auth plugin with ~50 duck-typed methods.

**Architecture:** Traces stored in DynamoDB (META, tags, request metadata, assessments). Spans live in X-Ray and are lazy-cached to DynamoDB on first `get_trace` call, with span attributes (types, statuses, names, models) denormalized onto trace META and FTS-indexed. Auth uses the USER# partition with typed permission SKs and GSI4 for reverse queries.

**Tech Stack:** Same as Plans 1-2, plus `werkzeug` (for password hashing, already a transitive dep of mlflow/flask).

**Spec:** `docs/superpowers/specs/2026-03-15-mlflow-dynamodbstore-design.md`

**Depends on:** Plans 1 + 2 (foundation, CRUD, search, FTS)

---

## File Structure (new/modified files only)

```
src/mlflow_dynamodbstore/
├── tracking_store.py                   # MODIFY: Add trace methods (start_trace, get_trace, search_traces, etc.)
├── auth/
│   ├── __init__.py                     # NEW
│   ├── store.py                        # NEW: DynamoDBAuthStore (~50 methods)
│   ├── app.py                          # NEW: create_app(Flask) → Flask
│   └── client.py                       # NEW: DynamoDBAuthClient (REST client)
├── xray/
│   ├── __init__.py                     # NEW
│   ├── client.py                       # NEW: X-Ray API wrapper (GetTraceSummaries, BatchGetTraces)
│   ├── filter_translator.py            # NEW: MLflow span.* filter → X-Ray filter expression
│   ├── span_converter.py               # NEW: X-Ray segments → MLflow Span objects
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

Extend `tracking_store.py` with: `start_trace`, `get_trace_info`, `set_trace_tag`, `delete_trace_tag`.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tracking_traces.py
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

    def test_trace_has_ttl(self, tracking_store):
        # Verify trace META item has ttl attribute set from trace_retention_days config
        ...

    def test_set_trace_tag(self, tracking_store):
        # Create trace, set tag, verify tag exists + denormalization + FTS
        ...

    def test_delete_trace_tag(self, tracking_store):
        # Create trace, set tag, delete tag, verify cleanup (tag item + denorm + FTS)
        ...

    def test_trace_client_request_ptr(self, tracking_store):
        # Verify CLIENTPTR materialized item + GSI1 entry
        ...
```

- [ ] **Step 2: Implement trace methods**

`start_trace`: Write trace META with TTL (from `CONFIG#TTL_POLICY`), LSI attributes (`timestamp_ms`, `end_time_ms`, `status#timestamp_ms`, `lower(trace_name)`, `execution_time_ms`), GSI1 entry (`TRACE#<trace_id>` → `EXP#<exp_id>`). Write CLIENTPTR item if `client_request_id` exists.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add trace metadata CRUD to tracking store"
```

### Task 2: Assessments

Extend with: `create_assessment`, `update_assessment`, `delete_assessment`, `get_assessment`.

- [ ] **Step 1: Write failing tests**

Test create (ULID ID, FTS tokens for value), update (FTS diff), delete (FTS + FTS_REV cleanup), get.

- [ ] **Step 2: Implement assessment methods**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add assessment CRUD with FTS indexing"
```

### Task 3: Trace request metadata

Extend with: trace request metadata write/read during `start_trace` and `end_trace`.

- [ ] **Step 1: Write failing tests**

Test that request metadata items (`T#<trace_id>#RMETA#<key>`) are written, readable, and FTS-indexed.

- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add trace request metadata with FTS"
```

### Task 4: search_traces (DynamoDB-only)

Extend with: `search_traces` using the query planner from Plan 2.

- [ ] **Step 1: Write failing tests**

Test search by timestamp (LSI1), status (LSI3), execution_time (LSI5), trace name (LSI4), tag/metadata filters, FTS keyword search, assessment filters (client-side).

- [ ] **Step 2: Implement search_traces using plan_query**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add search_traces with DynamoDB query support"
```

### Task 5: delete_traces

Implement physical trace deletion with full cleanup (trace items + FTS + FTS_REV via reverse index).

- [ ] **Step 1: Write failing tests**

Test that delete_traces removes trace META, tags, metadata, assessments, CLIENTPTR, and all FTS/FTS_REV items.

- [ ] **Step 2: Implement using FTS_REV reverse index**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add delete_traces with FTS reverse index cleanup"
```

---

## Chunk 2: X-Ray Integration

### Task 6: X-Ray client wrapper

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/client.py`
- Create: `tests/unit/xray/test_client.py`

- [ ] **Step 1: Write failing tests**

Mock `boto3.client('xray')` to test: `get_trace_summaries(filter_expression, start_time, end_time)`, `batch_get_traces(trace_ids)`. Test time window chunking (6-hour max per query).

- [ ] **Step 2: Implement XRayClient**

Wrapper around `boto3.client('xray')`. Handles time window chunking, pagination of `GetTraceSummaries`, and `BatchGetTraces` with 5-ID batches.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add X-Ray client wrapper"
```

### Task 7: X-Ray filter translator

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/filter_translator.py`
- Create: `tests/unit/xray/test_filter_translator.py`

- [ ] **Step 1: Write failing tests**

```python
class TestFilterTranslator:
    def test_span_type_equals(self):
        xray_filter = translate_span_filter("span.type = 'LLM'", annotation_config)
        assert xray_filter == 'annotation.mlflow_spanType = "LLM"'

    def test_span_name_equals(self):
        xray_filter = translate_span_filter("span.name = 'ChatModel'", annotation_config)
        assert xray_filter == 'annotation.mlflow_spanName = "ChatModel"'

    def test_span_like_not_translatable(self):
        # LIKE can't be pushed to X-Ray annotations
        xray_filter, remaining = translate_span_filters(
            ["span.type = 'LLM'", "span.name LIKE '%chat%'"], annotation_config
        )
        assert xray_filter == 'annotation.mlflow_spanType = "LLM"'
        assert len(remaining) == 1  # LIKE stays as post-filter
```

- [ ] **Step 2: Implement filter translator**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add MLflow span filter → X-Ray filter translator"
```

### Task 8: X-Ray span converter

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/span_converter.py`
- Create: `tests/unit/xray/test_span_converter.py`

- [ ] **Step 1: Write failing tests**

Test conversion of X-Ray segment JSON → MLflow Span objects. Test annotation→attribute mapping, parent-child relationship, timing conversion.

- [ ] **Step 2: Implement span converter**

Map X-Ray fields to MLflow span fields per the spec's conversion table.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add X-Ray segment → MLflow Span converter"
```

### Task 9: Annotation config

**Files:**
- Create: `src/mlflow_dynamodbstore/xray/annotation_config.py`
- Create: `tests/unit/xray/test_annotation_config.py`

- [ ] **Step 1: Write failing tests**

Test default annotation mapping, custom mapping from config, lookup by MLflow attribute name.

- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add configurable X-Ray annotation mapping"
```

### Task 10: get_trace with span proxy + lazy caching + indexing

Modify `tracking_store.py`: update `get_trace` to fetch spans from X-Ray, cache to DynamoDB, and index span attributes.

- [ ] **Step 1: Write failing tests**

Test the full flow: get_trace → cache miss → X-Ray fetch → cache write (`T#<trace_id>#SPANS`) → span attribute denormalization on META (span_types, span_statuses, span_models, span_names) → FTS items for span names/content → subsequent get_trace uses cache.

Note: X-Ray calls must be mocked (moto doesn't mock X-Ray). Use `unittest.mock.patch` for `XRayClient`.

- [ ] **Step 2: Implement get_trace with X-Ray proxy**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add get_trace with X-Ray span proxy, lazy caching, and span indexing"
```

### Task 11: search_traces hybrid (DynamoDB + X-Ray)

Modify `search_traces`: for `span.*` filters, query both DynamoDB (cached traces) and X-Ray (uncached), union results.

- [ ] **Step 1: Write failing tests**

Test: span.type filter returns cached traces via FilterExpression AND uncached traces via X-Ray, unioned and deduplicated.

- [ ] **Step 2: Implement hybrid search**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add hybrid DynamoDB + X-Ray search for span filters"
```

### Task 12: OTel annotation processor

**Files:**
- Create: `src/mlflow_dynamodbstore/otel/annotation_processor.py`
- Create: `tests/unit/otel/test_annotation_processor.py`

- [ ] **Step 1: Write failing tests**

Test that the SpanProcessor maps `mlflow.spanType` → `mlflow_spanType` annotation on OTel spans before export.

- [ ] **Step 2: Implement AnnotationSpanProcessor(SpanProcessor)**

Intercepts `on_end` to add X-Ray annotations from configurable MLflow attribute mappings.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add OTel SpanProcessor for X-Ray annotation mapping"
```

---

## Chunk 3: Auth Plugin

### Task 13: DynamoDBAuthStore — user methods

**Files:**
- Create: `src/mlflow_dynamodbstore/auth/store.py`
- Create: `tests/unit/auth/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
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

    def test_delete_user(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.delete_user("alice")
        assert auth_store.has_user("alice") is False

    def test_create_duplicate_user_raises(self, auth_store):
        auth_store.create_user("alice", "pass1")
        with pytest.raises(Exception):
            auth_store.create_user("alice", "pass2")
```

- [ ] **Step 2: Implement user CRUD**

`USER#<username>` partition with `U#META` SK. Password hashing via `werkzeug.security.generate_password_hash` / `check_password_hash`. Strongly consistent reads for `authenticate_user` and `get_user`. `list_users` via GSI2 (`AUTH_USERS`). `delete_user` scans and deletes all items in `USER#<username>` partition (META + all permissions).

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add DynamoDBAuthStore user CRUD"
```

### Task 14: DynamoDBAuthStore — experiment permissions

- [ ] **Step 1: Write failing tests**

Test CRUD for experiment permissions: create, get, list (by user), update, delete. Test GSI4 reverse query: "who has access to experiment X?".

- [ ] **Step 2: Implement experiment permission methods**

SK: `U#PERM#EXP#<experiment_id>`. GSI4: `PERM#EXP#<experiment_id>` → `USER#<username>`.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add experiment permissions to auth store"
```

### Task 15: DynamoDBAuthStore — model + workspace + scorer permissions

Implement remaining permission types following the same pattern as Task 14.

- [ ] **Step 1: Write failing tests for each permission type**

Model permissions: CRUD + `delete_registered_model_permissions(name)` (bulk) + `rename_registered_model_permissions(old, new)`.
Workspace permissions: CRUD + `list_workspace_permissions(workspace)` + `list_accessible_workspace_names(username)`.
Scorer permissions: CRUD + `delete_scorer_permissions_for_scorer(exp_id, scorer_name)`.

- [ ] **Step 2: Implement all permission methods**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add model, workspace, and scorer permissions to auth store"
```

### Task 16: Auth app plugin (create_app)

**Files:**
- Create: `src/mlflow_dynamodbstore/auth/app.py`
- Create: `tests/unit/auth/test_app.py`

- [ ] **Step 1: Write failing tests**

Test that `create_app(flask_app)` returns a Flask app with our `DynamoDBAuthStore` initialized. Test that the admin user is created from config. Test that auth middleware is registered.

- [ ] **Step 2: Implement create_app**

Our `create_app` function:
1. Initialize `DynamoDBAuthStore` with table from store URI
2. Create admin user from config (`MLFLOW_AUTH_ADMIN_USERNAME`, `MLFLOW_AUTH_ADMIN_PASSWORD`)
3. Reuse MLflow's auth before/after request hooks, replacing the store singleton
4. Register URL rules for auth endpoints

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add dynamodb-auth Flask app plugin"
```

### Task 17: Auth client

**Files:**
- Create: `src/mlflow_dynamodbstore/auth/client.py`

- [ ] **Step 1: Implement DynamoDBAuthClient**

REST client mirroring `DynamoDBAuthStore` methods over HTTP. Can inherit from or follow MLflow's `AuthServiceClient` pattern.

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: add DynamoDBAuthClient REST client"
```

---

## Chunk 4: Integration Tests + Verification

### Task 18: Trace integration tests

**Files:**
- Create: `tests/integration/test_traces.py`

- [ ] **Step 1: Write integration tests via moto server**

Full trace lifecycle: start_trace → set_trace_tag → create_assessment → search_traces → delete_traces. Verify TTL attributes, FTS items, FTS_REV cleanup, CLIENTPTR items.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "test: add trace integration tests"
```

### Task 19: Auth integration tests

**Files:**
- Create: `tests/integration/test_auth.py`

- [ ] **Step 1: Write integration tests via moto server**

Full auth lifecycle: create user → create experiment permission → authenticate → verify access → rename model permissions → delete user (cascade).

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

```bash
git commit -m "test: add auth integration tests"
```

### Task 20: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --cov=mlflow_dynamodbstore --cov-report=term-missing
```

- [ ] **Step 2: Run linters**

```bash
uv run ruff check src/ tests/
uv run mypy src/mlflow_dynamodbstore/
```

- [ ] **Step 3: Verify auth app plugin works end-to-end**

```bash
uv run mlflow server \
  --app-name dynamodb-auth \
  --backend-store-uri dynamodb://us-east-1/test-table \
  --default-artifact-root /tmp/artifacts \
  --port 5001 &
# Test login, create experiment, verify permissions
curl -u admin:password1234 http://localhost:5001/api/2.0/mlflow/experiments/list
```

- [ ] **Step 4: Commit any fixes**

```bash
git commit -m "chore: fix lint and type errors from Plan 3 verification"
```
