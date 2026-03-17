# Phase 1b: Logged Models — DynamoDB Implementation

## Context

The MLflow 3.x UI calls `search_logged_models` on every experiment overview page load. The DynamoDB tracking store does not implement this method, causing 500 errors. This spec covers the second sub-phase: Logged Models.

Logged Models are experiment-scoped entities that represent ML models created during training or evaluation. They have a lifecycle (PENDING → READY/FAILED), carry tags, params, and metrics, and support search with complex filter expressions across attributes, metrics, params, and tags.

## Scope

10 store methods:

| Method | Category |
|--------|----------|
| `create_logged_model` | Lifecycle |
| `get_logged_model` | Lifecycle |
| `finalize_logged_model` | Lifecycle |
| `delete_logged_model` | Lifecycle (soft delete + TTL) |
| `search_logged_models` | Search |
| `set_logged_model_tags` | Tags |
| `delete_logged_model_tag` | Tags |
| `log_logged_model_params` | Params |
| `record_logged_model` | Run association |
| `set_model_versions_tags` | Registry link |

## DynamoDB Item Design

Logged Models are experiment-scoped (each belongs to exactly one experiment), so they live under the existing `EXP#<exp_id>` partition — same pattern as runs and traces.

### Item Types

| Item | PK | SK | lsi1sk | lsi2sk | lsi3sk | lsi4sk | Attributes |
|------|----|----|--------|--------|--------|--------|------------|
| Model META | `EXP#<exp_id>` | `LM#<model_id>` | `active#<model_id>` or `deleted#<model_id>` | `<creation_timestamp_ms>` | `<status>#<model_id>` | `lower(name)` | name, artifact_location, creation_timestamp_ms, last_updated_timestamp_ms, status, lifecycle_stage, model_type, source_run_id, status_message, tags (denormalized dict), params (denormalized dict), workspace |
| Model Tag | `EXP#<exp_id>` | `LM#<model_id>#TAG#<key>` | — | — | — | — | key, value |
| Model Param | `EXP#<exp_id>` | `LM#<model_id>#PARAM#<key>` | — | — | — | — | key, value |
| Model Metric | `EXP#<exp_id>` | `LM#<model_id>#METRIC#<metric_name>#<run_id>` | — | — | — | — | metric_name, metric_value (Decimal), metric_timestamp_ms, metric_step, run_id, dataset_name, dataset_digest |
| RANK (global metric) | `EXP#<exp_id>` | `RANK#lm#<metric_name>#<inverted_value>#<model_id>` | — | — | — | — | model_id, metric_name, metric_value (Decimal) |
| RANK (dataset-scoped) | `EXP#<exp_id>` | `RANK#lmd#<metric_name>#<dataset_name>#<dataset_digest>#<inverted_value>#<model_id>` | — | — | — | — | model_id, metric_name, metric_value (Decimal), dataset_name, dataset_digest |

### ID Generation

- `model_id`: `m-<ulid>` — matches SQLAlchemy store's `m-{uuid}` format but uses ULID for time-sortability.

### GSI Usage

| GSI | PK | SK | Purpose |
|-----|----|----|---------|
| GSI1 | `LM#<model_id>` | `EXP#<exp_id>` | Reverse lookup: find experiment for a model_id. Written on META items. |

No new GSI prefixes needed for listing — logged models are queried within an experiment using the existing partition + SK prefix pattern, consistent with how runs and traces work.

### LSI Usage

LSIs are overloaded within the `EXP#<exp_id>` partition, following the existing convention:

| LSI | Model META usage | Purpose |
|-----|-----------------|---------|
| LSI1 | `active#<model_id>` or `deleted#<model_id>` | Filter by lifecycle_stage |
| LSI2 | `<creation_timestamp_ms>` | Sort by creation time |
| LSI3 | `<status>#<model_id>` | Filter by status (PENDING, READY, FAILED) |
| LSI4 | `lower(name)` | Sort/filter by name |

Tag, Param, Metric, and RANK items do not populate LSI attributes.

### Artifact Location

Computed on create: `<experiment.artifact_location>/models/<model_id>/artifacts/` — matching the SQLAlchemy store convention.

## Access Patterns

Every store method maps to Query or GetItem — never Scan.

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Filter | Notes |
|---|---------------|--------|-----------|-------------|---------------|--------|-------|
| AP1 | Get model by ID (known experiment) | `get_logged_model` | GetItem | Table | `PK=EXP#<exp_id>, SK=LM#<model_id>` | — | O(1) |
| AP2 | Get model by ID (unknown experiment) | `get_logged_model` | Query | GSI1 | `PK=LM#<model_id>` | — | Returns experiment_id, then AP1 |
| AP3 | Get tags for model | `get_logged_model` | Query | Table | `PK=EXP#<exp_id>, SK begins_with LM#<model_id>#TAG#` | — | Bounded by tag count |
| AP4 | Get params for model | `get_logged_model` | Query | Table | `PK=EXP#<exp_id>, SK begins_with LM#<model_id>#PARAM#` | — | Bounded by param count |
| AP5 | Get metrics for model | `get_logged_model` | Query | Table | `PK=EXP#<exp_id>, SK begins_with LM#<model_id>#METRIC#` | — | Bounded by metric count |
| AP6 | Search models in experiments | `search_logged_models` | Query | Table | `PK=EXP#<exp_id>, SK begins_with LM#` | `lifecycle_stage != deleted` | One query per experiment_id; merge results |
| AP7 | Search models — sort by creation time | `search_logged_models` | Query | LSI2 | `PK=EXP#<exp_id>` | `SK begins_with LM#` | Default sort order |
| AP8 | Search models — filter by status | `search_logged_models` | Query | LSI3 | `PK=EXP#<exp_id>, lsi3sk begins_with <status>#` | `SK begins_with LM#` | Status-scoped |
| AP9 | Search models — sort by name | `search_logged_models` | Query | LSI4 | `PK=EXP#<exp_id>` | `SK begins_with LM#` | Name-sorted |
| AP10 | Soft delete model | `delete_logged_model` | UpdateItem + BatchWriteItem | Table | `PK=EXP#<exp_id>, SK=LM#<model_id>` | — | Set lifecycle_stage=deleted + TTL on META, children, and RANK items |
| AP11 | Finalize model | `finalize_logged_model` | UpdateItem | Table | `PK=EXP#<exp_id>, SK=LM#<model_id>` | — | Set status, update LSI3 |
| AP12 | ORDER BY metric DESC (global) | `search_logged_models` | Query | Table | `PK=EXP#<exp_id>, SK begins_with RANK#lm#<metric_name>#` | — | ScanIndexForward=True (inverted values → descending); collect model_ids, then batch-get META |
| AP13 | ORDER BY metric DESC (dataset-scoped) | `search_logged_models` | Query | Table | `PK=EXP#<exp_id>, SK begins_with RANK#lmd#<metric_name>#<dataset_name>#<dataset_digest>#` | — | Same inverted-value trick; scoped to dataset |
| AP14 | Filter metric > threshold | `search_logged_models` | Query | Table | `PK=EXP#<exp_id>, SK between RANK#lm#<metric>#<inv(threshold)> and RANK#lm#<metric>#<inv(min)>` | — | Range query on inverted values |
| AP15 | Filter metric > threshold (dataset-scoped) | `search_logged_models` | Query | Table | `PK=EXP#<exp_id>, SK between RANK#lmd#<metric>#<ds_name>#<ds_digest>#<inv(threshold)> and ...` | — | Range query scoped to dataset |

### Filter String Strategy

`search_logged_models` supports filter expressions like `name = 'foo'`, `metrics.accuracy > 0.9`, `params.lr = '0.01'`, `tags.env = 'prod'`.

**Attribute filters** (name, status, model_type, source_run_id): applied in-memory against denormalized META fields. For status, can use LSI3 to pre-filter.

**Metrics filters**: use RANK items for DynamoDB-native range queries. When `filter_string` contains `metrics.accuracy > 0.5`, query `RANK#lm#accuracy#` with SK range condition (AP14). With dataset scope via the `datasets` parameter, query `RANK#lmd#accuracy#<dataset_name>#<dataset_digest>#` (AP15). The RANK query returns model_ids directly — no need to load metric sub-items per model.

**Params/Tags filters**: require loading sub-items per candidate model, applied in-memory. This matches the existing `search_traces` approach for span-level filters.

### RANK Items for Logged Model Metrics

RANK items are materialized sort/filter indexes for metric values, following the same pattern as run metric RANK items in the original design spec.

**Two RANK item types:**

| Type | SK Pattern | Purpose |
|------|-----------|---------|
| Global | `RANK#lm#<metric_name>#<inverted_value>#<model_id>` | ORDER BY / filter on metric without dataset scope |
| Dataset-scoped | `RANK#lmd#<metric_name>#<dataset_name>#<dataset_digest>#<inverted_value>#<model_id>` | ORDER BY / filter on metric scoped to a specific dataset |

**Value inversion**: Metric values are inverted (subtracted from a large constant) and zero-padded so that lexicographic ascending sort in DynamoDB yields descending numeric order. This is the same technique used for run RANK items.

**Write path**: When a metric is logged on a logged model, write:
1. The metric sub-item (`LM#<model_id>#METRIC#<metric_name>#<run_id>`).
2. A global RANK item (`RANK#lm#<metric_name>#<inv_value>#<model_id>`).
3. If the metric has dataset_name + dataset_digest: a dataset-scoped RANK item (`RANK#lmd#...`).

Only the **latest value** per (metric_name, model_id) is kept in RANK items — the previous RANK item is deleted and replaced on each update. For dataset-scoped RANK items, the key is (metric_name, dataset_name, dataset_digest, model_id).

**Soft delete**: RANK items receive TTL along with all other children on `delete_logged_model`. They are excluded from search results by cross-referencing the model's lifecycle_stage against META items.

**DynamoDB-native queries enabled:**

| Query | DynamoDB Operation | Access Pattern |
|-------|-------------------|----------------|
| `ORDER BY metrics.accuracy DESC` | `SK begins_with RANK#lm#accuracy#`, ScanIndexForward=True | AP12 |
| `ORDER BY metrics.accuracy ASC` | `SK begins_with RANK#lm#accuracy#`, ScanIndexForward=False | AP12 |
| `metrics.accuracy > 0.5` | `SK between RANK#lm#accuracy#<inv(0.5)>... and RANK#lm#accuracy#<inv(max)>...` | AP14 |
| Dataset-scoped ORDER BY | `SK begins_with RANK#lmd#accuracy#eval_set#abc123#` | AP13 |
| Dataset-scoped filter | Same range query with `RANK#lmd#` prefix | AP15 |

### Scan Risk Assessment

**All 15 access patterns are efficient.** No table scans. Every operation uses GetItem, Query with precise key conditions, or UpdateItem. RANK items enable DynamoDB-native metric sorting and filtering without loading sub-items per model. Search operations query within experiment partitions (bounded).

## Store Methods

### Lifecycle

**`create_logged_model(experiment_id, name, source_run_id, tags, params, model_type)`**
1. Verify experiment exists.
2. Generate `m-<ulid>` model_id.
3. Compute artifact_location: `<experiment.artifact_location>/models/<model_id>/artifacts/`.
4. Put META item with status=PENDING, lifecycle_stage=active, GSI1 and LSI projections.
5. Write tag items if provided + denormalize to META.
6. Write param items if provided + denormalize to META.
7. Return `LoggedModel` entity.

**`get_logged_model(model_id, allow_deleted=False)`**
1. Resolve experiment_id via GSI1 (`LM#<model_id>`).
2. GetItem META from table.
3. If not found or (lifecycle_stage=deleted and not allow_deleted): raise RESOURCE_DOES_NOT_EXIST.
4. Query tags, params, metrics sub-items.
5. Return `LoggedModel` entity.

**`finalize_logged_model(model_id, status)`**
1. Resolve experiment_id via GSI1.
2. UpdateItem on META: set status (READY or FAILED), update last_updated_timestamp_ms, update LSI3.
3. Return updated `LoggedModel` entity.

**`delete_logged_model(model_id)`**
1. Resolve experiment_id via GSI1.
2. Compute `ttl = now + soft_deleted_retention_days`.
3. UpdateItem on META: set lifecycle_stage=deleted, ttl=\<ttl\>, update LSI1 to `deleted#<model_id>`, update last_updated_timestamp_ms.
4. Query all child items (`SK begins_with LM#<model_id>#`) — tags, params, metrics.
5. Query RANK items (`SK begins_with RANK#lm#` and `RANK#lmd#`) that contain this model_id — extract from SK suffix.
6. BatchWriteItem: set ttl=\<ttl\> on each child and RANK item.
6. Soft-deleted models are excluded from `search_logged_models` by LSI1 lifecycle filter. After retention expires, DynamoDB TTL auto-deletes all items.

This matches the existing `delete_run` pattern exactly.

### Search

**`search_logged_models(experiment_ids, filter_string, datasets, max_results, order_by, page_token)`**
1. Parse filter_string into attribute/metric/param/tag predicates. Parse order_by.
2. **If order_by is a metric** (e.g., `metrics.accuracy`):
   a. For each experiment_id: query RANK items (`SK begins_with RANK#lm#<metric>#` or `RANK#lmd#<metric>#<ds_name>#<ds_digest>#`). This returns model_ids pre-sorted by metric value.
   b. Batch-get META items for the returned model_ids.
   c. Exclude soft-deleted models (lifecycle_stage=deleted).
3. **Otherwise** (order_by is an attribute like creation_time, name):
   a. For each experiment_id: query META items using appropriate LSI (LSI2 for creation_time, LSI4 for name), filtering `SK begins_with LM#` to exclude non-model items.
   b. Exclude soft-deleted models.
4. **If filter includes metric predicates** (e.g., `metrics.accuracy > 0.5`):
   a. Query RANK items with range condition (AP14/AP15) to get matching model_ids.
   b. Intersect with candidate set from step 2 or 3.
5. For attribute predicates: apply in-memory against META fields.
6. For param/tag predicates: lazy-load sub-items per candidate model, apply in-memory.
7. Apply pagination: cursor-based token encoding model_id of last result.
8. Return `PagedList[LoggedModel]`.

### Tags & Params

**`set_logged_model_tags(model_id, tags)`**
1. Resolve experiment_id via GSI1.
2. For each tag: put item `SK=LM#<model_id>#TAG#<key>` (upsert).
3. Update denormalized tags dict on META, update last_updated_timestamp_ms.

**`delete_logged_model_tag(model_id, key)`**
1. Resolve experiment_id via GSI1.
2. Delete item `SK=LM#<model_id>#TAG#<key>`.
3. Remove key from denormalized tags dict on META, update last_updated_timestamp_ms.

**`log_logged_model_params(model_id, params)`**
1. Resolve experiment_id via GSI1.
2. For each param: put item `SK=LM#<model_id>#PARAM#<key>`.
3. Update denormalized params dict on META, update last_updated_timestamp_ms.

### Run Association

**`record_logged_model(run_id, mlflow_model)`**
1. Get run to find experiment_id.
2. Read the run's `mlflow.loggedModels` tag (JSON array).
3. Append model info dict to the array.
4. Write back the updated tag value.

**`set_model_versions_tags(name, version, model_id)`**
1. Resolve experiment_id via GSI1.
2. Set tag `SK=LM#<model_id>#TAG#mlflow.registeredModelName` with value = name.
3. Set tag `SK=LM#<model_id>#TAG#mlflow.registeredModelVersion` with value = version.
4. Update denormalized tags dict on META.

## Schema Constants

New constants to add to `dynamodb/schema.py`:

```python
# Logged Model sort keys (within EXP# partition)
SK_LM_PREFIX = "LM#"

# RANK prefixes for logged model metrics
SK_RANK_LM_PREFIX = "RANK#lm#"
SK_RANK_LMD_PREFIX = "RANK#lmd#"

# GSI1 prefix for logged model reverse lookup
GSI1_LM_PREFIX = "LM#"
```

No new partition key prefixes needed — logged models reuse the existing `EXP#<exp_id>` partition family.

## Testing Strategy

### Unit Tests (moto, direct store)

- **Lifecycle**: create → get → finalize (READY) → get; create → finalize (FAILED)
- **Soft delete with TTL**: create → delete → get raises; get with allow_deleted=True succeeds; verify TTL set on META and all children
- **Tags**: set, overwrite, delete, verify denormalized dict on META stays in sync
- **Params**: log params on create, log additional params after create
- **Search**: across multiple experiments, filter by name/status/model_type, filter by tag/param, order by creation_time (default), order by name
- **Metrics filter via RANK**: search with `metrics.accuracy > 0.5` uses RANK item range query, not in-memory filtering
- **Metrics order via RANK**: ORDER BY `metrics.accuracy DESC` uses RANK item prefix query
- **Dataset-scoped metrics**: filter and order by metric with dataset_name + dataset_digest uses `RANK#lmd#` items
- **RANK lifecycle**: verify RANK items receive TTL on soft delete
- **Pagination**: search with max_results and page_token
- **Artifact location**: verify computed path matches `<exp_artifact_loc>/models/<model_id>/artifacts/`
- **Status state machine**: PENDING → READY works, PENDING → FAILED works
- **record_logged_model**: appends to run's `mlflow.loggedModels` tag
- **set_model_versions_tags**: sets registry name/version tags
- **Edge cases**: get non-existent model raises, finalize non-existent raises, delete idempotent

### Integration Tests (moto server, REST)

- `MlflowClient` round-trip: create → get → finalize → search → delete
- Tag and param operations via client
- Search with filter_string via client

### E2E Tests (full server)

- Logged model CRUD via HTTP endpoints
- Search logged models returns (currently 500s → should return empty list)
- Verify experiment overview page loads without 500s

### Coverage

100% patch coverage on new code.

## Out of Scope

- Artifact upload/download (handled by artifact repository layer, not tracking store)
- `ListLoggedModelArtifacts` handler (artifact repo concern)
- Logged model metrics ingestion pipeline (metrics are written by evaluation runs, not by the store methods in scope)
