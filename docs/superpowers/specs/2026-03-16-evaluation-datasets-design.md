# Phase 1a: Evaluation Datasets — DynamoDB Implementation

## Context

The MLflow 3.x UI calls `search_datasets`, `search_logged_models`, and `list_scorers` on every experiment overview page load. The DynamoDB tracking store does not implement these methods, causing 500 errors. This spec covers the first sub-phase: Evaluation Datasets.

Evaluation Datasets are a V3 feature (distinct from the legacy V2 "logged datasets" used for run input tracking). They are standalone entities that can be associated with multiple experiments, contain typed records with input/output/expectation triples, and support auto schema inference and profiling.

## Scope

12 store methods + 1 legacy method:

| Method | Category |
|--------|----------|
| `create_dataset` | Lifecycle |
| `get_dataset` | Lifecycle |
| `delete_dataset` | Lifecycle |
| `search_datasets` | Lifecycle |
| `upsert_dataset_records` | Records |
| `_load_dataset_records` | Records |
| `delete_dataset_records` | Records |
| `set_dataset_tags` | Tags |
| `delete_dataset_tag` | Tags |
| `get_dataset_experiment_ids` | Association |
| `add_dataset_to_experiments` | Association |
| `remove_dataset_from_experiments` | Association |
| `_search_datasets` | Legacy V2 |

## DynamoDB Item Design

Evaluation Datasets get their own partition family (`DS#<dataset_id>`) because they span multiple experiments — unlike runs and traces which are scoped to a single experiment.

### Item Types

| Item | PK | SK | Attributes |
|------|----|----|------------|
| Dataset META | `DS#<dataset_id>` | `DS#META` | name, digest, schema (JSON), profile (JSON), tags (denormalized dict), created_time, last_update_time, created_by, last_updated_by, workspace |
| Dataset Tag | `DS#<dataset_id>` | `DS#TAG#<key>` | key, value |
| Dataset Record | `DS#<dataset_id>` | `DS#REC#<record_id>` | inputs (JSON), outputs (JSON), expectations (JSON), tags (JSON), source (JSON), input_hash, created_time, last_update_time, created_by, last_updated_by |
| Experiment Link | `DS#<dataset_id>` | `DS#EXP#<exp_id>` | (minimal association item) |

### ID Generation

- `dataset_id`: `eval_<ulid>` — matches SQLAlchemy store's `eval_` prefix convention.
- `record_id`: `edrec_<ulid>` — matches SQLAlchemy store's `edrec_` prefix convention.

### GSI Usage

GSI1 is shared across entity types. Existing PK prefixes: `RUN#`, `TRACE#`, `CLIENT#`, `DS#`. The new `DS_EXP#` prefix is lexicographically distinct from all of these. No provisioner changes needed — all 5 GSIs already use `ALL` projection.

| GSI | PK | SK | Purpose |
|-----|----|----|---------|
| GSI1 | `DS_EXP#<exp_id>` | `DS#<dataset_id>` | Reverse lookup: find datasets associated with an experiment. Written on Experiment Link items. |
| GSI2 | `DS_LIST#<workspace>` | `<dataset_id>` | List all datasets in a workspace. Written on META items. |
| GSI3 | `DS_NAME#<workspace>#<name>` | `<dataset_id>` | Unique name lookup within workspace. Written on META items. |

## Access Patterns

Every store method must map to Query or GetItem — never Scan. The table below lists each access pattern, the DynamoDB operation used, and which key structure serves it.

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Filter | Notes |
|---|---------------|--------|-----------|-------------|---------------|--------|-------|
| AP1 | Get dataset by ID | `get_dataset` | GetItem | Table | `PK=DS#<id>, SK=DS#META` | — | O(1) |
| AP2 | Get tags for dataset | `get_dataset` | Query | Table | `PK=DS#<id>, SK begins_with DS#TAG#` | — | Bounded by tag count |
| AP3 | Check name uniqueness | `create_dataset` | Query | GSI3 | `PK=DS_NAME#<ws>#<name>` | — | Expect 0 or 1 result |
| AP4 | List all datasets in workspace | `search_datasets` (no experiment filter) | Query | GSI2 | `PK=DS_LIST#<ws>` | — | Returns dataset_ids; paginated |
| AP5 | Find datasets for an experiment | `search_datasets` (with experiment_ids) | Query | GSI1 | `PK=DS_EXP#<exp_id>` | — | One query per experiment_id; merge results |
| AP6 | Get experiment_ids for a dataset | `get_dataset_experiment_ids` | Query | Table | `PK=DS#<id>, SK begins_with DS#EXP#` | — | Bounded by experiment count |
| AP7 | Load records (paginated) | `_load_dataset_records` | Query | Table | `PK=DS#<id>, SK begins_with DS#REC#` | — | DynamoDB-native `Limit` + `LastEvaluatedKey` |
| AP8 | Find record by input hash (dedup) | `upsert_dataset_records` | Query | Table | `PK=DS#<id>, SK begins_with DS#REC#` | `FilterExpression: input_hash = :h` | **Partition scan with filter** — reads all records but filters server-side. See scaling note below. |
| AP9 | Get all items for deletion | `delete_dataset` | Query | Table | `PK=DS#<id>` (no SK condition) | — | Full partition read; bounded by total items in dataset |
| AP10 | Delete specific records | `delete_dataset_records` | BatchDeleteItem | Table | `PK=DS#<id>, SK=DS#REC#<rec_id>` per item | — | O(N) where N = record_ids count |
| AP11 | Legacy: datasets linked to runs | `_search_datasets` | Query | Table | `PK=EXP#<exp_id>, SK begins_with D#` | — | Existing V2 pattern |
| AP12 | Legacy: run-dataset associations | `_search_datasets` | Query | Table | `PK=EXP#<exp_id>, SK begins_with DLINK#` | — | Existing V2 pattern |
| AP13 | Batch get META items | `search_datasets` (after AP5) | BatchGetItem | Table | `PK=DS#<id>, SK=DS#META` per item | — | After collecting IDs from GSI1 |

### Scan Risk Assessment

- **AP8 (record dedup)** is the only pattern that reads more data than needed. It queries the full `DS#REC#` SK range within a single partition and applies a `FilterExpression` server-side on `input_hash`. DynamoDB still reads all record items (consuming read capacity) but returns only matches. This is acceptable for datasets with up to ~10,000 records. Mitigation options for larger datasets:
  - **(a)** Add a denormalized `input_hash_index` map on the META item (`{hash: record_id}`) — limited by DynamoDB's 400KB item size (~5,000 entries).
  - **(b)** Add a dedicated SK pattern `DS#HASH#<input_hash>#<record_id>` as a secondary lookup item written alongside each record. Adds one extra write per record but gives O(1) dedup lookups.
  - **(c)** Use an LSI with `input_hash` as the sort key — requires table recreation (not viable for existing tables).

  **Phase 1 decision**: Accept AP8 as-is with the 10K ceiling. Option (b) is the recommended future upgrade path.

- **All other patterns** are efficient: GetItem (O(1)), Query with precise key conditions, or BatchGetItem with known keys. No table scans.

### Record Deduplication

Each record has an `input_hash` field: SHA256 of the JSON-serialized sorted inputs dict (8-char hex prefix). On upsert, query all records (`SK begins_with DS#REC#`) with a `FilterExpression` on `input_hash` to find existing records with matching inputs. If found, update in place; otherwise insert new.

**Scaling note**: This approach scans all records in the partition with a server-side filter. Acceptable for Phase 1 up to ~10,000 records per dataset. A future phase may add an LSI on `input_hash` or a denormalized hash→record_id map on META if larger datasets are needed.

### Digest Computation

Dataset digest = first 8 chars of SHA256(`name:last_update_time`). Recomputed on every mutation (record upsert/delete, tag change, experiment association change).

## Store Methods

### Lifecycle

**`create_dataset(name, tags, experiment_ids)`**
1. Check uniqueness via GSI3 (`DS_NAME#<workspace>#<name>`).
2. Generate `eval_<ulid>` dataset_id.
3. Compute initial digest from `name:now_ms`.
4. Put META item with GSI2 and GSI3 keys.
5. Write tag items if provided.
6. Write experiment link items + GSI1 projections if experiment_ids provided.
7. Return `EvaluationDataset` entity.

**`get_dataset(dataset_id)`**
1. Get META item by `PK=DS#<dataset_id>, SK=DS#META`.
2. Query tag items (`SK begins_with DS#TAG#`).
3. Return `EvaluationDataset` with lazy-loading callbacks for `records` and `experiment_ids`.

**`delete_dataset(dataset_id)`**
1. Query ALL items with `PK=DS#<dataset_id>`.
2. Batch delete all items (META, tags, records, experiment links). Requires adding a `batch_delete(keys)` method to `DynamoDBTable` (boto3's `batch_writer` supports `delete_item`; the existing `batch_write` only wraps `put_item`).
3. GSI entries are automatically cleaned up since they project from deleted items.

**`search_datasets(experiment_ids, filter_string, max_results, order_by, page_token)`**
1. If `experiment_ids` provided: query GSI1 for each experiment, collect dataset_ids, then batch-get META items.
2. If no `experiment_ids`: query GSI2 (`DS_LIST#<workspace>`) for all datasets.
3. Apply `filter_string` in-memory (support `name LIKE '%pattern%'`).
4. Apply `order_by` in-memory (name, created_time, last_update_time).
5. Return `PagedList[EvaluationDataset]` with cursor token.

### Records

**`upsert_dataset_records(dataset_id, records)`**
1. Verify dataset exists (get META).
2. For each record:
   a. Compute `input_hash` = SHA256(json.dumps(sorted inputs)).
   b. Query existing records, filter on `input_hash` to find duplicate.
   c. If exists: update outputs, expectations, tags, source, last_update_time.
   d. If not: generate `edrec_<ulid>` ID, put new record item.
3. Recompute schema by merging field types from all record inputs/outputs/expectations.
4. Recompute profile: `{num_records: count}`.
5. Update META item with new schema, profile, digest, last_update_time.
6. Return `{inserted: N, updated: M}`.

**`_load_dataset_records(dataset_id, max_results, page_token)`**
1. Query `SK begins_with DS#REC#` with `Limit=max_results`.
2. Cursor-based pagination using DynamoDB's native `LastEvaluatedKey`, encoded via the existing `encode_page_token({"lek": last_evaluated_key})` utility. Requires adding a paginated variant of `DynamoDBTable.query` that accepts `ExclusiveStartKey` and returns `LastEvaluatedKey` instead of auto-exhausting.
3. Return `(list[DatasetRecord], next_page_token)`.

**`delete_dataset_records(dataset_id, record_ids)`**
1. Batch delete record items by constructing `SK=DS#REC#<record_id>` for each (uses `batch_delete`).
2. Recompute profile count. Schema is not recomputed (may become stale if deleted records were the only source of a field — acceptable for Phase 1).
3. Update META digest and last_update_time.
4. Return count of deleted records.

### Tags

**`set_dataset_tags(dataset_id, tags)`**
1. For each tag: put item `SK=DS#TAG#<key>` (overwrite = upsert).
2. Update denormalized tags dict on META item.
3. Update last_update_time and digest.

**`delete_dataset_tag(dataset_id, key)`**
1. Delete item `SK=DS#TAG#<key>`.
2. Remove key from denormalized tags dict on META.
3. Update last_update_time and digest.

### Experiment Association

**`get_dataset_experiment_ids(dataset_id)`**
1. Query `SK begins_with DS#EXP#`.
2. Extract experiment_ids from SK suffixes.

**`add_dataset_to_experiments(dataset_id, experiment_ids)`**
1. For each experiment_id: put link item `SK=DS#EXP#<exp_id>` with GSI1 projection.
2. Update META last_update_time and digest.

**`remove_dataset_from_experiments(dataset_id, experiment_ids)`**
1. For each experiment_id: delete link item `SK=DS#EXP#<exp_id>`.
2. Update META last_update_time and digest.

### Legacy V2

**`_search_datasets(experiment_ids)`**
1. For each experiment_id: query `PK=EXP#<exp_id>, SK begins_with D#` to get dataset items.
2. Query `DLINK#` items to get run-dataset associations.
3. Return list of `DatasetSummary` objects (experiment_id, name, digest, context).

## Schema Constants

New constants to add to `dynamodb/schema.py`:

```python
# Dataset partition
PK_DATASET_PREFIX = "DS#"
SK_DATASET_META = "DS#META"
SK_DATASET_TAG_PREFIX = "DS#TAG#"
SK_DATASET_RECORD_PREFIX = "DS#REC#"
SK_DATASET_EXP_PREFIX = "DS#EXP#"

# GSI prefixes for evaluation datasets
# Note: GSI1_DS_PREFIX = "DS#" already exists for legacy V2 dataset items.
# The new DS_EXP# prefix is for experiment-dataset associations (distinct purpose).
GSI1_DS_EXP_PREFIX = "DS_EXP#"
GSI2_DS_LIST_PREFIX = "DS_LIST#"
GSI3_DS_NAME_PREFIX = "DS_NAME#"
```

## Testing Strategy

### Unit Tests (moto, direct store)

- CRUD lifecycle: create → get → search → delete
- Uniqueness enforcement: duplicate name within workspace rejected
- Record upsert: new insert, update existing (same input_hash), mixed batch
- Auto schema inference: correct field types inferred from records
- Auto profile: num_records tracked correctly after insert/delete
- Tag CRUD: set, overwrite, delete, idempotent delete
- Experiment association: add, remove, get, idempotent add
- Pagination: records and search with cursors
- Filter string: `name LIKE 'pattern%'`
- Order by: name, created_time, last_update_time
- Edge cases: delete non-existent dataset, empty records list, get non-existent raises

### Integration Tests (moto server, REST)

- `MlflowClient` round-trip: create → get → delete
- Record upsert and retrieval via client
- Search with experiment_ids filter
- Tag operations via client

### E2E Tests (full server)

- Remove xfail from `test_generate_demo_evaluation` once datasets are implemented
- Dataset CRUD via HTTP endpoints

### Coverage

100% patch coverage on new code.

## Out of Scope

- Online scoring (Phase 1c: Scorers)
- Logged models (Phase 1b)
- Dataset import/export (not in AbstractStore)
- Workspace isolation (uses existing `_workspace` field pattern)
