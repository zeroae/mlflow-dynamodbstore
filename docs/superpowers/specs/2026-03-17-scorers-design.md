# Phase 1c: Scorers — DynamoDB Implementation

## Context

The MLflow 3.x UI calls `list_scorers` on every experiment overview page load. The DynamoDB tracking store does not implement this method, causing 500 errors. This spec covers the third sub-phase: Scorers.

Scorers are experiment-scoped, versioned entities that define evaluation logic (e.g., relevance, correctness). Each scorer has a name, a serialized scorer definition (JSON), and auto-incrementing version numbers. Scorers can optionally have online scoring configurations that sample and score traces in real time.

## Scope

8 store methods:

| Method | Category |
|--------|----------|
| `register_scorer` | Lifecycle |
| `get_scorer` | Lifecycle |
| `list_scorers` | Lifecycle |
| `list_scorer_versions` | Lifecycle |
| `delete_scorer` | Lifecycle |
| `upsert_online_scoring_config` | Online Config |
| `get_online_scoring_configs` | Online Config |
| `get_active_online_scorers` | Online Config |

**Not in scope:**
- `calculate_trace_filter_correlation` — trace analysis, not scorer storage; deferred to Phase 3 (advanced traces).
- AI Gateway endpoint resolution — separate plan. The DynamoDB store stores endpoint names directly in `serialized_scorer` and skips ID↔name resolution.
- Scorer invocation (async job infrastructure) — handled by MLflow's built-in job system, not the tracking store.

## DynamoDB Item Design

Scorers are experiment-scoped (each belongs to exactly one experiment), so they live under the existing `EXP#<exp_id>` partition — same pattern as runs, traces, and logged models.

### Item Types

| Item | PK | SK | lsi3sk | Attributes |
|------|----|----|--------|------------|
| Scorer META | `EXP#<exp_id>` | `SCOR#<scorer_id>` | `lower(scorer_name)` | scorer_name, scorer_id, latest_version, workspace |
| Scorer Version | `EXP#<exp_id>` | `SCOR#<scorer_id>#V#<padded_ver>` | — | scorer_version, serialized_scorer (JSON text), creation_time |
| Online Scoring Config | `EXP#<exp_id>` | `SCOR#<scorer_id>#OSCFG` | — | online_scoring_config_id, scorer_id, sample_rate (Decimal), experiment_id, filter_string |

Tag, Param, and Metric items are not used for scorers. The `serialized_scorer` field on version items contains the complete scorer definition as a JSON string.

**Online Scoring Config SK**: Uses a fixed suffix `#OSCFG` (no config_id in the SK) so that each scorer has exactly one config item. Upserts are atomic single-item `PutItem` overwrites — no delete-then-write window.

### ID Generation

- `scorer_id`: ULID — generated on first registration of a scorer name within an experiment.
- `online_scoring_config_id`: ULID — generated on each `upsert_online_scoring_config` call, stored as an attribute (not in SK).
- `scorer_version`: Sequential integer, auto-incremented from `latest_version` on the META item.
- `padded_ver`: 10-digit zero-padded version (e.g., `0000000001`) for lexicographic sort.

### GSI Usage

| GSI | PK | SK | Purpose |
|-----|----|----|---------|
| GSI1 | `SCOR#<scorer_id>` | `EXP#<exp_id>` | Reverse lookup: scorer_id → experiment_id. Written on META items. Needed by `get_online_scoring_configs(scorer_ids)` to resolve experiment partition. |
| GSI2 | `ACTIVE_SCORERS#<workspace>` | `<scorer_id>` | Cross-experiment lookup: find all active online scorers (sample_rate > 0). Written on Online Scoring Config items only when sample_rate > 0. Omitted when sample_rate = 0. |
| GSI3 | `SCOR_NAME#<workspace>#<exp_id>#<scorer_name>` | `<scorer_id>` | Unique name lookup within experiment. Written on META items. |

### LSI Usage

| LSI | Scorer META usage | Purpose |
|-----|-------------------|---------|
| LSI3 | `lower(scorer_name)` | List scorers by name within experiment. Only META items populate LSI3, so querying LSI3 returns META items exclusively — no version/config over-read. |

Only META items populate LSI attributes. Version and config items omit them.

## Access Patterns

Every store method maps to Query or GetItem — never Scan.

| # | Access Pattern | Caller | Operation | Table/Index | Key Condition | Notes |
|---|---------------|--------|-----------|-------------|---------------|-------|
| AP1 | Resolve scorer name → ID | `get_scorer`, `delete_scorer`, `upsert_online_scoring_config` | Query | GSI3 | `PK=SCOR_NAME#<ws>#<exp_id>#<name>` | Expect 0 or 1 result |
| AP2 | Get specific version | `get_scorer(version=N)` | GetItem | Table | `PK=EXP#<exp_id>, SK=SCOR#<scorer_id>#V#<padded_ver>` | O(1) |
| AP3 | Get latest version | `get_scorer(version=None)` | Query | Table | `PK=EXP#<exp_id>, SK begins_with SCOR#<scorer_id>#V#`, ScanIndexForward=False, Limit=1 | Returns highest version. Authoritative — `latest_version` on META is a write-optimization cache only. |
| AP4 | List all scorer META items | `list_scorers` | Query | LSI3 | `PK=EXP#<exp_id>`, FilterExpression: `SK begins_with SCOR#` | Only META items populate LSI3, so this returns META items exclusively without reading version/config items. Then batch-get latest version per scorer via AP3. |
| AP5 | List all versions of a scorer | `list_scorer_versions` | Query | Table | `PK=EXP#<exp_id>, SK begins_with SCOR#<scorer_id>#V#` | Returns all versions, sorted ascending |
| AP6 | Delete scorer (all items) | `delete_scorer(version=None)` | Query + BatchDelete | Table | `PK=EXP#<exp_id>, SK begins_with SCOR#<scorer_id>` | Deletes META + all versions + config |
| AP7 | Delete single version | `delete_scorer(version=N)` | DeleteItem | Table | `PK=EXP#<exp_id>, SK=SCOR#<scorer_id>#V#<padded_ver>` | Single item delete |
| AP8 | Get online config for a scorer | `get_online_scoring_configs` | GetItem | Table | `PK=EXP#<exp_id>, SK=SCOR#<scorer_id>#OSCFG` | O(1) per scorer_id. Requires resolving experiment_id via GSI1 first. |
| AP9 | Get all active online scorers | `get_active_online_scorers` | Query | GSI2 | `PK=ACTIVE_SCORERS#<workspace>` | Returns config items with active scoring; deduplicate by scorer_id. Batch-get META + latest version per scorer. |
| AP10 | Check name uniqueness | `register_scorer` | Query | GSI3 | `PK=SCOR_NAME#<ws>#<exp_id>#<name>` | Same as AP1; 0 results = new scorer |
| AP11 | Resolve scorer_id → experiment_id | `get_online_scoring_configs` | Query | GSI1 | `PK=SCOR#<scorer_id>` | Returns experiment_id for constructing table PK |

### Scan Risk Assessment

**All 11 access patterns are efficient.** No table scans. Every operation uses GetItem, Query with precise key conditions, or BatchDelete with known keys. The cross-experiment `get_active_online_scorers` uses GSI2 with a dedicated partition key, bounded by the number of active scoring configs. `list_scorers` uses LSI3 to return only META items, avoiding over-read of version and config items.

## Store Methods

### Lifecycle

**`register_scorer(experiment_id, name, serialized_scorer)`**
1. Verify experiment exists.
2. Query GSI3 for existing scorer with this name in this experiment (AP1/AP10).
3. If no existing scorer:
   a. Generate ULID scorer_id.
   b. Put META item with `latest_version=1`, GSI1, GSI3, LSI3 projections. **Use `ConditionExpression: attribute_not_exists(SK)`** to guard against concurrent first-registration races. If condition fails, retry from step 2 (take existing-scorer path).
   c. Put Version item `SK=SCOR#<scorer_id>#V#0000000001`.
4. If existing scorer:
   a. Atomic `UpdateItem` on META with `ADD latest_version 1`, `ReturnValues=UPDATED_NEW` to get the new version number in one round-trip.
   b. Put new Version item with the returned version number.
5. Return `ScorerVersion` entity.

**`get_scorer(experiment_id, name, version=None)`**
1. Resolve scorer_id via GSI3 (AP1).
2. If not found: raise RESOURCE_DOES_NOT_EXIST.
3. If version specified: GetItem version item (AP2).
4. If version=None: Query for latest version (AP3). This is authoritative (SK-order based), not dependent on the `latest_version` cache.
5. Return `ScorerVersion` entity.

**`list_scorers(experiment_id)`**
1. Query LSI3 within experiment partition with `FilterExpression: SK begins_with SCOR#` (AP4). Since only META items populate LSI3, this returns scorer META items exclusively — no version or config items are read.
2. For each META item: get latest version via AP3 (one query per scorer, bounded by scorer count which is typically small).
3. Return `list[ScorerVersion]`.

**`list_scorer_versions(experiment_id, name)`**
1. Resolve scorer_id via GSI3.
2. Query all version items (AP5).
3. Return `list[ScorerVersion]` ordered by version ascending.

**`delete_scorer(experiment_id, name, version=None)`**
1. Resolve scorer_id via GSI3.
2. If version=None:
   a. Query ALL items with `SK begins_with SCOR#<scorer_id>` (META + versions + config).
   b. Batch delete all items.
   c. GSI entries auto-cleaned.
3. If version specified:
   a. Delete single version item (AP7).
   b. Query remaining versions (AP5) to check if any remain.
   c. If no versions remain: also delete META item (and config item if present).
   d. If versions remain and the deleted version was the latest: `UpdateItem` on META with `SET latest_version = :new_max, ConditionExpression: latest_version = :old_max`. The `latest_version` field is a write-optimization cache for `register_scorer`; `get_scorer(version=None)` always uses AP3 (SK sort) as the authoritative source.

### Online Scoring Configuration

**`upsert_online_scoring_config(experiment_id, scorer_name, sample_rate, filter_string=None)`**
1. Validate sample_rate is numeric, in [0.0, 1.0].
2. Resolve scorer_id via GSI3.
3. Generate new `online_scoring_config_id` (ULID).
4. Put config item `SK=SCOR#<scorer_id>#OSCFG` (atomic overwrite — fixed SK means PutItem replaces any existing config):
   - If sample_rate > 0: include GSI2 attributes (`ACTIVE_SCORERS#<workspace>` → `<scorer_id>`).
   - If sample_rate = 0: omit GSI2 attributes (removes from active scorers index). The config item still exists (records that online scoring was explicitly disabled) but is not indexed in GSI2.
5. Return `OnlineScoringConfig` entity.

**`get_online_scoring_configs(scorer_ids)`**
1. For each scorer_id: resolve experiment_id via GSI1 (AP11).
2. For each (experiment_id, scorer_id): GetItem config at `SK=SCOR#<scorer_id>#OSCFG` (AP8).
3. Return `list[OnlineScoringConfig]` (excluding scorer_ids with no config item).

**`get_active_online_scorers()`**
1. Query GSI2: `PK=ACTIVE_SCORERS#<workspace>` (AP9).
2. Deduplicate results by scorer_id (guard against transient GSI consistency).
3. For each result: extract scorer_id and experiment_id from config item attributes.
4. Batch-get Scorer META items to get scorer_name.
5. For each scorer: get latest version (AP3) to get serialized_scorer.
6. Return `list[OnlineScorer]` — each containing scorer name, serialized_scorer, and online_config.

## Schema Constants

New constants to add to `dynamodb/schema.py`:

```python
# Scorer sort keys (within EXP# partition)
SK_SCORER_PREFIX = "SCOR#"
SK_SCORER_OSCFG_SUFFIX = "#OSCFG"

# GSI prefixes for scorers
GSI1_SCOR_PREFIX = "SCOR#"
GSI2_ACTIVE_SCORERS_PREFIX = "ACTIVE_SCORERS#"
GSI3_SCOR_NAME_PREFIX = "SCOR_NAME#"
```

No new partition key prefixes — scorers reuse the existing `EXP#<exp_id>` partition family.

## Entity Mapping

| MLflow Entity | DynamoDB Source |
|---------------|----------------|
| `ScorerVersion.experiment_id` | From PK or query context |
| `ScorerVersion.scorer_name` | META item `scorer_name` attribute |
| `ScorerVersion.scorer_version` | Version item `scorer_version` attribute |
| `ScorerVersion.serialized_scorer` | Version item `serialized_scorer` attribute (JSON string) |
| `ScorerVersion.creation_time` | Version item `creation_time` attribute |
| `ScorerVersion.scorer_id` | Extracted from SK (`SCOR#<scorer_id>`) |
| `OnlineScoringConfig.online_scoring_config_id` | Config item `online_scoring_config_id` attribute |
| `OnlineScoringConfig.scorer_id` | Config item `scorer_id` attribute |
| `OnlineScoringConfig.sample_rate` | Config item `sample_rate` attribute (Decimal → float) |
| `OnlineScoringConfig.experiment_id` | Config item `experiment_id` attribute |
| `OnlineScoringConfig.filter_string` | Config item `filter_string` attribute (nullable) |

## Gateway Endpoint Resolution

The SQLAlchemy store resolves gateway endpoint IDs to endpoint names (and vice versa) using a `gateway_endpoints` table. The DynamoDB store does not have this table.

**Decision**: Store endpoint names directly in `serialized_scorer`. No ID↔name resolution. AI Gateway support (endpoint management, bindings) is a separate plan.

**Impact**: Scorers that reference gateway models will store the endpoint name as-is. If AI Gateway is later implemented, a migration step can populate endpoint bindings.

## Testing Strategy

### Unit Tests (moto, direct store)

- **Register**: first registration creates META + Version; second registration increments version
- **Concurrent registration guard**: verify `ConditionExpression` on initial META put prevents duplicate scorers
- **Get scorer**: by name (latest), by name + specific version, non-existent raises
- **List scorers**: returns latest version per scorer name, empty list for no scorers
- **List versions**: all versions of a scorer, ordered ascending
- **Delete**: delete all versions removes META + versions + config; delete single version; delete last version also removes META
- **Delete latest version**: verify `latest_version` cache is updated and `get_scorer(version=None)` returns correct result
- **Version numbering**: sequential auto-increment, gaps after single-version delete
- **Online config upsert**: create config, replace existing config (atomic overwrite), sample_rate=0 removes from GSI2
- **Get online configs**: returns configs for given scorer_ids, empty for no configs
- **Get active online scorers**: returns only scorers with sample_rate > 0, cross-experiment, deduplicates by scorer_id
- **Name uniqueness**: same name in different experiments creates separate scorers
- **Edge cases**: register with empty serialized_scorer, delete non-existent scorer raises

### Integration Tests (moto server, REST)

- `MlflowClient` round-trip: register → get → list → delete
- Online config operations via REST
- List scorers returns empty list (not 500)

### E2E Tests (full server)

- Experiment overview page loads without 500 on `/scorers/list`
- Register and list scorers via HTTP endpoints

### Coverage

100% patch coverage on new code.

## Out of Scope

- AI Gateway endpoint management and bindings (separate plan)
- Scorer invocation / async job execution (MLflow built-in)
- `calculate_trace_filter_correlation` (Phase 3: advanced traces)
- Online scoring scheduler/processor (MLflow built-in job infrastructure)
- Scorer name validation (reuse MLflow's `validate_scorer_name`)
