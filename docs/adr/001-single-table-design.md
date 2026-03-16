# ADR-001: Single Table Design

| Field    | Value                  |
|----------|------------------------|
| Status   | Accepted               |
| Date     | 2025-01-15             |
| Authors  | ZeroAE                 |

## Context

mlflow-dynamodbstore must persist all MLflow entities in DynamoDB: experiments,
runs, metrics, parameters, tags, registered models, model versions, aliases,
traces, authentication users/permissions, and workspaces. The schema must
support MLflow's diverse access patterns (list by workspace, search by name,
filter by tags, paginated metric history, full-text search) while keeping
operational complexity low.

## Decision

**Use a single DynamoDB table with composite primary keys (PK + SK) and
secondary indexes to store all entity types.**

Every item shares the same table and the same `PK`/`SK` attribute pair. Entity
types are distinguished by key prefixes:

| Prefix     | Partition contents                                |
|------------|---------------------------------------------------|
| `EXP#`     | Experiments, runs, metrics, params, tags, traces  |
| `RM#`      | Registered models, versions, aliases, model tags  |
| `WORKSPACE#` | Workspace metadata                             |
| `USER#`    | Auth users and permissions                        |
| `CONFIG`   | System configuration (TTL policy, FTS fields)     |

Within each partition, the sort key encodes the entity hierarchy:

```
EXP#<experiment_id>
    E#META                          -- experiment metadata
    E#TAG#<key>                     -- experiment tag
    R#<run_ulid>                    -- run metadata
    R#<run_ulid>#METRIC#<key>       -- latest metric value
    R#<run_ulid>#MHIST#<key>#<step> -- metric history point
    R#<run_ulid>#PARAM#<key>        -- parameter
    R#<run_ulid>#TAG#<key>          -- run tag
    T#<trace_ulid>                  -- trace metadata
    FTS#<level>#<token>#...         -- full-text search (forward)
    FTS_REV#<entity>#...            -- full-text search (reverse)
```

### Secondary Indexes

The table uses 5 Global Secondary Indexes (GSIs) and 5 Local Secondary Indexes
(LSIs) to support all required access patterns:

**GSIs** (cross-partition queries):

| Index  | PK prefix            | Purpose                                    |
|--------|----------------------|--------------------------------------------|
| GSI1   | `RUN#`, `TRACE#`, `CLIENT#`, `DS#` | Look up runs/traces by ID across experiments |
| GSI2   | `EXPERIMENTS#`, `MODELS#`, `FTS_NAMES#` | List experiments/models per workspace; cross-partition FTS |
| GSI3   | `EXP_NAME#`, `MODEL_NAME#`, `ALIAS#` | Unique name lookups and alias resolution |
| GSI4   | `PERM#`              | Permission lookups by resource             |
| GSI5   | `EXP_NAMES#`, `MODEL_NAMES#` | Sorted name listings for search         |

**LSIs** (same-partition alternate sort orders):

LSI1 through LSI5 provide alternative sort keys on experiment and model
partitions, enabling sorted queries by creation time, update time, status,
and other frequently filtered attributes.

## Alternatives Considered

### Multi-table: One Table per Entity Type

A separate table for experiments, runs, metrics, models, etc.

- **Pros:** Simpler per-table schema; easier to reason about individual entity access.
- **Cons:** Cross-entity queries require scatter-gather across tables; transactions
  spanning tables increase latency and cost; more CloudFormation resources to
  manage; per-table capacity planning is harder to optimize.

### Hybrid: Core Entities + Search Table

Core data in one table, with a second table dedicated to search indexes (FTS
trigrams, denormalized tags).

- **Pros:** Isolates search write amplification from core CRUD operations.
- **Cons:** Adds a second table to provision and monitor; DynamoDB's GSIs and
  LSIs already provide sufficient indexing within a single table; two tables
  complicates consistency (search items can lag behind source of truth).

## Consequences

### Benefits

- **Single billing unit.** One table means one set of read/write capacity
  settings and one cost line item to monitor.
- **Simplified operations.** A single CloudFormation stack provisions the entire
  data layer. Backups, restores, and point-in-time recovery cover all entities
  at once.
- **Transactional writes.** DynamoDB transactions work within a single table
  without cross-table coordination. Creating a run with its initial tags is
  one `TransactWriteItems` call.
- **Collocated data.** Runs, metrics, params, and tags share a partition with
  their parent experiment, so listing a run's details is a single partition
  query.

### Trade-offs

- **Access patterns must be designed upfront.** DynamoDB does not support ad-hoc
  queries. Every access pattern needs a corresponding key design or secondary
  index. Adding a new query shape may require a new GSI or schema migration.
- **10 GB partition limit from LSIs.** Because the table uses LSIs, each
  partition key collection (e.g., all items under `EXP#<id>`) is capped at
  10 GB. Experiments with very high metric volume need monitoring.
- **Hot partition risk.** If a single experiment accumulates a disproportionate
  number of runs or metrics, its partition can become a throughput bottleneck.
  Adaptive capacity helps, but the risk should be monitored.
- **Schema evolution requires care.** Changing key prefixes or index projections
  requires a data migration. The schema module (`dynamodb/schema.py`) is the
  single source of truth for all key constants to reduce drift.
- **Write amplification for search.** Full-text search uses trigram tokenization,
  generating multiple forward and reverse FTS items per indexed field. This
  increases write costs but keeps reads fast by avoiding scan-and-filter.
- **Denormalization for query performance.** Tag values are denormalized onto
  META items and into LSI sort keys so that filtered searches can avoid
  additional lookups. This trades storage and write complexity for read
  efficiency.
