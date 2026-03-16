# DynamoDB Single-Table Schema

This page documents the complete DynamoDB single-table design used by `mlflow-dynamodbstore`.
All entities share a single table, differentiated by **partition key (PK)** and **sort key (SK)** prefixes.

## Partition Families

| PK Prefix | Description |
|---|---|
| `EXP#<experiment_id>` | Experiment partition -- contains experiment metadata, runs, traces, metrics, params, tags, datasets, rankings, and FTS entries |
| `RM#<model_name>` | Model registry partition -- model versions, tags, and aliases |
| `WORKSPACE#<name>` | Workspace partition -- workspace-level configuration |
| `USER#<username>` | User partition -- user credentials and permissions |
| `CONFIG` | Global configuration partition |

## Entity Types by Partition

### Experiment Partition (`EXP#<experiment_id>`)

#### Core Entities

| SK Pattern | Entity | Description |
|---|---|---|
| `E#META` | Experiment metadata | Name, artifact location, lifecycle stage, creation/update time |
| `E#TAG#<key>` | Experiment tag | Key-value tag on the experiment |
| `E#NAME_REV` | Reversed name | Reversed experiment name for prefix search support |

#### Run Entities

| SK Pattern | Entity | Description |
|---|---|---|
| `R#<run_id>` | Run metadata | Run name, status, lifecycle stage, start/end time, user, artifact URI |
| `R#<run_id>#METRIC#<key>` | Latest metric | Most recent value for a metric key |
| `R#<run_id>#MHIST#<key>#<step>#<ts>` | Metric history | Full metric history entry at a given step and timestamp |
| `R#<run_id>#PARAM#<key>` | Parameter | Run parameter key-value pair |
| `R#<run_id>#TAG#<key>` | Run tag | Key-value tag on a run |
| `R#<run_id>#INPUT#<name>` | Dataset input | Dataset input linked to a run |
| `R#<run_id>#LM#<model_name>` | Logged model | Model logged during a run |

#### Trace Entities

| SK Pattern | Entity | Description |
|---|---|---|
| `T#<trace_id>` | Trace metadata | Trace request metadata (name, status, timestamps) |
| `T#<trace_id>#TAG#<key>` | Trace tag | Key-value tag on a trace |
| `T#<trace_id>#SPANS` | Cached spans | Pre-serialized span data for the trace |

#### Dataset Entities

| SK Pattern | Entity | Description |
|---|---|---|
| `D#<dataset_hash>` | Dataset | Dataset metadata (name, digest, source, schema, profile) |
| `DLINK#<run_id>#<dataset_name>` | Dataset link | Association between a run and a dataset |

#### Ranking Entities

| SK Pattern | Entity | Description |
|---|---|---|
| `RANK#m#<key>#<inv_value>#<run_id>` | Metric rank | Inverted metric value for descending sort order |
| `RANK#p#<key>#<value>#<run_id>` | Param rank | Parameter value for lexicographic sorting |

#### Full-Text Search Entities

| SK Pattern | Entity | Description |
|---|---|---|
| `FTS#<level>#<token>#<entity_type>#<entity_id>` | FTS forward index | Token-to-entity mapping for search |
| `FTS_REV#<entity_type>#<entity_id>#<level>#<token>` | FTS reverse index | Entity-to-token mapping for index maintenance |

### Model Registry Partition (`RM#<model_name>`)

| SK Pattern | Entity | Description |
|---|---|---|
| `M#META` | Model metadata | Model name, description, creation/update time, tags |
| `M#V#<version>` | Model version | Version metadata (source, run link, status, stage) |
| `M#TAG#<key>` | Model tag | Key-value tag on a registered model |
| `M#ALIAS#<alias>` | Model alias | Named alias pointing to a specific version |

### Workspace Partition (`WORKSPACE#<name>`)

| SK Pattern | Entity | Description |
|---|---|---|
| `WS#META` | Workspace metadata | Workspace name and configuration |

### User Partition (`USER#<username>`)

| SK Pattern | Entity | Description |
|---|---|---|
| `U#META` | User metadata | Username, password hash, admin flag |
| `U#PERM#<experiment_id>` | Experiment permission | User's permission level for a specific experiment |
| `U#RPERM#<model_name>` | Registry permission | User's permission level for a registered model |

### Config Partition (`CONFIG`)

| SK Pattern | Entity | Description |
|---|---|---|
| `CFG#<key>` | Configuration entry | Global configuration key-value pair |

## Global Secondary Indexes (GSIs)

GSIs provide alternative access patterns by projecting different key combinations.

| Index | Partition Key | Sort Key | Purpose |
|---|---|---|---|
| **GSI1** | `gsi1pk` | `gsi1sk` | Reverse lookups -- find experiment by run ID or trace ID |
| **GSI2** | `gsi2pk` | `gsi2sk` | Collection queries -- list all experiments in a workspace, full-text search |
| **GSI3** | `gsi3pk` | `gsi3sk` | Name lookups -- find experiment by name, model by name |
| **GSI4** | `gsi4pk` | `gsi4sk` | Permission lookups -- find all permissions for a user or resource |
| **GSI5** | `gsi5pk` | `gsi5sk` | Name listing queries -- paginated name-ordered listings |

### GSI Key Projections by Entity

#### GSI1 -- Reverse Lookups

| Entity | `gsi1pk` | `gsi1sk` |
|---|---|---|
| Run | `RUN#<run_id>` | `EXP#<experiment_id>` |
| Trace | `TRACE#<trace_id>` | `EXP#<experiment_id>` |

#### GSI2 -- Collection Queries

| Entity | `gsi2pk` | `gsi2sk` |
|---|---|---|
| Experiment | `WS#<workspace>` | `EXP#<experiment_id>` |
| FTS forward | `FTS#<token>` | `<level>#<entity_type>#<entity_id>` |

#### GSI3 -- Name Lookups

| Entity | `gsi3pk` | `gsi3sk` |
|---|---|---|
| Experiment | `EXPNAME#<name>` | `EXP#<experiment_id>` |
| Registered model | `RMNAME#<name>` | `RM#<model_name>` |

#### GSI4 -- Permission Lookups

| Entity | `gsi4pk` | `gsi4sk` |
|---|---|---|
| Experiment permission | `EXPPERM#<experiment_id>` | `USER#<username>` |
| Registry permission | `RMPERM#<model_name>` | `USER#<username>` |

#### GSI5 -- Name Listing Queries

| Entity | `gsi5pk` | `gsi5sk` |
|---|---|---|
| Experiment | `WS#<workspace>#EXP` | `<experiment_name>` |
| Registered model | `RM_LIST` | `<model_name>` |

## Local Secondary Indexes (LSIs)

LSIs share the table's partition key (PK) but use an alternate sort key, enabling filtered queries within a partition.

| Index | Sort Key | Purpose | Typical Values |
|---|---|---|---|
| **LSI1** | `lsi1sk` | Lifecycle filter | `ACTIVE`, `DELETED` -- filter runs by lifecycle stage |
| **LSI2** | `lsi2sk` | Time-based ordering | ISO-8601 timestamps -- sort runs/traces by creation time |
| **LSI3** | `lsi3sk` | Status filter | `RUNNING`, `FINISHED`, `FAILED`, `KILLED` -- filter runs by status |
| **LSI4** | `lsi4sk` | Name-based ordering | Run name or trace name -- alphabetical listing |
| **LSI5** | `lsi5sk` | Metric-based ordering | Numeric metric values -- sort runs by a metric |

### LSI Usage by Entity

| Entity | LSI1 (lifecycle) | LSI2 (time) | LSI3 (status) | LSI4 (name) | LSI5 (metric) |
|---|---|---|---|---|---|
| Run | Lifecycle stage | Start time | Run status | Run name | Primary metric |
| Trace | -- | Request time | Trace status | Trace name | -- |

## Key Design Principles

**Single-table design.**
All entities reside in one DynamoDB table, minimizing the number of API calls and enabling transactional writes across entity types.

**Hierarchical partitioning.**
The experiment partition groups all related data (runs, metrics, params, traces) under a single partition key, making experiment-scoped queries efficient.

**Inverted indexes via GSIs.**
GSIs serve as inverted indexes for reverse lookups (run to experiment), name resolution, and cross-partition queries without table scans.

**LSIs for filtered scans.**
Local secondary indexes enable filtered and sorted queries within a partition without reading irrelevant items.

**Composite sort keys.**
Multi-segment sort keys (e.g., `R#<run_id>#METRIC#<key>`) enable efficient range queries using `begins_with` conditions.
