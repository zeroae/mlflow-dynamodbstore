# Access Patterns

This page documents the major DynamoDB access patterns used by `mlflow-dynamodbstore`,
grouped by feature area. Each pattern describes the operation, the keys or index used,
and the DynamoDB mechanism.

## Experiments

| Operation | Key / Index | Mechanism |
|---|---|---|
| Get experiment by ID | PK=`EXP#<id>`, SK=`E#META` | `GetItem` |
| Get experiment by name | GSI3: `gsi3pk`=`EXPNAME#<name>` | `Query` on GSI3 |
| List experiments in workspace | GSI2: `gsi2pk`=`WS#<workspace>` | `Query` on GSI2 |
| List experiments by name | GSI5: `gsi5pk`=`WS#<workspace>#EXP`, sorted by `gsi5sk` | `Query` on GSI5 |
| Create experiment | PK=`EXP#<id>`, SK=`E#META` | `PutItem` with condition (name uniqueness via GSI3) |
| Delete/restore experiment | PK=`EXP#<id>`, SK=`E#META` | `UpdateItem` (toggle lifecycle stage) |
| Set experiment tag | PK=`EXP#<id>`, SK=`E#TAG#<key>` | `PutItem` |
| Search experiments (FTS) | GSI2: `gsi2pk`=`FTS#<token>` | `Query` on GSI2, intersect results |

## Runs

| Operation | Key / Index | Mechanism |
|---|---|---|
| Get run by ID | GSI1: `gsi1pk`=`RUN#<run_id>` to resolve experiment, then PK=`EXP#<id>`, SK=`R#<run_id>` | `Query` on GSI1 + `GetItem` |
| List runs in experiment | PK=`EXP#<id>`, SK `begins_with` `R#` | `Query` with key condition |
| Filter runs by lifecycle | PK=`EXP#<id>`, LSI1 with `lsi1sk`=`ACTIVE` or `DELETED` | `Query` on LSI1 |
| Sort runs by start time | PK=`EXP#<id>`, LSI2 ordered by `lsi2sk` | `Query` on LSI2 |
| Filter runs by status | PK=`EXP#<id>`, LSI3 with `lsi3sk`=`RUNNING` etc. | `Query` on LSI3 |
| Sort runs by name | PK=`EXP#<id>`, LSI4 ordered by `lsi4sk` | `Query` on LSI4 |
| Sort runs by metric | PK=`EXP#<id>`, LSI5 ordered by `lsi5sk` | `Query` on LSI5 |
| Create run | PK=`EXP#<id>`, SK=`R#<run_id>` | `PutItem` |
| Update run info | PK=`EXP#<id>`, SK=`R#<run_id>` | `UpdateItem` |
| Delete/restore run | PK=`EXP#<id>`, SK=`R#<run_id>` | `UpdateItem` (toggle lifecycle stage) |

## Run Data (Metrics, Params, Tags)

| Operation | Key / Index | Mechanism |
|---|---|---|
| Log metric | PK=`EXP#<id>`, SK=`R#<run_id>#METRIC#<key>` | `PutItem` (latest value) |
| Log metric history | PK=`EXP#<id>`, SK=`R#<run_id>#MHIST#<key>#<step>#<ts>` | `PutItem` |
| Get metric history | PK=`EXP#<id>`, SK `begins_with` `R#<run_id>#MHIST#<key>` | `Query` with key condition |
| Log param | PK=`EXP#<id>`, SK=`R#<run_id>#PARAM#<key>` | `PutItem` |
| Get all params for run | PK=`EXP#<id>`, SK `begins_with` `R#<run_id>#PARAM#` | `Query` with key condition |
| Set run tag | PK=`EXP#<id>`, SK=`R#<run_id>#TAG#<key>` | `PutItem` |
| Get all tags for run | PK=`EXP#<id>`, SK `begins_with` `R#<run_id>#TAG#` | `Query` with key condition |
| Log batch (metrics + params + tags) | Multiple items in experiment partition | `BatchWriteItem` or `TransactWriteItems` |

## Search and Ranking

| Operation | Key / Index | Mechanism |
|---|---|---|
| Search runs by metric value | PK=`EXP#<id>`, SK `begins_with` `RANK#m#<key>#` | `Query` with key condition (inverted values for descending order) |
| Search runs by param value | PK=`EXP#<id>`, SK `begins_with` `RANK#p#<key>#` | `Query` with key condition |
| Full-text search (forward) | PK=`EXP#<id>`, SK `begins_with` `FTS#<level>#<token>` | `Query` with key condition |
| Full-text search (global) | GSI2: `gsi2pk`=`FTS#<token>` | `Query` on GSI2 |
| FTS index cleanup | PK=`EXP#<id>`, SK `begins_with` `FTS_REV#<entity_type>#<entity_id>` | `Query` reverse index, then `BatchWriteItem` deletes |

## Traces

| Operation | Key / Index | Mechanism |
|---|---|---|
| Get trace by ID | GSI1: `gsi1pk`=`TRACE#<trace_id>` to resolve experiment, then PK=`EXP#<id>`, SK=`T#<trace_id>` | `Query` on GSI1 + `GetItem` |
| List traces in experiment | PK=`EXP#<id>`, SK `begins_with` `T#` | `Query` with key condition |
| Sort traces by time | PK=`EXP#<id>`, LSI2 ordered by `lsi2sk` | `Query` on LSI2 |
| Sort traces by name | PK=`EXP#<id>`, LSI4 ordered by `lsi4sk` | `Query` on LSI4 |
| Set trace tag | PK=`EXP#<id>`, SK=`T#<trace_id>#TAG#<key>` | `PutItem` |
| Get trace spans | PK=`EXP#<id>`, SK=`T#<trace_id>#SPANS` | `GetItem` |
| Create trace + spans | PK=`EXP#<id>`, SK=`T#<trace_id>` and `T#<trace_id>#SPANS` | `TransactWriteItems` |

## Datasets

| Operation | Key / Index | Mechanism |
|---|---|---|
| Create dataset | PK=`EXP#<id>`, SK=`D#<dataset_hash>` | `PutItem` |
| Get dataset | PK=`EXP#<id>`, SK=`D#<dataset_hash>` | `GetItem` |
| Link dataset to run | PK=`EXP#<id>`, SK=`DLINK#<run_id>#<dataset_name>` | `PutItem` |
| List datasets for run | PK=`EXP#<id>`, SK `begins_with` `DLINK#<run_id>#` | `Query` with key condition |

## Logged Models

| Operation | Key / Index | Mechanism |
|---|---|---|
| Log model to run | PK=`EXP#<id>`, SK=`R#<run_id>#LM#<model_name>` | `PutItem` |
| List logged models for run | PK=`EXP#<id>`, SK `begins_with` `R#<run_id>#LM#` | `Query` with key condition |

## Model Registry

| Operation | Key / Index | Mechanism |
|---|---|---|
| Create registered model | PK=`RM#<name>`, SK=`M#META` | `PutItem` with condition |
| Get registered model | PK=`RM#<name>`, SK=`M#META` | `GetItem` |
| Get model by name | GSI3: `gsi3pk`=`RMNAME#<name>` | `Query` on GSI3 |
| List registered models | GSI5: `gsi5pk`=`RM_LIST`, sorted by `gsi5sk` | `Query` on GSI5 |
| Create model version | PK=`RM#<name>`, SK=`M#V#<version>` | `PutItem` |
| Get model version | PK=`RM#<name>`, SK=`M#V#<version>` | `GetItem` |
| List model versions | PK=`RM#<name>`, SK `begins_with` `M#V#` | `Query` with key condition |
| Set model tag | PK=`RM#<name>`, SK=`M#TAG#<key>` | `PutItem` |
| Set model alias | PK=`RM#<name>`, SK=`M#ALIAS#<alias>` | `PutItem` |
| Get model by alias | PK=`RM#<name>`, SK=`M#ALIAS#<alias>` | `GetItem` |

## Authentication and Authorization

| Operation | Key / Index | Mechanism |
|---|---|---|
| Create user | PK=`USER#<username>`, SK=`U#META` | `PutItem` with condition |
| Get user | PK=`USER#<username>`, SK=`U#META` | `GetItem` |
| Authenticate user | PK=`USER#<username>`, SK=`U#META` | `GetItem` + password hash comparison |
| Grant experiment permission | PK=`USER#<username>`, SK=`U#PERM#<experiment_id>` | `PutItem` |
| Check experiment permission | PK=`USER#<username>`, SK=`U#PERM#<experiment_id>` | `GetItem` |
| List permissions for experiment | GSI4: `gsi4pk`=`EXPPERM#<experiment_id>` | `Query` on GSI4 |
| Grant registry permission | PK=`USER#<username>`, SK=`U#RPERM#<model_name>` | `PutItem` |
| Check registry permission | PK=`USER#<username>`, SK=`U#RPERM#<model_name>` | `GetItem` |
| List permissions for model | GSI4: `gsi4pk`=`RMPERM#<model_name>` | `Query` on GSI4 |

## Workspace

| Operation | Key / Index | Mechanism |
|---|---|---|
| Get workspace | PK=`WORKSPACE#<name>`, SK=`WS#META` | `GetItem` |
| Create/update workspace | PK=`WORKSPACE#<name>`, SK=`WS#META` | `PutItem` |

## Configuration

| Operation | Key / Index | Mechanism |
|---|---|---|
| Get config value | PK=`CONFIG`, SK=`CFG#<key>` | `GetItem` |
| Set config value | PK=`CONFIG`, SK=`CFG#<key>` | `PutItem` |
| List all config | PK=`CONFIG`, SK `begins_with` `CFG#` | `Query` with key condition |
