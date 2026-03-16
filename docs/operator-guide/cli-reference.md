# CLI Reference

The `mlflow-dynamodbstore` CLI provides admin commands for managing your
DynamoDB-backed MLflow deployment. All commands require `--table` and `--region`
options.

```bash
mlflow-dynamodbstore --help
```

## Global Options

Every subcommand accepts:

| Option     | Description            | Required |
|------------|------------------------|----------|
| `--table`  | DynamoDB table name    | Yes      |
| `--region` | AWS region (e.g. `us-east-1`) | Yes |

---

## denormalize-tags

Manage tag denormalization patterns. Denormalized tags are copied onto META items
so they can be used in filter expressions without additional queries.

### list

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.denormalize_tags
    :command: list_patterns
    :prog_name: mlflow-dynamodbstore denormalize-tags list

**Examples:**

```bash
# List global patterns
mlflow-dynamodbstore denormalize-tags list \
    --table mlflow --region us-east-1

# List effective patterns for a specific experiment
mlflow-dynamodbstore denormalize-tags list \
    --table mlflow --region us-east-1 \
    --experiment-id 42
```

### add

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.denormalize_tags
    :command: add_pattern
    :prog_name: mlflow-dynamodbstore denormalize-tags add

**Examples:**

```bash
# Add a global pattern
mlflow-dynamodbstore denormalize-tags add \
    --table mlflow --region us-east-1 \
    "mlflow.user"

# Add a per-experiment pattern
mlflow-dynamodbstore denormalize-tags add \
    --table mlflow --region us-east-1 \
    --experiment-id 42 \
    "team"
```

### remove

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.denormalize_tags
    :command: remove_pattern
    :prog_name: mlflow-dynamodbstore denormalize-tags remove

**Examples:**

```bash
mlflow-dynamodbstore denormalize-tags remove \
    --table mlflow --region us-east-1 \
    "mlflow.user"
```

### backfill

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.denormalize_tags
    :command: backfill
    :prog_name: mlflow-dynamodbstore denormalize-tags backfill

Scans all tag items, matches against configured patterns, and updates META items
with denormalized tag attributes. Run this after adding new patterns to
retroactively populate existing data.

**Examples:**

```bash
# Backfill all experiments
mlflow-dynamodbstore denormalize-tags backfill \
    --table mlflow --region us-east-1

# Backfill a single experiment
mlflow-dynamodbstore denormalize-tags backfill \
    --table mlflow --region us-east-1 \
    --experiment-id 42
```

!!! warning
    Backfill performs a full table scan and may consume significant read/write
    capacity. Run during off-peak hours or on tables with on-demand billing.

---

## fts-trigrams

Manage full-text search trigram field configuration. Trigrams enable
`LIKE`-style search on DynamoDB by indexing 3-character substrings.

The fields `experiment_name`, `run_name`, and `model_name` are always indexed.
Use these commands to add additional fields.

### list

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.fts_trigrams
    :command: list_fields
    :prog_name: mlflow-dynamodbstore fts-trigrams list

**Example:**

```bash
mlflow-dynamodbstore fts-trigrams list \
    --table mlflow --region us-east-1
```

### add

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.fts_trigrams
    :command: add_field
    :prog_name: mlflow-dynamodbstore fts-trigrams add

**Example:**

```bash
mlflow-dynamodbstore fts-trigrams add \
    --table mlflow --region us-east-1 \
    "description"
```

---

## ttl-policy

Manage TTL (time-to-live) retention policies. See the
[TTL Lifecycle](ttl-lifecycle.md) page for how these policies interact with
soft-delete, traces, and metric history.

### show

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.ttl_policy
    :command: show_policy
    :prog_name: mlflow-dynamodbstore ttl-policy show

**Example:**

```bash
mlflow-dynamodbstore ttl-policy show \
    --table mlflow --region us-east-1
```

Sample output:

```
metric_history_retention_days: 365 days
soft_deleted_retention_days: 90 days
trace_retention_days: 30 days
```

### set

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.ttl_policy
    :command: set_policy
    :prog_name: mlflow-dynamodbstore ttl-policy set

Set one or more retention values. Pass `0` to disable a TTL category.

**Examples:**

```bash
# Set all three policies
mlflow-dynamodbstore ttl-policy set \
    --table mlflow --region us-east-1 \
    --soft-deleted-retention-days 90 \
    --trace-retention-days 30 \
    --metric-history-retention-days 365

# Disable trace TTL
mlflow-dynamodbstore ttl-policy set \
    --table mlflow --region us-east-1 \
    --trace-retention-days 0
```

---

## cleanup-expired

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.cleanup_expired
    :command: cleanup_expired
    :prog_name: mlflow-dynamodbstore cleanup-expired

Find and expire orphaned children of TTL-deleted experiments. When DynamoDB TTL
removes an experiment's META item, child items (runs, tags, params, metrics)
remain. This command sets `ttl = now` on those orphans so DynamoDB will
garbage-collect them.

**Examples:**

```bash
# Preview what would be cleaned up
mlflow-dynamodbstore cleanup-expired \
    --table mlflow --region us-east-1 \
    --dry-run

# Run cleanup
mlflow-dynamodbstore cleanup-expired \
    --table mlflow --region us-east-1
```

!!! tip
    Run `--dry-run` first to see the scope of orphaned items before committing
    to cleanup. Schedule this command periodically (e.g. daily via cron or
    EventBridge) to keep your table lean.

---

## cache-spans

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.cache_spans
    :command: cache_spans
    :prog_name: mlflow-dynamodbstore cache-spans

Pre-cache X-Ray spans for traces. This fetches span data from AWS X-Ray and
stores it in DynamoDB so subsequent `get_trace` calls avoid hitting the X-Ray
API.

**Examples:**

```bash
# Cache spans for one experiment
mlflow-dynamodbstore cache-spans \
    --table mlflow --region us-east-1 \
    --experiment-id 1

# Cache spans for multiple experiments
mlflow-dynamodbstore cache-spans \
    --table mlflow --region us-east-1 \
    --experiment-id 1 --experiment-id 2

# Only process traces from the last 7 days
mlflow-dynamodbstore cache-spans \
    --table mlflow --region us-east-1 \
    --experiment-id 1 \
    --days 7
```

!!! note
    X-Ray retains trace data for 30 days. Run `cache-spans` before that window
    closes to preserve span details in DynamoDB.

---

## delete-workspace

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli.delete_workspace
    :command: delete_workspace
    :prog_name: mlflow-dynamodbstore delete-workspace

Delete a workspace and optionally all its experiments and models.

**Modes:**

| Mode      | Behavior                                                            |
|-----------|---------------------------------------------------------------------|
| `soft`    | Marks the workspace as deleted; data is preserved                   |
| `cascade` | Permanently deletes the workspace and all its experiments and models|

**Examples:**

```bash
# Soft-delete a workspace
mlflow-dynamodbstore delete-workspace \
    --table mlflow --region us-east-1 \
    --workspace staging --mode soft

# Cascade delete (with confirmation prompt)
mlflow-dynamodbstore delete-workspace \
    --table mlflow --region us-east-1 \
    --workspace staging --mode cascade

# Cascade delete (skip confirmation)
mlflow-dynamodbstore delete-workspace \
    --table mlflow --region us-east-1 \
    --workspace staging --mode cascade --yes
```

!!! danger
    Cascade mode permanently deletes all experiments, runs, and models in the
    workspace. This action cannot be undone. The `default` workspace cannot be
    deleted.
