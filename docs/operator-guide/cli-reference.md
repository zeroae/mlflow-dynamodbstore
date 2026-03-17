# CLI Reference

The `mlflow-dynamodbstore` CLI provides admin commands for managing your
DynamoDB-backed MLflow deployment.

```bash
mlflow-dynamodbstore --help
```

## Global Options

Every subcommand inherits:

| Option           | Description                          | Required |
|------------------|--------------------------------------|----------|
| `--name`         | Stack/table name                     | Yes      |
| `--region`       | AWS region (e.g. `us-east-1`)        | Yes      |
| `--endpoint-url` | Custom endpoint (LocalStack/testing) | No       |

---

## Commands

::: mkdocs-click
    :module: mlflow_dynamodbstore.cli
    :command: cli
    :prog_name: mlflow-dynamodbstore
    :depth: 2
    :style: table
    :list_subcommands: true

---

## Examples

### Deploy a new stack

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 deploy
```

### Destroy a stack (retain table)

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 destroy --yes --retain
```

### Manage TTL policies

```bash
# Show current policy
mlflow-dynamodbstore --name mlflow --region us-east-1 ttl show

# Set retention
mlflow-dynamodbstore --name mlflow --region us-east-1 ttl set \
    --soft-deleted-retention-days 90 \
    --trace-retention-days 30

# Cleanup orphaned items
mlflow-dynamodbstore --name mlflow --region us-east-1 ttl cleanup --dry-run
```

### Manage tag denormalization

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 tag list
mlflow-dynamodbstore --name mlflow --region us-east-1 tag add "mlflow.user"
mlflow-dynamodbstore --name mlflow --region us-east-1 tag backfill
```

### Full-text search configuration

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 fts list
mlflow-dynamodbstore --name mlflow --region us-east-1 fts add "description"
```

### Cache trace spans

```bash
mlflow-dynamodbstore --name mlflow --region us-east-1 trace cache \
    --experiment-id 1 --days 7
```

### Delete a workspace

```bash
# Soft delete
mlflow-dynamodbstore --name mlflow --region us-east-1 workspace delete \
    --workspace staging --mode soft

# Cascade (with confirmation)
mlflow-dynamodbstore --name mlflow --region us-east-1 workspace delete \
    --workspace staging --mode cascade --yes
```

!!! danger
    `destroy` permanently deletes the CloudFormation stack. `workspace delete --mode cascade`
    permanently removes all data. These actions cannot be undone.

!!! tip
    Use `--retain` with `destroy` to remove the CloudFormation stack while keeping the
    DynamoDB table and its data intact.
