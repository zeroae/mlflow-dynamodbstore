# Configuration

## URI Format

```
dynamodb://<region>/<table-name>
dynamodb://<host>:<port>/<table-name>
dynamodb://http://<host>:<port>/<table-name>
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_DYNAMODB_SOFT_DELETED_RETENTION_DAYS` | Days before soft-deleted items are TTL'd | `90` |
| `MLFLOW_DYNAMODB_TRACE_RETENTION_DAYS` | Days before trace items are TTL'd | `30` |
| `MLFLOW_DYNAMODB_METRIC_HISTORY_RETENTION_DAYS` | Days before metric history is TTL'd | `365` |
| `MLFLOW_DYNAMODB_DENORMALIZE_TAGS` | Tag patterns to denormalize on META items | `mlflow.*` |
| `MLFLOW_DYNAMODB_FTS_TRIGRAM_FIELDS` | Fields with trigram FTS enabled | _(empty)_ |
