# mlflow-dynamodbstore

DynamoDB-backed MLflow tracking store, model registry, and auth plugin.

## Features

- **Tracking Store** — experiments, runs, metrics, params, tags
- **Model Registry** — registered models, versions, aliases
- **Auth Plugin** — users, permissions
- **Workspace Provider** — multi-workspace support

## Installation

```bash
pip install mlflow-dynamodbstore
```

## Quick Start

```bash
mlflow server \
  --app-name dynamodb-auth \
  --backend-store-uri dynamodb://us-east-1/my-table \
  --default-artifact-root s3://bucket/artifacts
```

The DynamoDB table is auto-provisioned via CloudFormation on first connection.

See the [Quickstart](user-guide/quickstart.md) for details.
