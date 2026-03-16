# Quickstart

## Prerequisites

- Python 3.11+
- AWS credentials configured
- An S3 bucket for artifacts

## Installation

```bash
pip install mlflow-dynamodbstore
```

## Start the Server

```bash
mlflow server \
  --app-name dynamodb-auth \
  --backend-store-uri dynamodb://us-east-1/my-table \
  --default-artifact-root s3://my-bucket/mlflow-artifacts
```

On first startup, the plugin creates a CloudFormation stack (`mlflow-dynamodbstore-my-table`) that provisions the DynamoDB table with all required indexes.
