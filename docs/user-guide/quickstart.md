# Quickstart

This guide walks you through installing mlflow-dynamodbstore, starting an MLflow server backed by DynamoDB, and running your first experiment.

## Prerequisites

- **Python 3.11+**
- **AWS credentials** configured (`aws configure` or environment variables)
- **An S3 bucket** for artifact storage
- **IAM permissions** for DynamoDB and CloudFormation (the plugin auto-provisions the table)

!!! tip "Required IAM Permissions"
    The IAM principal running the server needs:

    - `dynamodb:*` on the table resource
    - `cloudformation:CreateStack`, `cloudformation:DescribeStacks`, `cloudformation:UpdateStack`
    - `s3:*` on the artifact bucket (for MLflow artifact operations)

## Installation

```bash
uv pip install mlflow-dynamodbstore
```

This installs the plugin along with MLflow and its dependencies.

## Start the Server

```bash
mlflow server \
  --app-name dynamodb-auth \
  --backend-store-uri dynamodb://us-east-1/my-table \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

On first startup, the plugin creates a CloudFormation stack named `mlflow-dynamodbstore-my-table` that provisions:

- A DynamoDB table with 5 GSIs and 5 LSIs
- DynamoDB TTL enabled on the `ttl` attribute
- Pay-per-request billing mode

!!! note "Local Development"
    For local development with DynamoDB Local:

    ```bash
    mlflow server \
      --app-name dynamodb-auth \
      --backend-store-uri dynamodb://localhost:8000/my-table \
      --default-artifact-root ./mlartifacts
    ```

## Configure the Client

Point the MLflow client at your server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Create an Experiment

=== "CLI"

    ```bash
    mlflow experiments create --experiment-name "my-experiment"
    ```

=== "Python"

    ```python
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("my-experiment")
    ```

## Log a Run

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)
    mlflow.log_metric("loss", 0.42)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.set_tag("model_type", "xgboost")
```

## Search Runs

```python
import mlflow

runs = mlflow.search_runs(
    experiment_names=["my-experiment"],
    filter_string="params.learning_rate = '0.01'",
)
print(runs)
```

## View in the UI

Open [http://localhost:5000](http://localhost:5000) in your browser to see experiments, runs, and metrics in the MLflow UI.

## What's Next?

- **[Configuration](configuration.md)** -- URI format, environment variables, and tuning options
- **[Workspaces](workspaces.md)** -- isolate experiments and models across teams
- **[X-Ray Integration](xray-integration.md)** -- trace your LLM calls with dual-export to X-Ray
- **[CLI Reference](../operator-guide/cli-reference.md)** -- admin commands for maintenance tasks
