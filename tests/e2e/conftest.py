"""E2E test fixtures: real AWS DynamoDB + mlflow server subprocess.

Requires AWS credentials configured via AWS_PROFILE or environment variables.
Creates a CloudFormation stack on setup, deletes it on teardown.

Local usage:
    AWS_PROFILE=zeroae-code/AWSPowerUserAccess uv run pytest tests/e2e/ -m e2e -v
"""

import os
import socket
import subprocess
import sys
import time

import boto3
import pytest
import requests
from mlflow import MlflowClient

from mlflow_dynamodbstore.dynamodb.provisioner import get_stack_name

# Unique table name per test session to avoid collisions
_TABLE_NAME = f"e2e-mlflow-{int(time.time())}"
_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _delete_stack(table_name: str, region: str) -> None:
    """Delete the CloudFormation stack and wait for completion."""
    cfn = boto3.client("cloudformation", region_name=region)
    stack_name = get_stack_name(table_name)
    try:
        cfn.delete_stack(StackName=stack_name)
        cfn.get_waiter("stack_delete_complete").wait(
            StackName=stack_name,
            WaiterConfig={"Delay": 5, "MaxAttempts": 60},
        )
    except Exception as exc:
        print(f"Warning: failed to delete stack {stack_name}: {exc}")


@pytest.fixture(scope="session", autouse=True)
def verify_aws_credentials():
    """Fail fast if AWS credentials are not configured."""
    sts = boto3.client("sts", region_name=_REGION)
    try:
        identity = sts.get_caller_identity()
    except Exception as exc:
        pytest.skip(f"AWS credentials not available: {exc}")
    print(f"\nAWS identity: {identity['Arn']}")


@pytest.fixture(scope="session")
def mlflow_server(verify_aws_credentials):
    """Start an mlflow server backed by real AWS DynamoDB.

    Creates the CloudFormation stack on startup, tears it down on exit.
    """
    port = _find_free_port()
    env = os.environ.copy()
    env["MLFLOW_FLASK_SERVER_SECRET_KEY"] = "e2e-test-secret-key"

    store_uri = f"dynamodb://{_REGION}/{_TABLE_NAME}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            store_uri,
            "--default-artifact-root",
            "/tmp/mlflow-e2e-artifacts",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    tracking_uri = f"http://127.0.0.1:{port}"

    # Wait for server to respond (CFN stack creation can take 1-2 min)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            resp = requests.get(f"{tracking_uri}/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        proc.kill()
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        _delete_stack(_TABLE_NAME, _REGION)
        raise RuntimeError(f"MLflow server did not start within 180s.\nOutput:\n{stdout}")

    yield tracking_uri

    # Teardown: stop server, delete CloudFormation stack
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    _delete_stack(_TABLE_NAME, _REGION)


@pytest.fixture(scope="session")
def client(mlflow_server) -> MlflowClient:
    """Return an MlflowClient pointed at the e2e server."""
    return MlflowClient(tracking_uri=mlflow_server)
