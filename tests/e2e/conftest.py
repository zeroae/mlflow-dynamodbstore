"""E2E test fixtures: real AWS DynamoDB + mlflow server.

If MLFLOW_TRACKING_URI is set, uses the existing server.
Otherwise, starts a server subprocess backed by real DynamoDB.

Local usage (existing server):
    # Terminal 1: start server
    AWS_PROFILE=zeroae-code/AWSPowerUserAccess MLFLOW_FLASK_SERVER_SECRET_KEY=secret \\
      mlflow server --backend-store-uri dynamodb://us-east-1/e2e-mlflow \\
      --default-artifact-root /tmp/mlflow-e2e-artifacts --port 15123 --workers 1

    # Terminal 2: run tests
    MLFLOW_TRACKING_URI=http://127.0.0.1:15123 uv run pytest tests/e2e/ -m e2e -v

Local usage (auto-start):
    AWS_PROFILE=zeroae-code/AWSPowerUserAccess uv run pytest tests/e2e/ -m e2e -v
"""

import os
import socket
import subprocess
import sys
import tempfile
import time

import boto3
import pytest
import requests
from mlflow import MlflowClient

_TABLE_NAME = "e2e-mlflow"
_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
_SERVER_LOG = os.path.join(tempfile.gettempdir(), "mlflow-e2e-server.log")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    """Return tracking URI for an mlflow server.

    If MLFLOW_TRACKING_URI is set, uses the existing server.
    Otherwise, starts a subprocess backed by real AWS DynamoDB.
    """
    existing_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if existing_uri:
        # Verify the existing server is reachable
        try:
            resp = requests.get(f"{existing_uri}/health", timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            pytest.fail(f"MLFLOW_TRACKING_URI={existing_uri} is not reachable: {exc}")
        yield existing_uri
        return

    # Start a server subprocess
    port = _find_free_port()
    env = os.environ.copy()
    env["MLFLOW_FLASK_SERVER_SECRET_KEY"] = "e2e-test-secret-key"
    store_uri = f"dynamodb://{_REGION}/{_TABLE_NAME}"

    log_file = open(_SERVER_LOG, "w")
    print(f"Server log: {_SERVER_LOG}")

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
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    tracking_uri = f"http://127.0.0.1:{port}"

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
        log_file.close()
        with open(_SERVER_LOG) as f:
            log_content = f.read()
        raise RuntimeError(f"MLflow server did not start within 180s.\nLog:\n{log_content}")

    yield tracking_uri

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()


@pytest.fixture(scope="session")
def client(mlflow_server) -> MlflowClient:
    """Return an MlflowClient pointed at the e2e server."""
    return MlflowClient(tracking_uri=mlflow_server)
