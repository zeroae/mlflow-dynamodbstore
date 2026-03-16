"""E2E test fixtures: mlflow server over HTTP.

Supports three modes:
1. MLFLOW_TRACKING_URI set: uses existing server (fastest for iteration)
2. Default: starts moto server + mlflow subprocess (no AWS credentials needed)
3. E2E_USE_AWS=1: starts mlflow subprocess against real AWS DynamoDB

Local usage (pre-started server):
    MLFLOW_TRACKING_URI=http://127.0.0.1:15123 uv run pytest tests/e2e/ -m e2e -v

Local usage (moto, no AWS needed):
    uv run pytest tests/e2e/ -m e2e -v

Local usage (real AWS):
    E2E_USE_AWS=1 AWS_PROFILE=zeroae-code/AWSPowerUserAccess uv run pytest tests/e2e/ -m e2e -v
"""

import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest
import requests
from mlflow import MlflowClient

_TABLE_NAME = "e2e-mlflow"
_REGION = "us-east-1"
_SERVER_LOG = os.path.join(tempfile.gettempdir(), "mlflow-e2e-server.log")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: int = 180) -> None:
    """Poll server health endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.ConnectionError:
            time.sleep(1)
    raise RuntimeError(f"Server at {url} did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def mlflow_server():
    """Return tracking URI for an mlflow server.

    If MLFLOW_TRACKING_URI is set, uses the existing server.
    If E2E_USE_AWS is set, starts mlflow against real AWS.
    Otherwise, starts moto server + mlflow subprocess.
    """
    existing_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if existing_uri:
        try:
            resp = requests.get(f"{existing_uri}/health", timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            pytest.fail(f"MLFLOW_TRACKING_URI={existing_uri} is not reachable: {exc}")
        yield existing_uri
        return

    use_aws = os.environ.get("E2E_USE_AWS", "").strip() == "1"

    if use_aws:
        yield from _start_mlflow_aws()
    else:
        yield from _start_mlflow_moto()


def _start_mlflow_moto():
    """Start moto server + mlflow subprocess."""
    from moto.server import ThreadedMotoServer

    # Start moto server
    moto_server = ThreadedMotoServer(port=0)
    moto_server.start()
    _, moto_port = moto_server.get_host_and_port()
    moto_endpoint = f"http://localhost:{moto_port}"
    print(f"\nMoto server: {moto_endpoint}")

    # Start mlflow server pointing at moto
    mlflow_port = _find_free_port()
    store_uri = f"dynamodb://{moto_endpoint}/{_TABLE_NAME}"

    env = os.environ.copy()
    env["MLFLOW_FLASK_SERVER_SECRET_KEY"] = "e2e-test-secret-key"
    env["AWS_ACCESS_KEY_ID"] = "testing"
    env["AWS_SECRET_ACCESS_KEY"] = "testing"
    env["AWS_DEFAULT_REGION"] = _REGION

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
            str(mlflow_port),
            "--workers",
            "1",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    tracking_uri = f"http://127.0.0.1:{mlflow_port}"

    try:
        _wait_for_server(tracking_uri, timeout=60)
    except RuntimeError:
        proc.kill()
        log_file.close()
        moto_server.stop()
        with open(_SERVER_LOG) as f:
            raise RuntimeError(f"MLflow server failed to start.\nLog:\n{f.read()}")

    yield tracking_uri

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()
    moto_server.stop()


def _start_mlflow_aws():
    """Start mlflow subprocess against real AWS DynamoDB."""
    import boto3

    # Verify credentials
    sts = boto3.client("sts", region_name=_REGION)
    try:
        identity = sts.get_caller_identity()
        print(f"\nAWS identity: {identity['Arn']}")
    except Exception as exc:
        pytest.skip(f"AWS credentials not available: {exc}")

    mlflow_port = _find_free_port()
    store_uri = f"dynamodb://{_REGION}/{_TABLE_NAME}"

    env = os.environ.copy()
    env["MLFLOW_FLASK_SERVER_SECRET_KEY"] = "e2e-test-secret-key"

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
            str(mlflow_port),
            "--workers",
            "1",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    tracking_uri = f"http://127.0.0.1:{mlflow_port}"

    try:
        _wait_for_server(tracking_uri, timeout=180)
    except RuntimeError:
        proc.kill()
        log_file.close()
        with open(_SERVER_LOG) as f:
            raise RuntimeError(f"MLflow server failed to start.\nLog:\n{f.read()}")

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
