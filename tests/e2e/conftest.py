"""E2E test fixtures: mlflow server over HTTP.

Backend selection:
1. MLFLOW_TRACKING_URI set → uses existing server (fastest for iteration)
2. AWS credentials available (AWS_PROFILE or env vars) → real AWS DynamoDB
3. No credentials → moto server (no AWS needed, default in CI)

Local usage (pre-started server):
    MLFLOW_TRACKING_URI=http://127.0.0.1:15123 uv run pytest tests/e2e/ -m e2e -v

Local usage (real AWS, auto-detected from AWS_PROFILE):
    AWS_PROFILE=zeroae-code/AWSPowerUserAccess uv run pytest tests/e2e/ -m e2e -v

Local usage (moto fallback, no AWS):
    uv run pytest tests/e2e/ -m e2e -v
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


def _has_aws_credentials() -> bool:
    """Check if real AWS credentials are available."""
    import boto3

    try:
        sts = boto3.client("sts", region_name=_REGION)
        identity = sts.get_caller_identity()
        print(f"\nAWS identity: {identity['Arn']}")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def mlflow_server():
    """Return tracking URI for an mlflow server.

    If MLFLOW_TRACKING_URI is set, uses the existing server.
    If AWS credentials are available, uses real AWS DynamoDB.
    Otherwise, falls back to moto server.
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

    if _has_aws_credentials():
        yield from _start_mlflow_aws()
    else:
        yield from _start_mlflow_moto()


def pytest_report_header():
    """Show which backend will be used in the test header."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return [f"e2e backend: existing server at {os.environ['MLFLOW_TRACKING_URI']}"]
    if _has_aws_credentials():
        return ["e2e backend: real AWS DynamoDB"]
    return ["e2e backend: moto server"]


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
