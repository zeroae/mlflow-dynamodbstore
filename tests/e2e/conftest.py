"""E2E test fixtures: in-process mlflow server over HTTP.

The server runs in-process via threaded uvicorn so that coverage
instrumentation captures the full request path (store, auth plugin, etc.).

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
import threading
import time

import pytest
import requests
import uvicorn
from mlflow import MlflowClient

_TABLE_NAME = "e2e-mlflow"
_REGION = "us-east-1"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: int = 60) -> None:
    """Poll server health endpoint until ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return
        except requests.ConnectionError:
            time.sleep(0.2)
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


def _configure_mlflow_env(store_uri: str) -> None:
    """Set internal env vars that MLflow's server reads at import/request time."""
    os.environ["_MLFLOW_SERVER_FILE_STORE"] = store_uri
    os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = "/tmp/mlflow-e2e-artifacts"
    os.environ["MLFLOW_FLASK_SERVER_SECRET_KEY"] = "e2e-test-secret-key"
    os.environ["MLFLOW_ENABLE_WORKSPACES"] = "true"
    # Auth credentials for the dynamodb-auth plugin (admin/password1234 defaults)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password1234"


def _create_app():
    """Build the MLflow Flask app with dynamodb-auth plugin, then wrap in FastAPI."""
    from mlflow.server.handlers import initialize_backend_stores

    store_uri = os.environ["_MLFLOW_SERVER_FILE_STORE"]
    artifact_root = os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"]

    # Prime the tracking and registry store singletons
    initialize_backend_stores(
        backend_store_uri=store_uri,
        registry_store_uri=store_uri,
        default_artifact_root=artifact_root,
    )

    # Build Flask app with auth wired in (replicates --app-name dynamodb-auth)
    from mlflow_dynamodbstore.auth.app import create_app as create_auth_app

    flask_app = create_auth_app()

    # Wrap in FastAPI (replicates what uvicorn does with mlflow.server.fastapi_app)
    from mlflow.server.fastapi_app import create_fastapi_app

    return create_fastapi_app(flask_app)


def _start_server(host: str, port: int) -> uvicorn.Server:
    """Start uvicorn in a background thread, return the server handle."""
    app = _create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


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
    return ["e2e backend: moto (in-process)"]


def _start_mlflow_moto():
    """Start moto server + in-process mlflow server."""
    from moto.server import ThreadedMotoServer

    # Start moto server
    moto_server = ThreadedMotoServer(port=0)
    moto_server.start()
    _, moto_port = moto_server.get_host_and_port()
    moto_endpoint = f"http://localhost:{moto_port}"
    print(f"\nMoto server: {moto_endpoint}")

    store_uri = f"dynamodb://{moto_endpoint}/{_TABLE_NAME}"

    # Set env before importing mlflow.server (triggers is_running_as_server)
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = _REGION
    _configure_mlflow_env(store_uri)

    mlflow_port = _find_free_port()
    server = _start_server("127.0.0.1", mlflow_port)

    tracking_uri = f"http://127.0.0.1:{mlflow_port}"
    try:
        _wait_for_server(tracking_uri)
    except RuntimeError:
        server.should_exit = True
        moto_server.stop()
        raise

    yield tracking_uri

    server.should_exit = True
    moto_server.stop()


def _delete_stack(table_name: str, region: str) -> None:
    """Delete the CloudFormation stack and wait for completion."""
    import boto3

    from mlflow_dynamodbstore.dynamodb.provisioner import get_stack_name

    cfn = boto3.client("cloudformation", region_name=region)
    stack_name = get_stack_name(table_name)
    print(f"\nDeleting CloudFormation stack: {stack_name}")
    try:
        cfn.delete_stack(StackName=stack_name)
        cfn.get_waiter("stack_delete_complete").wait(
            StackName=stack_name,
            WaiterConfig={"Delay": 5, "MaxAttempts": 120},
        )
        print(f"Stack {stack_name} deleted.")
    except Exception as exc:
        print(f"Warning: failed to delete stack {stack_name}: {exc}")


def _start_mlflow_aws():
    """Start in-process mlflow server against real AWS DynamoDB.

    Deletes the CloudFormation stack on teardown.
    """
    store_uri = f"dynamodb://{_REGION}/{_TABLE_NAME}"
    _configure_mlflow_env(store_uri)

    mlflow_port = _find_free_port()
    server = _start_server("127.0.0.1", mlflow_port)

    tracking_uri = f"http://127.0.0.1:{mlflow_port}"
    try:
        _wait_for_server(tracking_uri, timeout=180)
    except RuntimeError:
        server.should_exit = True
        _delete_stack(_TABLE_NAME, _REGION)
        raise

    yield tracking_uri

    server.should_exit = True
    _delete_stack(_TABLE_NAME, _REGION)


@pytest.fixture(scope="session")
def client(mlflow_server) -> MlflowClient:
    """Return an MlflowClient pointed at the e2e server."""
    return MlflowClient(tracking_uri=mlflow_server)


@pytest.fixture(scope="session")
def http_session(mlflow_server) -> requests.Session:
    """Return a requests.Session with base URL and auth pre-configured."""
    import base64

    username = os.environ.get("MLFLOW_TRACKING_USERNAME", "")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD", "")

    session = requests.Session()
    if username and password:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        session.headers["Authorization"] = f"Basic {token}"

    # Store base URL for convenience
    session.base_url = mlflow_server  # type: ignore[attr-defined]
    return session
