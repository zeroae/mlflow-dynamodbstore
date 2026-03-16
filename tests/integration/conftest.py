import os

import pytest
import requests
from moto.server import ThreadedMotoServer


@pytest.fixture(scope="session", autouse=True)
def aws_credentials():
    """Set fake AWS credentials for the moto server (required by botocore)."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(scope="session")
def moto_endpoint(aws_credentials):
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://localhost:{port}"
    server.stop()


@pytest.fixture(autouse=True)
def reset_moto(moto_endpoint):
    yield
    requests.post(f"{moto_endpoint}/moto-api/reset")


@pytest.fixture
def tracking_store(moto_endpoint):
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    store = DynamoDBTrackingStore(
        store_uri=f"dynamodb://{moto_endpoint}/test-table",
        artifact_uri="/tmp/artifacts",
    )
    return store


@pytest.fixture
def registry_store(moto_endpoint):
    from mlflow_dynamodbstore.registry_store import DynamoDBRegistryStore

    store = DynamoDBRegistryStore(
        store_uri=f"dynamodb://{moto_endpoint}/test-table",
    )
    return store


@pytest.fixture
def workspace_store(moto_endpoint):
    from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore

    store = DynamoDBWorkspaceStore(store_uri=f"dynamodb://{moto_endpoint}/test-table")
    return store
