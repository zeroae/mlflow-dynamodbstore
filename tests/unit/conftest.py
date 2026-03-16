import pytest
from moto import mock_aws


@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        yield


@pytest.fixture
def tracking_store(mock_dynamodb):
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    store = DynamoDBTrackingStore(
        store_uri="dynamodb://us-east-1/test-table",
        artifact_uri="/tmp/artifacts",
    )
    return store


@pytest.fixture
def registry_store(mock_dynamodb):
    from mlflow_dynamodbstore.registry_store import DynamoDBRegistryStore

    store = DynamoDBRegistryStore(
        store_uri="dynamodb://us-east-1/test-table",
    )
    return store


@pytest.fixture
def workspace_store(mock_dynamodb):
    from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore

    store = DynamoDBWorkspaceStore(
        store_uri="dynamodb://us-east-1/test-table",
    )
    return store
