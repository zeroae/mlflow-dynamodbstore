import pytest


@pytest.fixture
def auth_store(mock_dynamodb):
    from mlflow_dynamodbstore.auth.store import DynamoDBAuthStore

    store = DynamoDBAuthStore(store_uri="dynamodb://us-east-1/test-table")
    return store
