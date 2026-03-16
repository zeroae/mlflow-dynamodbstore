"""REST client for DynamoDB-backed MLflow auth."""

from mlflow.server.auth.client import AuthServiceClient


class DynamoDBAuthClient(AuthServiceClient):
    """REST client for DynamoDB-backed MLflow authentication.

    This is a thin wrapper around MLflow's :class:`AuthServiceClient`.
    The server-side :class:`~mlflow_dynamodbstore.auth.store.DynamoDBAuthStore`
    handles all DynamoDB operations; the client simply delegates HTTP requests
    to the standard MLflow auth REST endpoints.
    """
