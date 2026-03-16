"""Tests for DynamoDBAuthClient."""

from mlflow.server.auth.client import AuthServiceClient

from mlflow_dynamodbstore.auth.client import DynamoDBAuthClient


class TestDynamoDBAuthClient:
    """Unit tests for the DynamoDBAuthClient."""

    def test_is_subclass_of_auth_service_client(self):
        assert issubclass(DynamoDBAuthClient, AuthServiceClient)

    def test_instantiation(self):
        client = DynamoDBAuthClient("http://localhost:5000")
        assert client.tracking_uri == "http://localhost:5000"

    def test_inherits_all_public_methods(self):
        """Verify that DynamoDBAuthClient exposes the same public API."""
        parent_methods = {
            name
            for name in dir(AuthServiceClient)
            if not name.startswith("_") and callable(getattr(AuthServiceClient, name))
        }
        client_methods = {
            name
            for name in dir(DynamoDBAuthClient)
            if not name.startswith("_") and callable(getattr(DynamoDBAuthClient, name))
        }
        assert parent_methods <= client_methods

    def test_entry_point_loadable(self):
        """Verify the entry point can be discovered."""
        from importlib.metadata import entry_points

        eps = entry_points(group="mlflow.app.client")
        names = [ep.name for ep in eps]
        assert "dynamodb-auth" in names

        ep = next(ep for ep in eps if ep.name == "dynamodb-auth")
        loaded = ep.load()
        assert loaded is DynamoDBAuthClient
