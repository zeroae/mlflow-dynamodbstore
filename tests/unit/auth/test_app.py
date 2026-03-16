"""Tests for the DynamoDB auth Flask app plugin."""

from __future__ import annotations

import pytest
from flask import Flask

from mlflow_dynamodbstore.auth.app import (
    _DEFAULT_ADMIN_USERNAME,
    _MLFLOW_AUTH_ADMIN_PASSWORD,
    _MLFLOW_AUTH_ADMIN_USERNAME,
    _MLFLOW_BACKEND_STORE_URI,
    create_app,
)
from mlflow_dynamodbstore.auth.store import DynamoDBAuthStore


@pytest.fixture
def flask_app():
    """A minimal Flask app for testing."""
    return Flask(__name__)


@pytest.fixture
def env_vars(monkeypatch):
    """Set default environment variables for DynamoDB auth."""
    monkeypatch.setenv(_MLFLOW_BACKEND_STORE_URI, "dynamodb://us-east-1/test-table")
    monkeypatch.setenv("MLFLOW_FLASK_SERVER_SECRET_KEY", "test-secret-key")


class TestCreateApp:
    def test_create_app_returns_flask_app(self, mock_dynamodb, flask_app, env_vars):
        """create_app(app) returns a Flask app."""
        result = create_app(flask_app)
        assert isinstance(result, Flask)

    def test_create_app_creates_admin_user(self, mock_dynamodb, flask_app, env_vars):
        """Admin user is created from env vars with default credentials."""
        create_app(flask_app)

        import mlflow.server.auth as auth_module

        store = auth_module.store
        assert isinstance(store, DynamoDBAuthStore)

        user = store.get_user(_DEFAULT_ADMIN_USERNAME)
        assert user.username == _DEFAULT_ADMIN_USERNAME
        assert user.is_admin is True

    def test_create_app_uses_dynamodb_store(self, mock_dynamodb, flask_app, env_vars):
        """The store is a DynamoDBAuthStore after create_app runs."""
        create_app(flask_app)

        import mlflow.server.auth as auth_module

        assert isinstance(auth_module.store, DynamoDBAuthStore)

    def test_create_app_custom_admin_credentials(
        self, mock_dynamodb, flask_app, env_vars, monkeypatch
    ):
        """Admin user uses custom credentials from env vars."""
        monkeypatch.setenv(_MLFLOW_AUTH_ADMIN_USERNAME, "custom-admin")
        monkeypatch.setenv(_MLFLOW_AUTH_ADMIN_PASSWORD, "custom-pass")

        create_app(flask_app)

        import mlflow.server.auth as auth_module

        store = auth_module.store
        user = store.get_user("custom-admin")
        assert user.username == "custom-admin"
        assert user.is_admin is True

    def test_create_app_requires_dynamodb_uri(self, flask_app, monkeypatch):
        """create_app raises ValueError when URI is not a dynamodb:// URI."""
        monkeypatch.setenv(_MLFLOW_BACKEND_STORE_URI, "sqlite:///mlflow.db")
        with pytest.raises(ValueError, match="must start with 'dynamodb://'"):
            create_app(flask_app)

    def test_create_app_missing_uri(self, flask_app, monkeypatch):
        """create_app raises ValueError when MLFLOW_BACKEND_STORE_URI is not set."""
        monkeypatch.delenv(_MLFLOW_BACKEND_STORE_URI, raising=False)
        with pytest.raises(ValueError, match="must start with 'dynamodb://'"):
            create_app(flask_app)

    def test_create_app_idempotent_admin_user(self, mock_dynamodb, flask_app, env_vars):
        """Calling create_app twice does not fail on existing admin user."""
        create_app(flask_app)

        # Create a second app and call create_app again — should not raise
        flask_app2 = Flask(__name__)
        # We need to call create_app again; the admin user already exists
        # This should succeed without error
        create_app(flask_app2)

    def test_create_app_registers_before_request(self, mock_dynamodb, flask_app, env_vars):
        """create_app registers before_request hooks on the Flask app."""
        result = create_app(flask_app)
        # MLflow's create_app registers before_request and after_request hooks
        assert len(result.before_request_funcs.get(None, [])) > 0

    def test_create_app_registers_auth_routes(self, mock_dynamodb, flask_app, env_vars):
        """create_app registers auth-related URL rules."""
        result = create_app(flask_app)
        # Check that some auth routes are registered
        rules = [rule.rule for rule in result.url_map.iter_rules()]
        # MLflow registers user management routes
        assert any("/users" in rule for rule in rules)
