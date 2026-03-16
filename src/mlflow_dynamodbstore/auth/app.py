"""Flask app plugin that integrates DynamoDBAuthStore with MLflow's auth system."""

from __future__ import annotations

import logging
import os

from flask import Flask

from mlflow_dynamodbstore.auth.store import DynamoDBAuthStore

_logger = logging.getLogger(__name__)

_MLFLOW_AUTH_ADMIN_USERNAME = "MLFLOW_AUTH_ADMIN_USERNAME"
_MLFLOW_AUTH_ADMIN_PASSWORD = "MLFLOW_AUTH_ADMIN_PASSWORD"
# MLflow's server passes the backend store URI via this internal env var
_MLFLOW_SERVER_FILE_STORE = "_MLFLOW_SERVER_FILE_STORE"
_MLFLOW_BACKEND_STORE_URI = "MLFLOW_BACKEND_STORE_URI"

_DEFAULT_ADMIN_USERNAME = "admin"
_DEFAULT_ADMIN_PASSWORD = "password1234"


def create_app(app: Flask | None = None) -> Flask:
    """Initialize DynamoDB auth on an existing MLflow Flask app.

    This function replaces MLflow's default SqlAlchemyStore-based auth
    with a DynamoDBAuthStore. It:

    1. Reads the DynamoDB URI from ``MLFLOW_BACKEND_STORE_URI``
    2. Creates a :class:`DynamoDBAuthStore`
    3. Monkey-patches it into ``mlflow.server.auth`` as the module-level ``store``
    4. Delegates to MLflow's own ``create_app`` for route/hook registration

    Environment variables:
        MLFLOW_BACKEND_STORE_URI: DynamoDB URI (e.g. ``dynamodb://us-east-1/my-table``)
        MLFLOW_AUTH_ADMIN_USERNAME: Admin username (default: ``admin``)
        MLFLOW_AUTH_ADMIN_PASSWORD: Admin password (default: ``password1234``)
    """
    import mlflow.server.auth as auth_module

    if app is None:
        from mlflow.server import app as default_app

        app = default_app

    # MLflow server sets _MLFLOW_SERVER_FILE_STORE in worker subprocesses;
    # fall back to MLFLOW_BACKEND_STORE_URI for direct usage
    store_uri = os.environ.get(_MLFLOW_SERVER_FILE_STORE, "") or os.environ.get(
        _MLFLOW_BACKEND_STORE_URI, ""
    )
    if not store_uri.startswith("dynamodb://"):
        raise ValueError(
            f"Backend store URI must start with 'dynamodb://', got: {store_uri!r}. "
            f"Set _MLFLOW_SERVER_FILE_STORE or MLFLOW_BACKEND_STORE_URI."
        )

    # Create our DynamoDB-backed auth store
    dynamodb_store = DynamoDBAuthStore(store_uri)

    # Replace the module-level store so all of MLflow's auth functions use it
    auth_module.store = dynamodb_store  # type: ignore[assignment]

    # Create admin user
    admin_username = os.environ.get(_MLFLOW_AUTH_ADMIN_USERNAME, _DEFAULT_ADMIN_USERNAME)
    admin_password = os.environ.get(_MLFLOW_AUTH_ADMIN_PASSWORD, _DEFAULT_ADMIN_PASSWORD)

    # Ensure admin user exists (idempotent across concurrent workers)
    try:
        dynamodb_store.create_user(admin_username, admin_password, is_admin=True)
        _logger.info("Created admin user '%s'.", admin_username)
    except Exception:
        # User already exists (race between workers) — safe to ignore
        pass

    # Delegate to MLflow's own create_app for route and hook registration.
    # Since we already replaced the store, MLflow's create_app will use our
    # DynamoDBAuthStore for its store.init_db() call (which is a no-op) and
    # all subsequent operations.
    return auth_module.create_app(app)  # type: ignore[no-any-return]
