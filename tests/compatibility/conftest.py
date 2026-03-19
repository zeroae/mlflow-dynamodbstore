"""Fixtures for compatibility tests between DynamoDB and SqlAlchemy stores."""

import sys
from collections import namedtuple
from pathlib import Path

import pytest
from moto import mock_aws

# Add vendor/mlflow to sys.path so MLflow test helpers are importable.
# This enables `from tests.helper_functions import random_str`.
# IMPORTANT: vendor/mlflow/tests/ must NEVER be added to pytest's testpaths.
_VENDOR_MLFLOW = str(Path(__file__).parents[2] / "vendor" / "mlflow")
if _VENDOR_MLFLOW not in sys.path:
    sys.path.insert(0, _VENDOR_MLFLOW)

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES  # noqa: E402
from mlflow.store.model_registry.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyStore as SqlAlchemyRegistryStore,
)
from mlflow.store.tracking.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyStore as SqlAlchemyTrackingStore,
)
from mlflow.store.workspace.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyStore as SqlAlchemyWorkspaceStore,
)

from mlflow_dynamodbstore.registry_store import DynamoDBRegistryStore  # noqa: E402
from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore  # noqa: E402
from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore  # noqa: E402

# Named tuples for Phase 1 contract tests
TrackingStores = namedtuple("TrackingStores", ["ddb", "sql"])
RegistryStores = namedtuple("RegistryStores", ["ddb", "sql"])
WorkspaceStores = namedtuple("WorkspaceStores", ["ddb", "sql"])


# --- DynamoDB store fixtures (moto-backed) ---


@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        yield


@pytest.fixture
def tracking_store(mock_dynamodb):
    return DynamoDBTrackingStore(
        store_uri="dynamodb://us-east-1/test-table",
        artifact_uri="/tmp/artifacts",
    )


@pytest.fixture
def registry_store(mock_dynamodb):
    return DynamoDBRegistryStore(
        store_uri="dynamodb://us-east-1/test-table",
    )


@pytest.fixture
def workspace_store(mock_dynamodb):
    return DynamoDBWorkspaceStore(
        store_uri="dynamodb://us-east-1/test-table",
    )


# --- SqlAlchemy store fixtures (SQLite-backed, for Phase 1 comparison) ---


@pytest.fixture
def sql_tracking_store(tmp_path):
    artifact_uri = str(tmp_path / "artifacts")
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return SqlAlchemyTrackingStore(db_uri, artifact_uri)


@pytest.fixture
def sql_registry_store(tmp_path):
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return SqlAlchemyRegistryStore(db_uri)


@pytest.fixture
def sql_workspace_store(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return SqlAlchemyWorkspaceStore(db_uri)


# --- Phase 1: Side-by-side store pairs ---


@pytest.fixture
def tracking_stores(tracking_store, sql_tracking_store):
    return TrackingStores(ddb=tracking_store, sql=sql_tracking_store)


@pytest.fixture
def registry_stores(registry_store, sql_registry_store):
    return RegistryStores(ddb=registry_store, sql=sql_registry_store)


@pytest.fixture
def workspace_stores(workspace_store, sql_workspace_store):
    return WorkspaceStores(ddb=workspace_store, sql=sql_workspace_store)


# --- Phase 2a: Alias fixtures for MLflow test function compatibility ---
# MLflow tests expect a fixture named `store`.
# Each compat test file should override this via its own conftest or
# by importing from here. The `store` fixture below defaults to registry
# but Phase 2a test files may need per-file conftest.py overrides.


@pytest.fixture
def store(registry_store):
    """Default store alias for Phase 2a MLflow test imports."""
    return registry_store


# --- Phase 2b: Workspace-enabled fixtures ---
# These names are LOAD-BEARING — they must exactly match the fixture names
# in MLflow's workspace test files (test_sqlalchemy_workspace_store.py).


@pytest.fixture
def workspace_tracking_store(monkeypatch, tracking_store):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    return tracking_store


@pytest.fixture
def workspace_registry_store(monkeypatch, registry_store):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    return registry_store
