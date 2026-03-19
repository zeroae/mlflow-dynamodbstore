"""Fixtures for compatibility tests between DynamoDB and SqlAlchemy stores."""

import sys
from collections import namedtuple
from pathlib import Path

import pytest
from moto import mock_aws

# Path to vendored MLflow tests for extending tests.__path__ below.
# IMPORTANT: vendor/mlflow/tests/ must NEVER be added to pytest's testpaths.
_VENDOR_MLFLOW_TESTS = str(Path(__file__).parents[2] / "vendor" / "mlflow" / "tests")

# With --import-mode=importlib, pytest registers our local `tests/` package in
# sys.modules before conftest runs, so `from tests.store.*` fails because our
# `tests` package has no `store` subpackage. Fix: extend `tests.__path__` to
# include vendor/mlflow/tests so that `tests.store.*` resolves to the vendor copy.
#
# We also extend any `tests.*` sub-packages that are already registered but whose
# __path__ does not yet include the vendor counterpart (e.g. tests.integration
# exists locally but lacks the vendor tests/integration/utils.py).
import tests as _our_tests_pkg  # noqa: E402

if _VENDOR_MLFLOW_TESTS not in _our_tests_pkg.__path__:
    _our_tests_pkg.__path__.append(_VENDOR_MLFLOW_TESTS)

# Pre-register vendor tests sub-packages that also exist locally so that imports
# resolve from BOTH locations (ours first, then vendor's as fallback).
# Without this, `from tests.integration.utils import ...` fails because our local
# tests/integration/ has no utils.py — Python finds our package first and stops.
import importlib.util as _ilu  # noqa: E402
import types as _types  # noqa: E402

_VENDOR_TESTS_PATH = Path(_VENDOR_MLFLOW_TESTS)


def _ensure_vendor_subpkg(mod_name: str) -> None:
    """Ensure mod_name's __path__ includes the vendor counterpart directory."""
    # Convert "tests.integration" -> Path("integration"), "tests" -> Path("")
    rel_parts = mod_name.split(".")[
        1:
    ]  # e.g. [] for "tests", ["integration"] for "tests.integration"
    rel_path = Path(*rel_parts) if rel_parts else Path("")
    vendor_sub = str(_VENDOR_TESTS_PATH / rel_path)
    if not Path(vendor_sub).is_dir():
        return
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if hasattr(mod, "__path__") and vendor_sub not in mod.__path__:
            mod.__path__.append(vendor_sub)
    else:
        # Pre-register as a namespace/regular package with combined __path__
        local_root = Path(_our_tests_pkg.__file__).parent
        local_sub = str(local_root / rel_path)
        search_paths = []
        if Path(local_sub).is_dir():
            search_paths.append(local_sub)
        search_paths.append(vendor_sub)
        pkg_init = Path(local_sub) / "__init__.py"
        if not pkg_init.exists():
            pkg_init = Path(vendor_sub) / "__init__.py"
        spec = _ilu.spec_from_file_location(
            mod_name,
            str(pkg_init) if pkg_init.exists() else None,
            submodule_search_locations=search_paths,
        )
        mod = _types.ModuleType(mod_name)
        mod.__path__ = search_paths  # type: ignore[assignment]
        mod.__spec__ = spec  # type: ignore[assignment]
        mod.__package__ = mod_name
        sys.modules[mod_name] = mod


# Pre-register known vendor sub-packages that conflict with our local tests.*
for _subpkg in ["tests.integration", "tests.helper_functions"]:
    _ensure_vendor_subpkg(_subpkg)

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


# --- SqlAlchemy store fixtures (in-memory SQLite, for Phase 1 comparison) ---
# Each fixture uses a unique named in-memory DB via file URI to avoid sharing
# state through SqlAlchemyStore's class-level engine cache (_engine_map).
_sql_counter = 0


def _unique_memory_uri() -> str:
    global _sql_counter
    _sql_counter += 1
    return f"sqlite:///file:mem{_sql_counter}?mode=memory&cache=shared&uri=true"


@pytest.fixture
def sql_tracking_store(tmp_path):
    artifact_uri = str(tmp_path / "artifacts")
    return SqlAlchemyTrackingStore(_unique_memory_uri(), artifact_uri)


@pytest.fixture
def sql_registry_store():
    return SqlAlchemyRegistryStore(_unique_memory_uri())


@pytest.fixture
def sql_workspace_store(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    return SqlAlchemyWorkspaceStore(_unique_memory_uri())


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
