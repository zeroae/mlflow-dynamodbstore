"""Phase 5: MLflow gateway store tests run against DynamoDB.

Test functions are imported from the vendored MLflow test suite.
The module-level `store` fixture overrides conftest's default (registry_store)
to use the tracking_store instead.

Excluded tests (11 total):
- test_secret_id_and_name_are_immutable_at_database_level: uses sqlalchemy.text()
  and ManagedSessionMaker to test SQL column triggers
- 10 config_resolver tests (test_get_resource_gateway_endpoint_configs,
  test_get_resource_endpoint_configs_*, test_get_gateway_endpoint_config*):
  config_resolver.py does isinstance(store, SqlAlchemyStore) check and uses
  ORM queries with ManagedSessionMaker
"""

import uuid

import pytest
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.utils.workspace_context import WorkspaceContext


@pytest.fixture
def store(tracking_store):
    return tracking_store


@pytest.fixture(autouse=True)
def set_kek_passphrase(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase-for-gateway-tests")


@pytest.fixture(autouse=True, params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def workspaces_enabled(request, monkeypatch):
    enabled = request.param
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true" if enabled else "false")
    if enabled:
        workspace_name = f"gateway-test-{uuid.uuid4().hex}"
        with WorkspaceContext(workspace_name):
            yield enabled
    else:
        yield enabled


# =============================================================================
# Secret Operations (13 tests)
# =============================================================================
# =============================================================================
# Model Definition Operations (10 tests)
# =============================================================================
# =============================================================================
# Endpoint Operations (10 tests)
# =============================================================================
# =============================================================================
# Model Attachment (4 tests)
# =============================================================================
# =============================================================================
# Bindings (3 tests)
# =============================================================================
# =============================================================================
# Tags (8 tests)
# =============================================================================
# =============================================================================
# Scorer Integration (5 tests)
# =============================================================================
# =============================================================================
# Fallback / Traffic Routing (2 tests)
# =============================================================================
from tests.store.tracking.test_gateway_sql_store import (  # noqa: E402, F401
    test_attach_duplicate_model_raises,
    test_attach_model_to_gateway_endpoint,
    test_create_gateway_endpoint,
    test_create_gateway_endpoint_auto_creates_experiment,
    test_create_gateway_endpoint_binding,
    test_create_gateway_endpoint_empty_models_raises,
    test_create_gateway_endpoint_nonexistent_model_raises,
    test_create_gateway_endpoint_with_fallback_routing,
    test_create_gateway_endpoint_with_traffic_split,
    test_create_gateway_model_definition,
    test_create_gateway_model_definition_duplicate_name_raises,
    test_create_gateway_model_definition_nonexistent_secret_raises,
    test_create_gateway_secret,
    test_create_gateway_secret_duplicate_name_raises,
    test_create_gateway_secret_with_auth_config,
    test_create_gateway_secret_with_dict_value,
    test_delete_gateway_endpoint,
    test_delete_gateway_endpoint_binding,
    test_delete_gateway_endpoint_tag,
    test_delete_gateway_endpoint_tag_nonexistent_endpoint_raises,
    test_delete_gateway_endpoint_tag_nonexistent_key_no_op,
    test_delete_gateway_model_definition,
    test_delete_gateway_model_definition_in_use_raises,
    test_delete_gateway_secret,
    test_detach_model_from_gateway_endpoint,
    test_detach_nonexistent_mapping_raises,
    test_endpoint_tags_deleted_with_endpoint,
    test_get_gateway_endpoint_by_id,
    test_get_gateway_endpoint_by_name,
    test_get_gateway_endpoint_requires_one_of_id_or_name,
    test_get_gateway_model_definition_by_id,
    test_get_gateway_model_definition_by_name,
    test_get_gateway_model_definition_requires_one_of_id_or_name,
    test_get_gateway_secret_info_by_id,
    test_get_gateway_secret_info_by_name,
    test_get_gateway_secret_info_not_found,
    test_get_gateway_secret_info_requires_one_of_id_or_name,
    test_get_scorer_resolves_endpoint_id_to_name,
    test_get_scorer_with_deleted_endpoint_sets_model_to_null,
    test_list_gateway_endpoint_bindings,
    test_list_gateway_endpoints,
    test_list_gateway_model_definitions,
    test_list_gateway_secret_infos,
    test_list_scorers_batch_resolves_endpoint_ids,
    test_register_scorer_resolves_endpoint_name_to_id,
    test_register_scorer_with_nonexistent_endpoint_raises,
    test_set_gateway_endpoint_tag,
    test_set_gateway_endpoint_tag_nonexistent_endpoint_raises,
    test_set_gateway_endpoint_tag_update_existing,
    test_set_multiple_endpoint_tags,
    test_update_gateway_endpoint,
    test_update_gateway_model_definition,
    test_update_gateway_secret,
    test_update_gateway_secret_clear_auth_config,
    test_update_gateway_secret_with_auth_config,
)
