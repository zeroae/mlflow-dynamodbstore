"""Phase 2a: MLflow registry store tests run against DynamoDB.

Test functions are imported from the vendored MLflow test suite.
Our conftest provides the `store` fixture backed by DynamoDB (moto).

Excluded tests:
- test_parse_search_registered_models_order_by: pure unit test, no store fixture
"""

import functools
from unittest import mock

import pytest

from tests.store.model_registry.test_sqlalchemy_store import (  # noqa: E402, F401
    test_copy_model_version,
    test_create_model_version,
    test_create_model_version_with_model_id_and_no_run_id,
    test_create_registered_model,
    test_create_registered_model_handle_prompt_properly,
    test_create_webhook,
    test_create_webhook_invalid_events,
    test_create_webhook_invalid_names,
    test_create_webhook_invalid_urls,
    test_create_webhook_valid_names,
    test_delete_model_deletes_alias,
    test_delete_model_version,
    test_delete_model_version_deletes_alias,
    test_delete_model_version_redaction,
    test_delete_model_version_tag,
    test_delete_registered_model,
    test_delete_registered_model_alias,
    test_delete_registered_model_tag,
    test_delete_webhook,
    test_delete_webhook_not_found,
    test_get_latest_versions,
    test_get_model_version_by_alias,
    test_get_model_version_download_uri,
    test_get_registered_model,
    test_get_webhook,
    test_get_webhook_not_found,
    test_list_webhooks,
    test_list_webhooks_invalid_max_results,
    test_list_webhooks_pagination,
    test_rename_registered_model,
    test_search_model_versions,
    test_search_model_versions_by_tag,
    test_search_model_versions_order_by_errors,
    test_search_model_versions_order_by_simple,
    test_search_model_versions_pagination,
    test_search_prompts,
    test_search_prompts_versions,
    test_search_registered_model_order_by,
    test_search_registered_model_order_by_errors,
    test_search_registered_model_pagination,
    test_search_registered_models,
    test_search_registered_models_by_tag,
    test_set_model_version_tag,
    test_set_registered_model_alias,
    test_set_registered_model_tag,
    test_transition_model_version_stage_when_archive_existing_versions_is_false,
    test_transition_model_version_stage_when_archive_existing_versions_is_true,
    test_update_model_version,
    test_update_registered_model,
    test_update_webhook,
    test_update_webhook_invalid_events,
    test_update_webhook_invalid_names,
    test_update_webhook_invalid_urls,
    test_update_webhook_not_found,
    test_update_webhook_partial,
    test_webhook_secret_encryption,
    test_webhook_status_transitions,
)


# --- Category 11: redirect MlflowClient mock to our module path ---
# The vendored test mocks sqlalchemy_store.MlflowClient, but our store imports
# MlflowClient at module level. Delegate our copy to the sa_store's (possibly mocked) copy.
def _sync_mlflow_client_mock(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import mlflow.store.model_registry.sqlalchemy_store as sa_store

        with mock.patch(
            "mlflow_dynamodbstore.registry_store.MlflowClient",
            side_effect=lambda: sa_store.MlflowClient(),
        ):
            return fn(*args, **kwargs)

    return wrapper


test_create_model_version_with_model_id_and_no_run_id = _sync_mlflow_client_mock(
    test_create_model_version_with_model_id_and_no_run_id
)


# --- Sync DynamoDB store timestamps with sqlalchemy_store mock ---
# The vendored tests mock sqlalchemy_store.get_current_time_millis to force identical
# timestamps. Our store imports get_current_time_millis separately, so we delegate to
# the sqlalchemy store's (possibly mocked) copy to stay in sync.
def _sync_time_mock(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import mlflow.store.model_registry.sqlalchemy_store as sa_store

        with mock.patch(
            "mlflow_dynamodbstore.registry_store.get_current_time_millis",
            side_effect=lambda: sa_store.get_current_time_millis(),
        ):
            return fn(*args, **kwargs)

    return wrapper


# --- Category 13: registered model order_by needs time mock sync ---
test_search_registered_model_order_by = _sync_time_mock(test_search_registered_model_order_by)

# --- Category 13: model version cross-partition ordering incomplete ---
_xfail_search_order = pytest.mark.xfail(
    reason="DynamoDB store model version cross-partition ordering incomplete"
)
test_search_model_versions_order_by_simple = _xfail_search_order(
    test_search_model_versions_order_by_simple
)

# --- Webhooks: not implemented ---
_xfail_webhook = pytest.mark.xfail(reason="Webhooks not implemented")
test_delete_webhook = _xfail_webhook(test_delete_webhook)
test_delete_webhook_not_found = _xfail_webhook(test_delete_webhook_not_found)
test_webhook_secret_encryption = _xfail_webhook(test_webhook_secret_encryption)
test_webhook_status_transitions = _xfail_webhook(test_webhook_status_transitions)
