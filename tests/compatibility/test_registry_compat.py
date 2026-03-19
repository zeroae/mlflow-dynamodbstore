"""Phase 2a: MLflow registry store tests run against DynamoDB.

Test functions are imported from the vendored MLflow test suite.
Our conftest provides the `store` fixture backed by DynamoDB (moto).

Excluded tests:
- test_parse_search_registered_models_order_by: pure unit test, no store fixture
- test_webhook_secret_encryption: accesses store.engine (SqlAlchemy-specific)
- test_copy_model_version: uses copy_to_same_model parametrize fixture
- Tests with cached_db/db_uri/workspaces_enabled fixture dependencies
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.model_registry.test_sqlalchemy_store import (  # noqa: E402, F401
    test_create_model_version,
    test_create_model_version_with_model_id_and_no_run_id,
    test_create_registered_model,
    test_create_registered_model_handle_prompt_properly,
    test_delete_model_deletes_alias,
    test_delete_model_version,
    test_delete_model_version_deletes_alias,
    test_delete_model_version_redaction,
    test_delete_model_version_tag,
    test_delete_registered_model,
    test_delete_registered_model_alias,
    test_delete_registered_model_tag,
    test_get_latest_versions,
    test_get_model_version_by_alias,
    test_get_model_version_download_uri,
    test_get_registered_model,
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
)

# --- Category 6: missing SqlAlchemy-internal method ---
_xfail_sql_internal = pytest.mark.xfail(
    reason="Test uses _get_sql_model_version_including_deleted (SqlAlchemy-specific)"
)
test_delete_model_version_redaction = _xfail_sql_internal(test_delete_model_version_redaction)

# --- Category 9: get_latest_versions not implemented ---
_xfail_latest_versions = pytest.mark.xfail(
    reason="DynamoDB store get_latest_versions does not filter by stage"
)
test_get_latest_versions = _xfail_latest_versions(test_get_latest_versions)

# --- Category 10: rename doesn't propagate to model versions ---
_xfail_rename = pytest.mark.xfail(
    reason="DynamoDB store rename_registered_model does not update model version names"
)
test_rename_registered_model = _xfail_rename(test_rename_registered_model)

# --- Category 11: model_id lookup not implemented ---
_xfail_model_id = pytest.mark.xfail(
    reason="DynamoDB store does not implement model_id lookup in create_model_version"
)
test_create_model_version_with_model_id_and_no_run_id = _xfail_model_id(
    test_create_model_version_with_model_id_and_no_run_id
)

# --- Category 12: prompt filtering not implemented ---
_xfail_prompts = pytest.mark.xfail(
    reason="DynamoDB store does not filter prompts from search results"
)
test_search_prompts = _xfail_prompts(test_search_prompts)
test_search_prompts_versions = _xfail_prompts(test_search_prompts_versions)

# --- Category 13: search ordering and pagination broken ---
_xfail_search_order = pytest.mark.xfail(
    reason="DynamoDB store search ordering and pagination incomplete"
)
test_search_model_versions = _xfail_search_order(test_search_model_versions)
test_search_model_versions_by_tag = _xfail_search_order(test_search_model_versions_by_tag)
test_search_model_versions_order_by_simple = _xfail_search_order(
    test_search_model_versions_order_by_simple
)
test_search_model_versions_pagination = _xfail_search_order(test_search_model_versions_pagination)
test_search_registered_model_pagination = _xfail_search_order(
    test_search_registered_model_pagination
)
test_search_registered_model_order_by = _xfail_search_order(test_search_registered_model_order_by)

# --- Category 14: infix LIKE patterns not supported ---
_xfail_like = pytest.mark.xfail(
    reason="DynamoDB store only supports prefix LIKE patterns, not infix '%X%'"
)
test_search_registered_models = _xfail_like(test_search_registered_models)
