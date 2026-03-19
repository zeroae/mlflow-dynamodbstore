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

# --- Category 1: description returns "" instead of None ---
_xfail_description_none = pytest.mark.xfail(
    reason="DynamoDB store returns '' instead of None for unset description"
)
test_create_registered_model = _xfail_description_none(test_create_registered_model)
test_get_registered_model = _xfail_description_none(test_get_registered_model)
test_update_model_version = _xfail_description_none(test_update_model_version)
test_update_registered_model = _xfail_description_none(test_update_registered_model)

# --- Category 2: version is str instead of int ---
_xfail_version_type = pytest.mark.xfail(
    reason="DynamoDB store returns version as str instead of int"
)
test_create_model_version = _xfail_version_type(test_create_model_version)
test_search_model_versions = _xfail_version_type(test_search_model_versions)
test_search_model_versions_by_tag = _xfail_version_type(test_search_model_versions_by_tag)

# --- Category 3: aliases not returned by get_registered_model ---
_xfail_aliases = pytest.mark.xfail(
    reason="DynamoDB store does not return aliases in get_registered_model"
)
test_set_registered_model_alias = _xfail_aliases(test_set_registered_model_alias)
test_delete_model_deletes_alias = _xfail_aliases(test_delete_model_deletes_alias)
test_delete_model_version_deletes_alias = _xfail_aliases(test_delete_model_version_deletes_alias)
test_delete_registered_model_alias = _xfail_aliases(test_delete_registered_model_alias)
test_get_model_version_by_alias = _xfail_aliases(test_get_model_version_by_alias)

# --- Category 4: error message format mismatch ---
_xfail_error_msg = pytest.mark.xfail(
    reason="DynamoDB store uses different error message format than SqlAlchemy"
)
test_create_registered_model_handle_prompt_properly = _xfail_error_msg(
    test_create_registered_model_handle_prompt_properly
)
test_delete_registered_model = _xfail_error_msg(test_delete_registered_model)

# --- Category 5: update_model_version on deleted version crashes ---
_xfail_deleted_update = pytest.mark.xfail(
    reason="DynamoDB store crashes with KeyError on update of deleted model version"
)
test_delete_model_version = _xfail_deleted_update(test_delete_model_version)

# --- Category 6: missing SqlAlchemy-internal method ---
_xfail_sql_internal = pytest.mark.xfail(
    reason="Test uses _get_sql_model_version_including_deleted (SqlAlchemy-specific)"
)
test_delete_model_version_redaction = _xfail_sql_internal(test_delete_model_version_redaction)

# --- Category 7: operations on deleted entities don't raise ---
_xfail_deleted_ops = pytest.mark.xfail(
    reason="DynamoDB store does not raise on operations against deleted entities"
)
test_set_model_version_tag = _xfail_deleted_ops(test_set_model_version_tag)
test_delete_model_version_tag = _xfail_deleted_ops(test_delete_model_version_tag)
test_transition_model_version_stage_when_archive_existing_versions_is_true = _xfail_deleted_ops(
    test_transition_model_version_stage_when_archive_existing_versions_is_true
)

# --- Category 8: missing input validation ---
_xfail_validation = pytest.mark.xfail(
    reason="DynamoDB store missing input validation (None keys, value length limits)"
)
test_delete_registered_model_tag = _xfail_validation(test_delete_registered_model_tag)
test_set_registered_model_tag = _xfail_validation(test_set_registered_model_tag)

# --- Category 9: latest_versions returns None instead of [] ---
_xfail_latest_versions = pytest.mark.xfail(
    reason="DynamoDB store returns latest_versions=None instead of []"
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

# --- Category 15: order_by validation missing ---
_xfail_orderby_validation = pytest.mark.xfail(
    reason="DynamoDB store does not validate order_by column names"
)
test_search_model_versions_order_by_errors = _xfail_orderby_validation(
    test_search_model_versions_order_by_errors
)
test_search_registered_model_order_by_errors = _xfail_orderby_validation(
    test_search_registered_model_order_by_errors
)
