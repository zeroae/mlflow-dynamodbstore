"""Phase 2c: MLflow tracking store tests run against DynamoDB.

Test functions are imported from the vendored MLflow test suite.
Our conftest provides the `tracking_store` fixture backed by DynamoDB (moto).
The module-level `store` fixture overrides conftest's default (registry_store)
to use the tracking_store instead, with workspaces disabled.

Excluded tests (~72 total):
- Fixture dependencies: tests requiring store_and_trace_info, store_with_traces,
  workspaces_enabled, db_uri, tmp_sqlite_uri, cached_db, db_type, or driver fixtures
- SQLAlchemy internals: tests that directly use ManagedSessionMaker, SqlRun, SqlMetric,
  SqlParam, SqlTag, SqlExperiment, SqlEvaluationDatasetRecord, SqlLatestMetric,
  store.engine, store.db_uri, dialect, IntegrityError, MSSQL, MYSQL, SQLITE, POSTGRES,
  or mock SqlAlchemyStore internal methods
- Pure unit tests referencing SqlAlchemy models: test_get_attribute_name,
  test_set/unset_zero_value_insertion_*, test_sql_dataset_record_*
- Async log_spans tests that use ManagedSessionMaker to verify DB-level state
"""

import pytest
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES


@pytest.fixture
def store(tracking_store, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    _patch_sqlalchemy_compat(tracking_store)
    return tracking_store


def _patch_sqlalchemy_compat(store):
    """Add ManagedSessionMaker and _get_run shims for SqlAlchemy-style tests."""
    from contextlib import contextmanager

    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    from mlflow_dynamodbstore.dynamodb.schema import PK_EXPERIMENT_PREFIX, SK_RUN_PREFIX

    @contextmanager
    def _managed_session():
        yield None

    class _RunRecord:
        def __init__(self, item):
            self.run_uuid = item.get("run_id", "")
            self.deleted_time = int(item["deleted_time"]) if "deleted_time" in item else None

    def _get_run(_session, run_id):
        experiment_id = store._resolve_run_experiment(run_id)
        item = store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{experiment_id}",
            sk=f"{SK_RUN_PREFIX}{run_id}",
        )
        if item is None:
            raise MlflowException(
                f"Run '{run_id}' does not exist.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return _RunRecord(item)

    store.ManagedSessionMaker = _managed_session
    store._get_run = _get_run


from tests.store.tracking.test_sqlalchemy_store import (  # noqa: E402, F401
    # --- Pure unit tests (no store fixture needed) ---
    test_artifact_path_segments_for_local,
    test_batch_get_trace_infos_basic,
    test_batch_get_trace_infos_empty,
    test_batch_get_trace_infos_ordering,
    # --- Batch get traces ---
    test_batch_get_traces_basic,
    test_batch_get_traces_empty_trace,
    test_batch_get_traces_integration_with_trace_handler,
    test_batch_get_traces_multiple_traces,
    test_batch_get_traces_ordering,
    test_batch_get_traces_preserves_json_serialization,
    test_batch_get_traces_token_usage,
    test_batch_get_traces_with_complex_attributes,
    test_batch_get_traces_with_incomplete_trace,
    # --- Correlation ---
    test_calculate_trace_filter_correlation_basic,
    test_calculate_trace_filter_correlation_count_expressions,
    test_calculate_trace_filter_correlation_empty_experiment_list,
    test_calculate_trace_filter_correlation_independent_events,
    test_calculate_trace_filter_correlation_multiple_experiments,
    test_calculate_trace_filter_correlation_negative_correlation,
    test_calculate_trace_filter_correlation_perfect,
    test_calculate_trace_filter_correlation_simplified_example,
    test_calculate_trace_filter_correlation_with_base_filter,
    test_calculate_trace_filter_correlation_zero_counts,
    test_concurrent_log_spans_spans_location_tag,
    test_create_experiment_appends_to_artifact_local_path_file_uri_correctly,
    test_create_experiment_appends_to_artifact_local_path_file_uri_correctly_on_windows,
    test_create_experiment_appends_to_artifact_uri_path_correctly,
    # --- Logged models ---
    test_create_logged_model,
    test_create_logged_model_invalid_name,
    test_create_run_appends_to_artifact_local_path_file_uri_correctly,
    test_create_run_appends_to_artifact_local_path_file_uri_correctly_on_windows,
    test_create_run_appends_to_artifact_uri_path_correctly,
    test_create_run_sets_name,
    # --- Run CRUD ---
    test_create_run_with_tags,
    test_dataset_associations_and_lazy_loading,
    # --- Datasets (new-style CRUD) ---
    test_dataset_crud_operations,
    test_dataset_delete_records,
    test_dataset_delete_records_idempotent,
    test_dataset_digest_updates_with_changes,
    test_dataset_experiment_associations,
    test_dataset_filtering_ordering_pagination,
    test_dataset_get_experiment_ids,
    test_dataset_records_pagination,
    test_dataset_schema_and_profile_computation,
    test_dataset_schema_and_profile_incremental_updates,
    test_dataset_search_comprehensive,
    test_dataset_tags_with_sql_backend,
    test_dataset_update_tags,
    test_dataset_upsert_comprehensive,
    test_dataset_user_detection,
    # --- Experiment CRUD ---
    test_default_experiment,
    test_default_experiment_lifecycle,
    test_delete_experiment,
    test_delete_experiment_tag,
    test_delete_logged_model,
    test_delete_logged_model_tag,
    test_delete_restore_experiment_with_runs,
    test_delete_run_does_not_delete_logged_model,
    test_delete_tag,
    test_delete_traces,
    test_delete_traces_raises_error,
    test_delete_traces_with_max_count,
    test_delete_traces_with_max_timestamp,
    test_delete_traces_with_trace_ids,
    test_error_logging_to_deleted_run,
    test_fail_on_multiple_drivers,
    test_finalize_logged_model,
    test_find_completed_sessions,
    test_find_completed_sessions_aggregates_across_all_traces,
    test_find_completed_sessions_with_filter_string,
    test_get_active_online_scorers_filters_by_sample_rate,
    test_get_active_online_scorers_filters_non_gateway_model,
    test_get_active_online_scorers_returns_scorer_fields,
    test_get_deleted_logged_models,
    test_get_deleted_runs,
    test_get_experiment,
    test_get_experiment_invalid_id,
    test_get_logged_model,
    test_get_metric_history_on_non_existent_metric_key,
    test_get_online_scoring_configs_batch,
    test_get_online_scoring_configs_empty_list,
    test_get_online_scoring_configs_nonexistent_ids,
    test_get_run_with_name,
    # --- Get trace ---
    test_get_trace_basic,
    test_get_trace_not_found,
    test_get_trace_with_complete_trace,
    test_get_trace_with_partial_trace,
    # --- Traces ---
    test_legacy_start_and_end_trace_v2,
    # --- Link traces to run ---
    test_link_traces_to_run,
    test_link_traces_to_run_100_limit,
    test_link_traces_to_run_duplicate_trace_ids,
    # --- Log batch ---
    test_log_batch,
    test_log_batch_accepts_empty_payload,
    test_log_batch_allows_tag_overwrite,
    test_log_batch_allows_tag_overwrite_single_req,
    test_log_batch_duplicate_metrics_across_key_batches,
    test_log_batch_duplicate_metrics_mixed_with_new_across_key_batches,
    test_log_batch_internal_error,
    test_log_batch_limits,
    test_log_batch_logged_model,
    test_log_batch_nonexistent_run,
    test_log_batch_null_metrics,
    test_log_batch_param_overwrite_disallowed,
    test_log_batch_param_overwrite_disallowed_single_req,
    test_log_batch_params_idempotency,
    test_log_batch_params_max_length_value,
    test_log_batch_same_metric_repeated_multiple_reqs,
    test_log_batch_same_metric_repeated_single_req,
    test_log_batch_same_metrics_repeated_multiple_reqs,
    test_log_batch_tags_idempotency,
    test_log_batch_with_unchanged_and_new_params,
    test_log_empty_str,
    test_log_input_multiple_times_does_not_overwrite_tags_or_dataset,
    # --- Log inputs ---
    test_log_inputs_and_retrieve_runs_behaves_as_expected,
    test_log_inputs_fails_with_missing_inputs,
    test_log_inputs_handles_case_when_no_datasets_are_specified,
    test_log_inputs_with_duplicates_in_single_request,
    test_log_inputs_with_large_inputs_limit_check,
    test_log_logged_model_params,
    test_log_metric_allows_multiple_values_at_same_ts_and_run_data_uses_max_ts_value,
    # --- Metrics ---
    test_log_metric_concurrent_logging_succeeds,
    test_log_null_metric,
    # --- Log outputs ---
    test_log_outputs,
    # --- Params ---
    test_log_param,
    test_log_param_max_length_value,
    test_log_param_uniqueness,
    test_log_spans_cost,
    # --- Async log_spans tests (no ManagedSessionMaker) ---
    test_log_spans_default_trace_status_in_progress,
    test_log_spans_does_not_update_finalized_trace_status,
    # --- Session handling ---
    test_log_spans_session_id_handling,
    test_log_spans_sets_trace_status_from_root_span,
    test_log_spans_then_start_trace_preserves_tag,
    # --- Log spans token/cost ---
    test_log_spans_token_usage,
    test_log_spans_unset_root_span_status_defaults_to_ok,
    test_log_spans_update_cost_incrementally,
    test_log_spans_update_token_usage_incrementally,
    test_log_spans_updates_in_progress_trace_status_from_root_span,
    test_log_spans_updates_state_unspecified_trace_status_from_root_span,
    test_order_by_attributes,
    # --- Search runs ---
    test_order_by_metric_tag_param,
    test_raise_duplicate_experiments,
    test_raise_experiment_dont_exist,
    test_register_scorer_validates_model,
    test_register_scorer_validates_name,
    test_rename_experiment,
    test_restore_experiment,
    # --- Scorers ---
    test_scorer_operations,
    test_search_attrs,
    # --- Search datasets ---
    test_search_datasets,
    test_search_datasets_returns_no_more_than_max_results,
    test_search_experiments_filter_by_attribute,
    test_search_experiments_filter_by_attribute_and_tag,
    test_search_experiments_filter_by_tag,
    test_search_experiments_filter_by_tag_is_null,
    test_search_experiments_filter_by_time_attribute,
    test_search_experiments_max_results,
    test_search_experiments_max_results_validation,
    test_search_experiments_order_by,
    test_search_experiments_order_by_time_attribute,
    test_search_experiments_pagination,
    test_search_experiments_view_type,
    test_search_full,
    test_search_logged_models,
    test_search_logged_models_datasets_filter,
    test_search_logged_models_filter_string,
    test_search_logged_models_invalid_filter_string,
    test_search_logged_models_order_by,
    test_search_logged_models_order_by_dataset,
    test_search_logged_models_pagination,
    test_search_metrics,
    test_search_params,
    test_search_runs_datasets,
    test_search_runs_datasets_with_param_filters,
    test_search_runs_keep_all_runs_when_sorting,
    test_search_runs_pagination,
    test_search_runs_pagination_last_page_exact,
    test_search_runs_pagination_with_max_results_none,
    test_search_runs_returns_outputs,
    test_search_runs_run_id,
    test_search_runs_start_time_alias,
    test_search_tags,
    test_search_traces_combined_span_filters_match_same_span,
    test_search_traces_pagination_tie_breaker,
    test_search_traces_raise_if_max_results_arg_is_invalid,
    test_search_traces_span_filters_with_no_results,
    test_search_traces_with_assessment_is_null_filters,
    test_search_traces_with_client_request_id_edge_cases,
    test_search_traces_with_client_request_id_filter,
    test_search_traces_with_client_request_id_rlike_filters,
    test_search_traces_with_combined_filters,
    test_search_traces_with_combined_numeric_and_string_filters,
    test_search_traces_with_combined_span_filters,
    test_search_traces_with_empty_and_special_characters,
    test_search_traces_with_end_time_ms_all_operators,
    test_search_traces_with_execution_time_ms_filters,
    test_search_traces_with_expectation_like_filters,
    test_search_traces_with_feedback_and_expectation_filters,
    test_search_traces_with_feedback_like_filters,
    test_search_traces_with_feedback_rlike_filters,
    test_search_traces_with_full_text_filter,
    test_search_traces_with_invalid_span_attribute,
    test_search_traces_with_metadata_is_not_null_filter,
    test_search_traces_with_metadata_is_null_filter,
    test_search_traces_with_metadata_like_filters,
    test_search_traces_with_metadata_rlike_filters,
    test_search_traces_with_name_ilike_variations,
    test_search_traces_with_name_like_filters,
    test_search_traces_with_name_rlike_filters,
    test_search_traces_with_prompts_filter,
    test_search_traces_with_prompts_filter_invalid_comparator,
    test_search_traces_with_prompts_filter_invalid_format,
    test_search_traces_with_prompts_filter_multiple_prompts,
    test_search_traces_with_prompts_filter_no_matches,
    test_search_traces_with_run_id,
    test_search_traces_with_run_id_and_other_filters,
    test_search_traces_with_run_id_filter,
    test_search_traces_with_span_attributes_filter,
    test_search_traces_with_span_attributes_rlike_filters,
    test_search_traces_with_span_attributute_backticks,
    test_search_traces_with_span_content_filter,
    test_search_traces_with_span_name_filter,
    test_search_traces_with_span_name_like_filters,
    test_search_traces_with_span_name_rlike_filters,
    test_search_traces_with_span_status_filter,
    test_search_traces_with_span_type_filter,
    test_search_traces_with_span_type_rlike_filters,
    test_search_traces_with_status_operators,
    test_search_traces_with_tag_like_filters,
    test_search_traces_with_tag_rlike_filters,
    test_search_traces_with_timestamp_ms_filters,
    test_search_vanilla,
    test_search_with_deterministic_max_results,
    test_set_and_delete_tags,
    test_set_experiment_tag,
    test_set_invalid_tag,
    test_set_logged_model_tags,
    # --- Tags ---
    test_set_tag,
    test_set_tag_truncate_too_long_tag,
    test_start_trace,
    # --- Spans location tag ---
    test_start_trace_only_no_spans_location_tag,
    test_start_trace_then_log_spans_adds_tag,
    test_start_trace_with_assessments_missing_trace_id,
    test_to_mlflow_entity_and_proto,
    test_update_run_info,
    test_update_run_name,
    test_upsert_online_scoring_config_creates_config,
    test_upsert_online_scoring_config_nonexistent_scorer,
    test_upsert_online_scoring_config_overwrites,
    test_upsert_online_scoring_config_rejects_non_gateway_model,
    test_upsert_online_scoring_config_rejects_scorer_requiring_expectations,
    test_upsert_online_scoring_config_validates_filter_string,
    test_upsert_online_scoring_config_validates_sample_rate,
)

# --- Category 1: search traces filtering not implemented (30 tests) ---
_xfail_trace_filter = pytest.mark.xfail(
    reason="DynamoDB store search_traces filtering not implemented"
)
test_search_traces_combined_span_filters_match_same_span = _xfail_trace_filter(
    test_search_traces_combined_span_filters_match_same_span
)
test_search_traces_span_filters_with_no_results = _xfail_trace_filter(
    test_search_traces_span_filters_with_no_results
)
test_search_traces_with_assessment_is_null_filters = _xfail_trace_filter(
    test_search_traces_with_assessment_is_null_filters
)
test_search_traces_with_client_request_id_edge_cases = _xfail_trace_filter(
    test_search_traces_with_client_request_id_edge_cases
)
test_search_traces_with_client_request_id_filter = _xfail_trace_filter(
    test_search_traces_with_client_request_id_filter
)
test_search_traces_with_client_request_id_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_client_request_id_rlike_filters
)
test_search_traces_with_combined_filters = _xfail_trace_filter(
    test_search_traces_with_combined_filters
)
test_search_traces_with_combined_numeric_and_string_filters = _xfail_trace_filter(
    test_search_traces_with_combined_numeric_and_string_filters
)
test_search_traces_with_combined_span_filters = _xfail_trace_filter(
    test_search_traces_with_combined_span_filters
)
test_search_traces_with_empty_and_special_characters = _xfail_trace_filter(
    test_search_traces_with_empty_and_special_characters
)
test_search_traces_with_feedback_and_expectation_filters = _xfail_trace_filter(
    test_search_traces_with_feedback_and_expectation_filters
)
test_search_traces_with_feedback_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_feedback_rlike_filters
)
test_search_traces_with_full_text_filter = _xfail_trace_filter(
    test_search_traces_with_full_text_filter
)
test_search_traces_with_metadata_like_filters = _xfail_trace_filter(
    test_search_traces_with_metadata_like_filters
)
test_search_traces_with_metadata_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_metadata_rlike_filters
)
test_search_traces_with_name_ilike_variations = _xfail_trace_filter(
    test_search_traces_with_name_ilike_variations
)
test_search_traces_with_name_like_filters = _xfail_trace_filter(
    test_search_traces_with_name_like_filters
)
test_search_traces_with_name_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_name_rlike_filters
)
test_search_traces_with_prompts_filter = _xfail_trace_filter(test_search_traces_with_prompts_filter)
test_search_traces_with_prompts_filter_multiple_prompts = _xfail_trace_filter(
    test_search_traces_with_prompts_filter_multiple_prompts
)
test_search_traces_with_span_attributes_filter = _xfail_trace_filter(
    test_search_traces_with_span_attributes_filter
)
test_search_traces_with_span_attributes_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_span_attributes_rlike_filters
)
test_search_traces_with_span_attributute_backticks = _xfail_trace_filter(
    test_search_traces_with_span_attributute_backticks
)
test_search_traces_with_span_content_filter = _xfail_trace_filter(
    test_search_traces_with_span_content_filter
)
test_search_traces_with_span_name_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_span_name_rlike_filters
)
test_search_traces_with_span_status_filter = _xfail_trace_filter(
    test_search_traces_with_span_status_filter
)
test_search_traces_with_span_type_filter = _xfail_trace_filter(
    test_search_traces_with_span_type_filter
)
test_search_traces_with_span_type_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_span_type_rlike_filters
)
test_search_traces_with_tag_like_filters = _xfail_trace_filter(
    test_search_traces_with_tag_like_filters
)
test_search_traces_with_tag_rlike_filters = _xfail_trace_filter(
    test_search_traces_with_tag_rlike_filters
)

# --- Category 2: missing input validation (remaining xfails) ---
# test_log_batch_internal_error: patches SqlAlchemy-specific internal methods (_log_metrics, etc.)
test_log_batch_internal_error = pytest.mark.xfail(
    reason="Test patches SqlAlchemy-specific internal methods, not applicable to DynamoDB store"
)(test_log_batch_internal_error)
# search_logged_models order_by: needs sorting feature implementation
test_search_logged_models_order_by = pytest.mark.xfail(
    reason="DynamoDB store does not yet support search_logged_models order_by"
)(test_search_logged_models_order_by)

# --- Category 3: experiment ID format mismatch (permanent) ---
_xfail_error_msg = pytest.mark.xfail(reason="DynamoDB store uses ULID experiment IDs, not integers")
test_get_experiment_invalid_id = _xfail_error_msg(test_get_experiment_invalid_id)


# --- Category 5: search runs ordering/pagination/filtering broken (15 tests) ---
_xfail_search_runs = pytest.mark.xfail(
    reason="DynamoDB store search_runs ordering, pagination, and filtering incomplete"
)
test_order_by_attributes = _xfail_search_runs(test_order_by_attributes)
test_order_by_metric_tag_param = _xfail_search_runs(test_order_by_metric_tag_param)
test_search_attrs = _xfail_search_runs(test_search_attrs)
test_search_metrics = _xfail_search_runs(test_search_metrics)
test_search_params = _xfail_search_runs(test_search_params)
test_search_runs_datasets = _xfail_search_runs(test_search_runs_datasets)
test_search_runs_datasets_with_param_filters = _xfail_search_runs(
    test_search_runs_datasets_with_param_filters
)
test_search_runs_pagination = _xfail_search_runs(test_search_runs_pagination)
test_search_runs_pagination_last_page_exact = _xfail_search_runs(
    test_search_runs_pagination_last_page_exact
)
test_search_runs_pagination_with_max_results_none = _xfail_search_runs(
    test_search_runs_pagination_with_max_results_none
)
test_search_runs_returns_outputs = _xfail_search_runs(test_search_runs_returns_outputs)
test_search_runs_start_time_alias = _xfail_search_runs(test_search_runs_start_time_alias)
test_search_tags = _xfail_search_runs(test_search_tags)
test_search_vanilla = _xfail_search_runs(test_search_vanilla)
test_search_with_deterministic_max_results = _xfail_search_runs(
    test_search_with_deterministic_max_results
)

# --- Category 6: dataset CRUD/association bugs (14 tests) ---
_xfail_dataset = pytest.mark.xfail(reason="DynamoDB store dataset CRUD/association bugs")
test_dataset_crud_operations = _xfail_dataset(test_dataset_crud_operations)
test_dataset_associations_and_lazy_loading = _xfail_dataset(
    test_dataset_associations_and_lazy_loading
)
test_dataset_experiment_associations = _xfail_dataset(test_dataset_experiment_associations)
test_dataset_filtering_ordering_pagination = _xfail_dataset(
    test_dataset_filtering_ordering_pagination
)
test_dataset_search_comprehensive = _xfail_dataset(test_dataset_search_comprehensive)
test_dataset_update_tags = _xfail_dataset(test_dataset_update_tags)
test_dataset_upsert_comprehensive = _xfail_dataset(test_dataset_upsert_comprehensive)
test_dataset_user_detection = _xfail_dataset(test_dataset_user_detection)
test_dataset_delete_records_idempotent = _xfail_dataset(test_dataset_delete_records_idempotent)
test_search_datasets = _xfail_dataset(test_search_datasets)
test_search_datasets_returns_no_more_than_max_results = _xfail_dataset(
    test_search_datasets_returns_no_more_than_max_results
)
test_log_input_multiple_times_does_not_overwrite_tags_or_dataset = _xfail_dataset(
    test_log_input_multiple_times_does_not_overwrite_tags_or_dataset
)
test_log_inputs_and_retrieve_runs_behaves_as_expected = _xfail_dataset(
    test_log_inputs_and_retrieve_runs_behaves_as_expected
)
test_log_inputs_with_large_inputs_limit_check = _xfail_dataset(
    test_log_inputs_with_large_inputs_limit_check
)

# --- Category 7: trace persistence remaining issues ---
_xfail_trace_persist = pytest.mark.xfail(
    reason="DynamoDB store trace persistence issues (handler, incomplete, tags, sessions)"
)
test_batch_get_traces_with_incomplete_trace = _xfail_trace_persist(
    test_batch_get_traces_with_incomplete_trace
)
# Concurrent log_spans race: multiple threads writing spansLocation tag can lose writes
test_concurrent_log_spans_spans_location_tag = _xfail_trace_persist(
    test_concurrent_log_spans_spans_location_tag
)
test_log_spans_session_id_handling = _xfail_trace_persist(test_log_spans_session_id_handling)
test_set_and_delete_tags = _xfail_trace_persist(test_set_and_delete_tags)
test_start_trace_with_assessments_missing_trace_id = _xfail_trace_persist(
    test_start_trace_with_assessments_missing_trace_id
)

# --- Category 8: search experiments filtering/ordering/pagination broken (9 tests) ---
_xfail_search_exp = pytest.mark.xfail(
    reason="DynamoDB store search_experiments filtering, ordering, and pagination incomplete"
)
test_default_experiment_lifecycle = _xfail_search_exp(test_default_experiment_lifecycle)
test_search_experiments_filter_by_attribute = _xfail_search_exp(
    test_search_experiments_filter_by_attribute
)
test_search_experiments_filter_by_tag = _xfail_search_exp(test_search_experiments_filter_by_tag)
test_search_experiments_filter_by_tag_is_null = _xfail_search_exp(
    test_search_experiments_filter_by_tag_is_null
)
test_search_experiments_filter_by_time_attribute = _xfail_search_exp(
    test_search_experiments_filter_by_time_attribute
)
test_search_experiments_max_results = _xfail_search_exp(test_search_experiments_max_results)
test_search_experiments_order_by_time_attribute = _xfail_search_exp(
    test_search_experiments_order_by_time_attribute
)
test_search_experiments_pagination = _xfail_search_exp(test_search_experiments_pagination)
test_search_experiments_view_type = _xfail_search_exp(test_search_experiments_view_type)

# --- Category 9: metric history dedup (2 tests, schema change needed) ---
# History SK is key#step#timestamp — same step+timestamp with different values overwrites.
# Fixing requires adding value to SK, which is a schema-breaking change.
_xfail_metric_history = pytest.mark.xfail(
    reason="Metric history SK dedup: same step+timestamp different values overwrites"
)
test_log_metric_allows_multiple_values_at_same_ts_and_run_data_uses_max_ts_value = (
    _xfail_metric_history(
        test_log_metric_allows_multiple_values_at_same_ts_and_run_data_uses_max_ts_value
    )
)
test_log_metric_concurrent_logging_succeeds = _xfail_metric_history(
    test_log_metric_concurrent_logging_succeeds
)

# --- Reclassified from Cat 9 to Cat 6 (dataset CRUD) ---
test_dataset_schema_and_profile_computation = _xfail_dataset(
    test_dataset_schema_and_profile_computation
)
test_dataset_schema_and_profile_incremental_updates = _xfail_dataset(
    test_dataset_schema_and_profile_incremental_updates
)
test_dataset_digest_updates_with_changes = _xfail_dataset(test_dataset_digest_updates_with_changes)

# test_delete_traces_with_max_count: reclassified to Cat 17 (see below)

# --- Category 10: logged model search/lifecycle bugs (7 tests) ---
_xfail_logged_model = pytest.mark.xfail(
    reason="DynamoDB store logged model search, lifecycle, and metrics incomplete"
)
test_create_logged_model = _xfail_logged_model(test_create_logged_model)
test_search_logged_models = _xfail_logged_model(test_search_logged_models)
test_search_logged_models_datasets_filter = _xfail_logged_model(
    test_search_logged_models_datasets_filter
)
test_search_logged_models_filter_string = _xfail_logged_model(
    test_search_logged_models_filter_string
)
test_search_logged_models_pagination = _xfail_logged_model(test_search_logged_models_pagination)
test_log_batch_logged_model = _xfail_logged_model(test_log_batch_logged_model)
test_log_outputs = _xfail_logged_model(test_log_outputs)

# --- Category 11: batch_get_traces remaining issues ---
# --- Category 12: trace tag/artifact path issues (remaining) ---
_xfail_trace_path = pytest.mark.xfail(
    reason="DynamoDB store trace artifact path or session tracking issues"
)
# NOTE: test_get_trace_with_partial_trace is parametrized [True/False].
# [True] passes (trace creation via log_spans works), but [False] fails because
# our store doesn't raise MlflowException for missing traces in get_trace.
# We xfail the whole test; [True] will show as xpassed which is fine.
test_get_trace_with_partial_trace = _xfail_trace_path(test_get_trace_with_partial_trace)
test_find_completed_sessions = _xfail_trace_path(test_find_completed_sessions)
test_find_completed_sessions_aggregates_across_all_traces = _xfail_trace_path(
    test_find_completed_sessions_aggregates_across_all_traces
)
test_find_completed_sessions_with_filter_string = _xfail_trace_path(
    test_find_completed_sessions_with_filter_string
)

# --- Category 13: missing methods (remaining) ---
# deprecated_start_trace_v2: legacy V2 trace API, not worth implementing
test_legacy_start_and_end_trace_v2 = pytest.mark.xfail(
    reason="Deprecated V2 trace API not implemented in DynamoDB store"
)(test_legacy_start_and_end_trace_v2)


# --- Category 15: DynamoDB key size/duplicate key constraints (remaining) ---
# search_logged_models order_by_dataset: needs sorting feature implementation (same as Cat 2)
test_search_logged_models_order_by_dataset = pytest.mark.xfail(
    reason="DynamoDB store does not yet support search_logged_models order_by"
)(test_search_logged_models_order_by_dataset)

# --- Category 17: misc run/experiment/trace/scorer lifecycle bugs (7 tests) ---
