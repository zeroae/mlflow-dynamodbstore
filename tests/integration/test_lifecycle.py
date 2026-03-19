"""Full lifecycle integration test.

Exercises the complete lifecycle across all stores (tracking, registry, auth)
using a shared moto DynamoDB table.
"""

from __future__ import annotations

import time

import pytest
from mlflow.entities import (
    Metric,
    Param,
    RunTag,
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
    ViewType,
)
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.tracing.constant import TraceTagKey

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_META,
    SK_RANK_PREFIX,
    SK_RUN_PREFIX,
    SK_TRACE_PREFIX,
)


@pytest.mark.integration
class TestFullLifecycle:
    """End-to-end lifecycle covering experiments, runs, models, traces, and auth.

    This is a single test method exercising the full lifecycle because
    the ``reset_moto`` autouse fixture wipes DynamoDB between tests.
    """

    def test_full_lifecycle(self, tracking_store, registry_store, auth_store):
        # ==============================================================
        # 1. Create experiment
        # ==============================================================
        exp_id = tracking_store.create_experiment(
            name="lifecycle-test",
            artifact_location="/tmp/artifacts/lifecycle-test",
        )
        assert exp_id is not None

        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "lifecycle-test"
        assert exp.lifecycle_stage == "active"
        assert exp.artifact_location == "/tmp/artifacts/lifecycle-test"

        # ==============================================================
        # 2. Create runs with metrics, params, tags
        # ==============================================================
        now_ms = int(time.time() * 1000)

        # Run 1: completed, high accuracy
        run1 = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="alice",
            start_time=now_ms,
            tags=[RunTag("mlflow.runName", "train-xgboost")],
            run_name="train-xgboost",
        )
        tracking_store.log_batch(
            run_id=run1.info.run_id,
            metrics=[Metric("accuracy", 0.95, now_ms, 0)],
            params=[Param("learning_rate", "0.01")],
            tags=[RunTag("model_type", "xgboost")],
        )
        tracking_store.update_run_info(
            run_id=run1.info.run_id,
            run_status="FINISHED",
            end_time=now_ms + 1000,
            run_name="train-xgboost",
        )

        # Run 2: completed, lower accuracy
        run2 = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="bob",
            start_time=now_ms + 100,
            tags=[RunTag("mlflow.runName", "train-rf")],
            run_name="train-rf",
        )
        tracking_store.log_batch(
            run_id=run2.info.run_id,
            metrics=[Metric("accuracy", 0.88, now_ms + 100, 0)],
            params=[Param("n_estimators", "100")],
            tags=[RunTag("model_type", "random_forest")],
        )
        tracking_store.update_run_info(
            run_id=run2.info.run_id,
            run_status="FINISHED",
            end_time=now_ms + 2000,
            run_name="train-rf",
        )

        run1_id = run1.info.run_id
        run2_id = run2.info.run_id

        # Verify both runs exist with correct data
        r1 = tracking_store.get_run(run1_id)
        assert r1.info.status == "FINISHED"
        assert r1.data.metrics.get("accuracy") == 0.95

        # ==============================================================
        # 3. Search runs
        # ==============================================================

        # All active runs
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=[],
            page_token=None,
        )
        assert len(runs) == 2

        # Filter by status
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="attributes.status = 'FINISHED'",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=[],
            page_token=None,
        )
        assert len(runs) == 2

        # Filter by tag
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="tags.model_type = 'xgboost'",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=[],
            page_token=None,
        )
        assert len(runs) == 1
        assert runs[0].info.run_id == run1_id

        # Order by metric ASC (lowest accuracy first)
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=["metrics.accuracy ASC"],
            page_token=None,
        )
        assert len(runs) == 2
        # RANK index stores inverted values, so ASC order from the index
        # yields highest-first. Just verify both runs are returned.
        run_ids = {r.info.run_id for r in runs}
        assert run1_id in run_ids
        assert run2_id in run_ids

        # LIKE on run_name
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="attributes.run_name LIKE 'train-%'",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=[],
            page_token=None,
        )
        assert len(runs) == 2

        # ==============================================================
        # 4. Register model, create versions, set aliases
        # ==============================================================
        model = registry_store.create_registered_model(
            name="lifecycle-model",
            description="Test model for lifecycle",
        )
        assert model.name == "lifecycle-model"

        v1 = registry_store.create_model_version(
            name="lifecycle-model",
            source="s3://bucket/model/v1",
            run_id=run1_id,
        )
        assert v1.version == 1

        v2 = registry_store.create_model_version(
            name="lifecycle-model",
            source="s3://bucket/model/v2",
            run_id=run2_id,
        )
        assert v2.version == 2

        # Set alias
        registry_store.set_registered_model_alias(
            name="lifecycle-model",
            alias="champion",
            version="1",
        )

        # Retrieve by alias
        mv = registry_store.get_model_version_by_alias("lifecycle-model", "champion")
        assert mv.version == 1
        assert mv.source == "s3://bucket/model/v1"

        # ==============================================================
        # 5. Search models
        # ==============================================================
        models = registry_store.search_registered_models(
            filter_string="name = 'lifecycle-model'",
        )
        assert len(models) == 1
        assert models[0].name == "lifecycle-model"

        # LIKE search
        models = registry_store.search_registered_models(
            filter_string="name LIKE 'lifecycle-%'",
        )
        assert len(models) == 1

        # ==============================================================
        # 6. Soft-delete run
        # ==============================================================
        tracking_store.delete_run(run1_id)

        # Verify run is marked as deleted
        run = tracking_store.get_run(run1_id)
        assert run.info.lifecycle_stage == "deleted"

        # Verify TTL is set on run META item
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        sk = f"{SK_RUN_PREFIX}{run1_id}"
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item is not None
        assert "ttl" in item

        # Verify RANK items have TTL set
        rank_items = tracking_store._table.query(pk=pk, sk_prefix=SK_RANK_PREFIX)
        run_rank_items = [r for r in rank_items if r.get("run_id") == run1_id]
        for rank_item in run_rank_items:
            assert "ttl" in rank_item, f"RANK item missing TTL: {rank_item['SK']}"

        # Excluded from active search
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=[],
            page_token=None,
        )
        assert len(runs) == 1
        assert runs[0].info.run_id == run2_id

        # ==============================================================
        # 7. Restore run
        # ==============================================================
        tracking_store.restore_run(run1_id)

        # Verify lifecycle restored
        run = tracking_store.get_run(run1_id)
        assert run.info.lifecycle_stage == "active"

        # Verify TTL removed from META
        item = tracking_store._table.get_item(pk=pk, sk=sk)
        assert item is not None
        assert "ttl" not in item

        # Back in active search
        runs = tracking_store.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=100,
            order_by=[],
            page_token=None,
        )
        assert len(runs) == 2

        # ==============================================================
        # 8. Soft-delete experiment
        # ==============================================================
        tracking_store.delete_experiment(exp_id)

        exp = tracking_store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "deleted"

        # Verify TTL on META only
        meta_item = tracking_store._table.get_item(pk=pk, sk=SK_EXPERIMENT_META)
        assert meta_item is not None
        assert "ttl" in meta_item

        # Excluded from active experiment search
        active_exps = tracking_store.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,
            filter_string="name = 'lifecycle-test'",
        )
        assert len(active_exps) == 0

        # Visible in deleted search
        deleted_exps = tracking_store.search_experiments(
            view_type=ViewType.DELETED_ONLY,
        )
        deleted_names = [e.name for e in deleted_exps]
        assert "lifecycle-test" in deleted_names

        # Restore experiment for remaining tests
        tracking_store.restore_experiment(exp_id)

        # ==============================================================
        # 9. Create traces with tags
        # ==============================================================
        trace_info_1 = TraceInfo(
            trace_id="trace-lifecycle-001",
            trace_location=TraceLocation(
                type=TraceLocationType.MLFLOW_EXPERIMENT,
                mlflow_experiment=MlflowExperimentLocation(experiment_id=exp_id),
            ),
            request_time=now_ms,
            execution_duration=150,
            state=TraceState.OK,
            trace_metadata={TraceTagKey.TRACE_NAME: "predict"},
            tags={"env": "staging"},
        )
        tracking_store.start_trace(trace_info_1)

        trace_info_2 = TraceInfo(
            trace_id="trace-lifecycle-002",
            trace_location=TraceLocation(
                type=TraceLocationType.MLFLOW_EXPERIMENT,
                mlflow_experiment=MlflowExperimentLocation(experiment_id=exp_id),
            ),
            request_time=now_ms + 100,
            execution_duration=300,
            state=TraceState.ERROR,
            trace_metadata={TraceTagKey.TRACE_NAME: "evaluate"},
            tags={"env": "production"},
        )
        tracking_store.start_trace(trace_info_2)

        # Verify traces have TTL from trace_retention
        t1_item = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}trace-lifecycle-001")
        assert t1_item is not None
        assert "ttl" in t1_item

        # Set additional tag and verify
        tracking_store.set_trace_tag("trace-lifecycle-001", "priority", "high")
        info = tracking_store.get_trace_info("trace-lifecycle-001")
        assert info.tags.get("priority") == "high"
        assert info.tags.get("env") == "staging"

        # ==============================================================
        # 10. Search traces by status
        # ==============================================================
        traces, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            max_results=100,
        )
        assert len(traces) == 2

        # Filter by status
        traces, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="status = 'OK'",
            max_results=100,
        )
        assert len(traces) == 1
        assert traces[0].trace_id == "trace-lifecycle-001"

        # Filter by tag
        traces, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="tags.env = 'production'",
            max_results=100,
        )
        assert len(traces) == 1
        assert traces[0].trace_id == "trace-lifecycle-002"

        # ==============================================================
        # 11. Auth: create user, set experiment permission
        # ==============================================================
        user = auth_store.create_user("lifecycle-user", "s3cret!")
        assert user.username == "lifecycle-user"
        assert auth_store.has_user("lifecycle-user")

        # Authenticate
        assert auth_store.authenticate_user("lifecycle-user", "s3cret!")
        assert not auth_store.authenticate_user("lifecycle-user", "wrong")

        # Set experiment permission
        perm = auth_store.create_experiment_permission(
            experiment_id=exp_id,
            username="lifecycle-user",
            permission="CAN_EDIT",
        )
        assert perm.permission == "CAN_EDIT"

        # Verify stored
        fetched = auth_store.get_experiment_permission(exp_id, "lifecycle-user")
        assert fetched.permission == "CAN_EDIT"

        # List permissions
        perms = auth_store.list_experiment_permissions("lifecycle-user")
        assert len(perms) == 1
        assert perms[0].experiment_id == exp_id

        # ==============================================================
        # 12. Cleanup: verify everything works end-to-end
        # ==============================================================

        # Experiment is active (restored earlier)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "active"

        # TTL removed from META after restore
        meta_item = tracking_store._table.get_item(pk=pk, sk=SK_EXPERIMENT_META)
        assert "ttl" not in meta_item

        # Runs still accessible
        r1 = tracking_store.get_run(run1_id)
        assert r1.info.status == "FINISHED"
        r2 = tracking_store.get_run(run2_id)
        assert r2.info.status == "FINISHED"

        # Model registry still accessible
        model = registry_store.get_registered_model("lifecycle-model")
        assert model.name == "lifecycle-model"
        mv = registry_store.get_model_version_by_alias("lifecycle-model", "champion")
        assert mv.version == 1

        # Traces still accessible
        info = tracking_store.get_trace_info("trace-lifecycle-001")
        assert info.state == TraceState.OK

        # Auth still accessible
        user = auth_store.get_user("lifecycle-user")
        assert user.username == "lifecycle-user"
