"""Integration tests for DynamoDBTrackingStore via moto HTTP server."""

from mlflow.entities import Metric, Param, RunTag, ViewType


class TestTrackingIntegration:
    def test_full_experiment_lifecycle(self, tracking_store):
        # Create
        exp_id = tracking_store.create_experiment("int-test", artifact_location="s3://bucket")
        assert exp_id is not None

        # Get
        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "int-test"

        # Rename
        tracking_store.rename_experiment(exp_id, "int-test-renamed")
        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "int-test-renamed"

        # Delete + Restore
        tracking_store.delete_experiment(exp_id)
        assert tracking_store.get_experiment(exp_id).lifecycle_stage == "deleted"
        tracking_store.restore_experiment(exp_id)
        assert tracking_store.get_experiment(exp_id).lifecycle_stage == "active"

    def test_full_run_lifecycle(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")

        # Create run
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="my-run",
        )

        # Log metrics, params, tags
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=[
                Metric("loss", 0.5, 1709251200000, 0),
                Metric("loss", 0.3, 1709251200001, 1),
            ],
            params=[Param("lr", "0.01")],
            tags=[RunTag("mlflow.note", "integration test")],
        )

        # Verify
        fetched = tracking_store.get_run(run.info.run_id)
        assert "loss" in fetched.data.metrics
        assert "lr" in fetched.data.params

        # History
        history = tracking_store.get_metric_history(run.info.run_id, "loss")
        assert len(history) == 2

        # Delete + Restore
        tracking_store.delete_run(run.info.run_id)
        assert tracking_store.get_run(run.info.run_id).info.lifecycle_stage == "deleted"
        tracking_store.restore_run(run.info.run_id)
        assert tracking_store.get_run(run.info.run_id).info.lifecycle_stage == "active"

    def test_search_experiments_gsi2(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)

        active = tracking_store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        deleted = tracking_store.search_experiments(view_type=ViewType.DELETED_ONLY)

        active_names = [e.name for e in active]
        assert "exp-a" in active_names
        assert "exp-b" not in active_names
        assert any(e.name == "exp-b" for e in deleted)

    def test_batch_operations(self, tracking_store):
        """Test batch write with > 25 items (DynamoDB batch limit)."""
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=1709251200000,
            tags=[],
            run_name="batch-test",
        )

        # Log 35 metrics (> 25 DynamoDB batch limit, each metric generates 3 items = 105 items)
        metrics = [Metric(f"metric_{i}", float(i), 1709251200000, i) for i in range(35)]
        tracking_store.log_batch(
            run_id=run.info.run_id,
            metrics=metrics,
            params=[],
            tags=[],
        )

        fetched = tracking_store.get_run(run.info.run_id)
        assert len(fetched.data.metrics) == 35
