"""Tests for DynamoDBTrackingStore experiment CRUD operations."""

import pytest
from mlflow.entities import ExperimentTag, ViewType


class TestExperimentCRUD:
    def test_create_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        assert exp_id is not None
        assert len(exp_id) == 26  # ULID

    def test_get_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "test-exp"
        assert exp.experiment_id == exp_id
        assert exp.lifecycle_stage == "active"
        assert exp.artifact_location == "s3://bucket"
        assert exp.creation_time is not None
        assert exp.last_update_time is not None

    def test_get_experiment_by_name(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        exp = tracking_store.get_experiment_by_name("test-exp")
        assert exp is not None
        assert exp.experiment_id == exp_id

    def test_create_duplicate_name_raises(self, tracking_store):
        tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        with pytest.raises(Exception):
            tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")

    def test_get_default_experiment(self, tracking_store):
        exp = tracking_store.get_experiment("0")
        assert exp.name == "Default"
        assert exp.experiment_id == "0"

    def test_rename_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("old-name", artifact_location="s3://bucket")
        tracking_store.rename_experiment(exp_id, "new-name")
        exp = tracking_store.get_experiment(exp_id)
        assert exp.name == "new-name"
        assert tracking_store.get_experiment_by_name("old-name") is None
        assert tracking_store.get_experiment_by_name("new-name").experiment_id == exp_id

    def test_delete_and_restore_experiment(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_id)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "deleted"
        tracking_store.restore_experiment(exp_id)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "active"

    def test_search_experiments_active_only(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)
        results = tracking_store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        names = [e.name for e in results]
        assert "exp-a" in names
        assert "exp-b" not in names

    def test_search_experiments_deleted_only(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)
        results = tracking_store.search_experiments(view_type=ViewType.DELETED_ONLY)
        names = [e.name for e in results]
        assert "exp-a" not in names
        assert "exp-b" in names

    def test_search_experiments_all(self, tracking_store):
        tracking_store.create_experiment("exp-a", artifact_location="s3://bucket")
        exp_b = tracking_store.create_experiment("exp-b", artifact_location="s3://bucket")
        tracking_store.delete_experiment(exp_b)
        results = tracking_store.search_experiments(view_type=ViewType.ALL)
        names = [e.name for e in results]
        assert "exp-a" in names
        assert "exp-b" in names
        assert "Default" in names

    def test_set_experiment_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-exp", artifact_location="s3://bucket")
        tag = ExperimentTag("my-key", "my-value")
        tracking_store.set_experiment_tag(exp_id, tag)
        exp = tracking_store.get_experiment(exp_id)
        assert exp.tags["my-key"] == "my-value"

    def test_create_experiment_with_tags(self, tracking_store):
        tags = [ExperimentTag("k1", "v1"), ExperimentTag("k2", "v2")]
        exp_id = tracking_store.create_experiment(
            "test-exp", artifact_location="s3://bucket", tags=tags
        )
        exp = tracking_store.get_experiment(exp_id)
        assert exp.tags["k1"] == "v1"
        assert exp.tags["k2"] == "v2"

    def test_get_nonexistent_experiment_raises(self, tracking_store):
        with pytest.raises(Exception):
            tracking_store.get_experiment("nonexistent")

    def test_get_experiment_by_name_nonexistent(self, tracking_store):
        result = tracking_store.get_experiment_by_name("nonexistent")
        assert result is None

    def test_search_experiments_max_results(self, tracking_store):
        for i in range(5):
            tracking_store.create_experiment(f"exp-{i}", artifact_location="s3://bucket")
        results = tracking_store.search_experiments(view_type=ViewType.ACTIVE_ONLY, max_results=3)
        # Should return at most 3 (could include Default)
        assert len(results) <= 3
