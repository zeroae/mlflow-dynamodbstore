"""Tests for tag denormalization into run/experiment META items."""

from __future__ import annotations

from mlflow.entities import ExperimentTag, RunTag

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_META,
    SK_RUN_PREFIX,
)


class TestTagDenormalization:
    def test_mlflow_tag_denormalized_on_run_meta(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.set_tag(run.info.run_id, RunTag("mlflow.user", "alice"))
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_RUN_PREFIX}{run.info.run_id}",
        )
        assert meta is not None
        assert meta.get("tags", {}).get("mlflow.user") == "alice"

    def test_custom_tag_not_denormalized_by_default(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp2", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.set_tag(run.info.run_id, RunTag("custom_tag", "value"))
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_RUN_PREFIX}{run.info.run_id}",
        )
        assert "custom_tag" not in meta.get("tags", {})

    def test_delete_tag_removes_from_denormalized(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp3", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.set_tag(run.info.run_id, RunTag("mlflow.user", "alice"))
        tracking_store.delete_tag(run.info.run_id, "mlflow.user")
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_RUN_PREFIX}{run.info.run_id}",
        )
        assert "mlflow.user" not in meta.get("tags", {})

    def test_log_batch_denormalizes_mlflow_tags(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp4", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.log_batch(
            run.info.run_id,
            metrics=[],
            params=[],
            tags=[RunTag("mlflow.source.name", "train.py"), RunTag("custom", "no")],
        )
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_RUN_PREFIX}{run.info.run_id}",
        )
        assert meta.get("tags", {}).get("mlflow.source.name") == "train.py"
        assert "custom" not in meta.get("tags", {})

    def test_create_run_with_mlflow_tag_denormalizes(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp5", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id,
            user_id="u",
            start_time=1000,
            tags=[RunTag("mlflow.runName", "my-run"), RunTag("user.tag", "skip")],
            run_name="my-run",
        )
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_RUN_PREFIX}{run.info.run_id}",
        )
        assert meta.get("tags", {}).get("mlflow.runName") == "my-run"
        assert "user.tag" not in meta.get("tags", {})

    def test_run_meta_has_run_name_tag_on_create(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp6", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_RUN_PREFIX}{run.info.run_id}",
        )
        assert "tags" in meta
        # mlflow.runName is denormalized as a system tag
        assert meta["tags"].get("mlflow.runName") == "r"


class TestExperimentTagDenormalization:
    def test_mlflow_experiment_tag_denormalized_on_meta(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp-e1", artifact_location="s3://b")
        tracking_store.set_experiment_tag(exp_id, ExperimentTag("mlflow.note.content", "my note"))
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_META,
        )
        assert meta is not None
        assert meta.get("tags", {}).get("mlflow.note.content") == "my note"

    def test_custom_experiment_tag_not_denormalized(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp-e2", artifact_location="s3://b")
        tracking_store.set_experiment_tag(exp_id, ExperimentTag("team", "ml-platform"))
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_META,
        )
        assert "team" not in meta.get("tags", {})

    def test_experiment_meta_has_empty_tags_map_on_create(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp-e3", artifact_location="s3://b")
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_META,
        )
        assert "tags" in meta
        assert meta["tags"] == {}

    def test_create_experiment_with_mlflow_tag_denormalizes(self, tracking_store):
        exp_id = tracking_store.create_experiment(
            "exp-e4",
            artifact_location="s3://b",
            tags=[ExperimentTag("mlflow.note.content", "note"), ExperimentTag("custom", "skip")],
        )
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_META,
        )
        assert meta.get("tags", {}).get("mlflow.note.content") == "note"
        assert "custom" not in meta.get("tags", {})
