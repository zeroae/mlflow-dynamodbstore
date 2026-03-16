"""Tests for NAME_REV materialized items and FTS writes in DynamoDBTrackingStore."""

from __future__ import annotations

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_NAME_REV,
)


class TestNameRev:
    """Test that create/rename_experiment writes NAME_REV items for suffix ILIKE."""

    def test_create_experiment_writes_name_rev(self, tracking_store):
        """create_experiment should write a NAME_REV item with reversed lowercase name."""
        exp_id = tracking_store.create_experiment("MyExp", artifact_location="s3://b")
        name_rev = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["gsi5sk"].startswith("REV#")
        assert "pxeym" in name_rev["gsi5sk"].lower()  # "MyExp" reversed lowercase

    def test_create_experiment_name_rev_has_correct_gsi5pk(self, tracking_store):
        """NAME_REV item should have gsi5pk set to EXP_NAMES#<workspace>."""
        exp_id = tracking_store.create_experiment("MyExp", artifact_location="s3://b")
        name_rev = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["gsi5pk"] == "EXP_NAMES#default"

    def test_create_experiment_name_rev_has_name_attribute(self, tracking_store):
        """NAME_REV item should store the original name for display purposes."""
        exp_id = tracking_store.create_experiment("MyExp", artifact_location="s3://b")
        name_rev = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_NAME_REV,
        )
        assert name_rev is not None
        assert name_rev["name"] == "MyExp"

    def test_rename_experiment_updates_name_rev(self, tracking_store):
        """rename_experiment should update the NAME_REV item with the new reversed name."""
        exp_id = tracking_store.create_experiment("OldExp", artifact_location="s3://b")
        tracking_store.rename_experiment(exp_id, "NewExp")
        name_rev = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_NAME_REV,
        )
        assert name_rev is not None
        assert "pxewen" in name_rev["gsi5sk"].lower()  # "NewExp" reversed lowercase
        assert "pxedlo" not in name_rev["gsi5sk"].lower()  # "OldExp" reversed NOT present

    def test_rename_experiment_name_rev_contains_exp_id(self, tracking_store):
        """NAME_REV gsi5sk should contain the experiment ID as a suffix."""
        exp_id = tracking_store.create_experiment("MyExp", artifact_location="s3://b")
        name_rev = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=SK_EXPERIMENT_NAME_REV,
        )
        assert name_rev is not None
        assert exp_id in name_rev["gsi5sk"]


class TestFTSWrites:
    """Test that tracking store write operations populate FTS items."""

    def test_create_experiment_writes_fts_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("my pipeline", artifact_location="s3://b")
        # Query for FTS items in the experiment partition
        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        assert len(fts_items) > 0
        # Should have both W# (word) and 3# (trigram) items
        word_items = [i for i in fts_items if i["SK"].startswith("FTS#W#")]
        trigram_items = [i for i in fts_items if i["SK"].startswith("FTS#3#")]
        assert len(word_items) > 0
        assert len(trigram_items) > 0

    def test_create_experiment_fts_has_gsi2(self, tracking_store):
        exp_id = tracking_store.create_experiment("test exp", artifact_location="s3://b")
        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        # Experiment name FTS items should have GSI2 attributes
        for item in fts_items:
            assert "gsi2pk" in item

    def test_create_experiment_fts_has_reverse_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("my pipeline", artifact_location="s3://b")
        rev_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS_REV#")
        assert len(rev_items) > 0
        # All reverse items should reference entity type E and exp_id
        for item in rev_items:
            assert f"#E#{exp_id}" in item["SK"]

    def test_rename_experiment_updates_fts(self, tracking_store):
        exp_id = tracking_store.create_experiment("old name", artifact_location="s3://b")
        tracking_store.rename_experiment(exp_id, "new name")
        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        # Should have tokens for "new" and "name" but not "old"
        sks = [i["SK"] for i in fts_items]
        assert not any("old" in sk.lower() for sk in sks if sk.startswith("FTS#W#"))

    def test_rename_experiment_adds_new_tokens(self, tracking_store):
        exp_id = tracking_store.create_experiment("alpha pipeline", artifact_location="s3://b")
        tracking_store.rename_experiment(exp_id, "beta workflow")
        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#W#")
        sks = [i["SK"] for i in fts_items]
        # Old tokens for "alpha" and "pipeline" (stemmed: "pipelin") should be gone
        assert not any("alpha" in sk.lower() for sk in sks)
        assert not any("pipelin" in sk.lower() for sk in sks)

    def test_create_run_writes_fts_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline"
        )
        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        # Should have run name FTS items (no GSI2 for runs)
        run_fts = [i for i in fts_items if f"#R#{run.info.run_id}" in i["SK"]]
        assert len(run_fts) > 0
        for item in run_fts:
            assert "gsi2pk" not in item

    def test_create_run_fts_has_both_levels(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline"
        )
        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        run_fts = [i for i in fts_items if f"#R#{run.info.run_id}" in i["SK"]]
        word_items = [i for i in run_fts if i["SK"].startswith("FTS#W#")]
        trigram_items = [i for i in run_fts if i["SK"].startswith("FTS#3#")]
        assert len(word_items) > 0
        assert len(trigram_items) > 0

    def test_create_run_fts_has_reverse_items(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp2", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="pipeline"
        )
        rev_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS_REV#")
        run_rev = [i for i in rev_items if f"#R#{run.info.run_id}" in i["SK"]]
        assert len(run_rev) > 0

    def test_update_run_info_updates_fts(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="old-run-name"
        )
        run_id = run.info.run_id
        tracking_store.update_run_info(run_id, "FINISHED", 2000, "new-run-name")

        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#W#")
        run_fts = [i for i in fts_items if f"#R#{run_id}" in i["SK"]]
        sks = [i["SK"] for i in run_fts]
        # The stem of "old" should no longer appear
        assert not any("#old#" in sk.lower() for sk in sks)

    def test_log_batch_params_writes_fts_when_configured(self, tracking_store):
        """When run_param_value is configured for trigrams, log_batch should write FTS."""
        from mlflow.entities import Param

        tracking_store._config.set_fts_trigram_fields(["run_param_value"])

        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="run1"
        )
        run_id = run.info.run_id

        tracking_store.log_batch(
            run_id=run_id,
            metrics=[],
            params=[Param("model_type", "transformer network")],
            tags=[],
        )

        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        param_fts = [i for i in fts_items if f"#R#{run_id}" in i["SK"] and "#model_type" in i["SK"]]
        assert len(param_fts) > 0

    def test_log_batch_params_no_fts_when_not_configured(self, tracking_store):
        """When run_param_value is NOT configured, log_batch should NOT write FTS for params."""
        from mlflow.entities import Param

        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="run2"
        )
        run_id = run.info.run_id

        tracking_store.log_batch(
            run_id=run_id,
            metrics=[],
            params=[Param("model_type", "transformer network")],
            tags=[],
        )

        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        param_fts = [i for i in fts_items if "#model_type" in i["SK"]]
        assert len(param_fts) == 0

    def test_set_tag_writes_fts_when_configured(self, tracking_store):
        """When run_tag_value is configured for trigrams, set_tag should write FTS."""
        from mlflow.entities import RunTag

        tracking_store._config.set_fts_trigram_fields(["run_tag_value"])

        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="run1"
        )
        run_id = run.info.run_id

        tracking_store.set_tag(run_id, RunTag("description", "fraud detection model"))

        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        tag_fts = [i for i in fts_items if f"#R#{run_id}" in i["SK"] and "#description" in i["SK"]]
        assert len(tag_fts) > 0

    def test_set_tag_no_fts_when_not_configured(self, tracking_store):
        """When run_tag_value is NOT configured, set_tag should NOT write FTS."""
        from mlflow.entities import RunTag

        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="run1"
        )
        run_id = run.info.run_id

        tracking_store.set_tag(run_id, RunTag("description", "fraud detection model"))

        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        tag_fts = [i for i in fts_items if "#description" in i["SK"]]
        assert len(tag_fts) == 0

    def test_delete_tag_removes_fts_items(self, tracking_store):
        """delete_tag should remove FTS items for the tag."""
        from mlflow.entities import RunTag

        tracking_store._config.set_fts_trigram_fields(["run_tag_value"])

        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="run1"
        )
        run_id = run.info.run_id

        tracking_store.set_tag(run_id, RunTag("description", "fraud detection model"))
        tracking_store.delete_tag(run_id, "description")

        fts_items = tracking_store._table.query(pk=f"EXP#{exp_id}", sk_prefix="FTS#")
        tag_fts = [i for i in fts_items if "#description" in i["SK"]]
        assert len(tag_fts) == 0
