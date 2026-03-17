"""Tests for NAME_REV materialized items and FTS writes in DynamoDBTrackingStore."""

from __future__ import annotations

from mlflow.entities import Metric, RunTag, ViewType

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


class TestSearchRuns:
    """Tests for the full _search_runs implementation."""

    def test_search_by_status(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run1 = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="r1"
        )
        tracking_store.update_run_info(run1.info.run_id, "FINISHED", end_time=2000, run_name="r1")
        tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        runs, _ = tracking_store._search_runs(
            [exp_id], "status = 'FINISHED'", ViewType.ACTIVE_ONLY, 100, None, None
        )
        assert len(runs) == 1
        assert runs[0].info.run_id == run1.info.run_id

    def test_search_order_by_metric(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp2", artifact_location="s3://b")
        run1 = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="r1"
        )
        run2 = tracking_store.create_run(
            exp_id, user_id="u", start_time=1001, tags=[], run_name="r2"
        )
        tracking_store.log_batch(
            run1.info.run_id, metrics=[Metric("acc", 0.8, 0, 0)], params=[], tags=[]
        )
        tracking_store.log_batch(
            run2.info.run_id, metrics=[Metric("acc", 0.95, 0, 0)], params=[], tags=[]
        )
        # ASC = ascending original values (lowest first)
        runs, _ = tracking_store._search_runs(
            [exp_id], "", ViewType.ACTIVE_ONLY, 100, ["metric.acc ASC"], None
        )
        assert len(runs) == 2
        assert runs[0].info.run_id == run1.info.run_id  # 0.8 < 0.95

        # DESC = descending original values (highest first)
        runs, _ = tracking_store._search_runs(
            [exp_id], "", ViewType.ACTIVE_ONLY, 100, ["metric.acc DESC"], None
        )
        assert len(runs) == 2
        assert runs[0].info.run_id == run2.info.run_id  # 0.95 > 0.8

    def test_search_by_denormalized_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp3", artifact_location="s3://b")
        run1 = tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="r1"
        )
        run2 = tracking_store.create_run(
            exp_id, user_id="u", start_time=1001, tags=[], run_name="r2"
        )
        tracking_store.set_tag(run1.info.run_id, RunTag("mlflow.user", "alice"))
        tracking_store.set_tag(run2.info.run_id, RunTag("mlflow.user", "bob"))
        runs, _ = tracking_store._search_runs(
            [exp_id],
            "tag.mlflow.user = 'alice'",
            ViewType.ACTIVE_ONLY,
            100,
            None,
            None,
        )
        assert len(runs) == 1
        assert runs[0].info.run_id == run1.info.run_id

    def test_search_run_name_like_word(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp4", artifact_location="s3://b")
        tracking_store.create_run(
            exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline-v1"
        )
        tracking_store.create_run(
            exp_id, user_id="u", start_time=1001, tags=[], run_name="other-run"
        )
        runs, _ = tracking_store._search_runs(
            [exp_id],
            "run_name LIKE '%pipeline%'",
            ViewType.ACTIVE_ONLY,
            100,
            None,
            None,
        )
        assert len(runs) == 1

    def test_search_no_filter(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp5", artifact_location="s3://b")
        tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        runs, _ = tracking_store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 2

    def test_search_pagination(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp6", artifact_location="s3://b")
        for i in range(15):
            tracking_store.create_run(
                exp_id, user_id="u", start_time=1000 + i, tags=[], run_name=f"r{i}"
            )
        runs, token = tracking_store._search_runs(
            [exp_id], "", ViewType.ACTIVE_ONLY, 10, None, None
        )
        assert len(runs) == 10
        assert token is not None
        runs2, token2 = tracking_store._search_runs(
            [exp_id], "", ViewType.ACTIVE_ONLY, 10, None, token
        )
        assert len(runs2) == 5

    def test_multi_experiment_search(self, tracking_store):
        exp1 = tracking_store.create_experiment("exp7a", artifact_location="s3://b")
        exp2 = tracking_store.create_experiment("exp7b", artifact_location="s3://b")
        tracking_store.create_run(exp1, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.create_run(exp2, user_id="u", start_time=1001, tags=[], run_name="r2")
        runs, _ = tracking_store._search_runs(
            [exp1, exp2], "", ViewType.ACTIVE_ONLY, 100, None, None
        )
        assert len(runs) == 2


class TestSearchExperiments:
    """Tests for search_experiments with filter_string and order_by support."""

    def test_search_no_filter(self, tracking_store):
        """search_experiments with no filter returns all active experiments."""
        tracking_store.create_experiment("exp1", artifact_location="s3://b")
        tracking_store.create_experiment("exp2", artifact_location="s3://b")
        exps = tracking_store.search_experiments(ViewType.ACTIVE_ONLY)
        assert len(exps) >= 2

    def test_search_by_name_equals(self, tracking_store):
        """search_experiments with name = 'target-exp' returns only that experiment."""
        tracking_store.create_experiment("target-exp", artifact_location="s3://b")
        tracking_store.create_experiment("other-exp", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY, filter_string="name = 'target-exp'"
        )
        assert len(exps) == 1
        assert exps[0].name == "target-exp"

    def test_search_by_name_like_prefix(self, tracking_store):
        """search_experiments with name LIKE 'prod%' returns prefix matches."""
        tracking_store.create_experiment("prod-pipeline", artifact_location="s3://b")
        tracking_store.create_experiment("dev-pipeline", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY, filter_string="name LIKE 'prod%'"
        )
        assert len(exps) == 1
        assert exps[0].name == "prod-pipeline"

    def test_search_by_name_like_suffix(self, tracking_store):
        """search_experiments with name ILIKE '%pipeline' returns suffix matches."""
        tracking_store.create_experiment("prod-pipeline", artifact_location="s3://b")
        tracking_store.create_experiment("dev-workflow", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY, filter_string="name ILIKE '%pipeline'"
        )
        assert len(exps) == 1
        assert exps[0].name == "prod-pipeline"

    def test_search_by_name_like_contains(self, tracking_store):
        """search_experiments with name LIKE '%pipeline%' returns substring matches."""
        tracking_store.create_experiment("my-pipeline-v1", artifact_location="s3://b")
        tracking_store.create_experiment("my-other-job", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY, filter_string="name LIKE '%pipeline%'"
        )
        assert len(exps) == 1
        assert exps[0].name == "my-pipeline-v1"

    def test_search_by_tag(self, tracking_store):
        """search_experiments with tag filter returns only tagged experiments."""
        from mlflow.entities import ExperimentTag

        exp_id = tracking_store.create_experiment("tagged-exp", artifact_location="s3://b")
        tracking_store.set_experiment_tag(exp_id, ExperimentTag("team", "ml"))
        tracking_store.create_experiment("untagged-exp", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY, filter_string="tag.team = 'ml'"
        )
        assert len(exps) == 1
        assert exps[0].name == "tagged-exp"

    def test_search_order_by_name_asc(self, tracking_store):
        """search_experiments with order_by=['name ASC'] returns sorted by name."""
        tracking_store.create_experiment("zebra", artifact_location="s3://b")
        tracking_store.create_experiment("alpha", artifact_location="s3://b")
        tracking_store.create_experiment("middle", artifact_location="s3://b")
        exps = tracking_store.search_experiments(ViewType.ACTIVE_ONLY, order_by=["name ASC"])
        names = [e.name for e in exps]
        assert names == sorted(names)

    def test_search_order_by_name_desc(self, tracking_store):
        """search_experiments with order_by=['name DESC'] returns reverse sorted."""
        tracking_store.create_experiment("zebra", artifact_location="s3://b")
        tracking_store.create_experiment("alpha", artifact_location="s3://b")
        exps = tracking_store.search_experiments(ViewType.ACTIVE_ONLY, order_by=["name DESC"])
        names = [e.name for e in exps]
        assert names == sorted(names, reverse=True)

    def test_search_deleted_experiments(self, tracking_store):
        """search_experiments with DELETED_ONLY returns only deleted experiments."""
        exp_id = tracking_store.create_experiment("to-delete", artifact_location="s3://b")
        tracking_store.create_experiment("keep-me", artifact_location="s3://b")
        tracking_store.delete_experiment(exp_id)
        exps = tracking_store.search_experiments(ViewType.DELETED_ONLY)
        assert len(exps) == 1
        assert exps[0].name == "to-delete"

    def test_search_max_results(self, tracking_store):
        """search_experiments respects max_results."""
        for i in range(5):
            tracking_store.create_experiment(f"exp-{i}", artifact_location="s3://b")
        exps = tracking_store.search_experiments(ViewType.ACTIVE_ONLY, max_results=2)
        assert len(exps) == 2

    def test_search_by_name_equals_no_match(self, tracking_store):
        """search_experiments with name = 'nonexistent' returns empty."""
        tracking_store.create_experiment("some-exp", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY, filter_string="name = 'nonexistent'"
        )
        assert len(exps) == 0

    def test_search_filter_with_order_by(self, tracking_store):
        """search_experiments with both filter and order_by works correctly."""
        tracking_store.create_experiment("prod-zebra", artifact_location="s3://b")
        tracking_store.create_experiment("prod-alpha", artifact_location="s3://b")
        tracking_store.create_experiment("dev-pipeline", artifact_location="s3://b")
        exps = tracking_store.search_experiments(
            ViewType.ACTIVE_ONLY,
            filter_string="name LIKE 'prod%'",
            order_by=["name ASC"],
        )
        assert len(exps) == 2
        names = [e.name for e in exps]
        assert names == sorted(names)

    def test_search_view_type_all(self, tracking_store):
        """search_experiments with ALL returns both active and deleted."""
        exp_id = tracking_store.create_experiment("deleted-one", artifact_location="s3://b")
        tracking_store.create_experiment("active-one", artifact_location="s3://b")
        tracking_store.delete_experiment(exp_id)
        exps = tracking_store.search_experiments(ViewType.ALL)
        assert len(exps) >= 2
