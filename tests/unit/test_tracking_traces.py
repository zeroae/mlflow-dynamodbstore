"""Tests for trace metadata CRUD and assessment CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import time

import pytest
from mlflow.entities import (
    Assessment,
    AssessmentSource,
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
)
from mlflow.entities.assessment import ExpectationValue, FeedbackValue
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey

from mlflow_dynamodbstore.dynamodb.schema import (
    GSI1_CLIENT_PREFIX,
    GSI1_PK,
    GSI1_SK,
    GSI1_TRACE_PREFIX,
    LSI1_SK,
    LSI2_SK,
    LSI3_SK,
    LSI4_SK,
    LSI5_SK,
    PK_CONFIG,
    PK_EXPERIMENT_PREFIX,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
    SK_TRACE_PREFIX,
)


def _make_trace_info(experiment_id: str, **overrides) -> TraceInfo:
    """Helper to build a TraceInfo with sensible defaults."""
    defaults = dict(
        trace_id="tr-abc123",
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=1709251200000,
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={TraceTagKey.TRACE_NAME: "my-trace"},
        tags={"user_tag": "hello"},
    )
    defaults.update(overrides)
    return TraceInfo(**defaults)


def _create_experiment(tracking_store) -> str:
    """Create an experiment and return its id."""
    return tracking_store.create_experiment("test-exp")


class TestStartTrace:
    def test_start_trace(self, tracking_store):
        """Creates experiment, starts trace, verifies get_trace_info returns it."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)

        result = tracking_store.start_trace(trace_info)

        assert result.trace_id == "tr-abc123"
        assert result.state == TraceState.OK

        fetched = tracking_store.get_trace_info("tr-abc123")
        assert fetched.trace_id == "tr-abc123"
        assert fetched.request_time == 1709251200000
        assert fetched.execution_duration == 500
        assert fetched.state == TraceState.OK

    def test_trace_has_ttl(self, tracking_store):
        """Verifies trace META item has TTL from CONFIG#TTL_POLICY.trace_retention_days."""
        exp_id = _create_experiment(tracking_store)

        # Set TTL policy: 7 days
        tracking_store._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": "TTL_POLICY",
                "trace_retention_days": 7,
            }
        )

        trace_info = _make_trace_info(exp_id)
        before = int(time.time())
        tracking_store.start_trace(trace_info)
        after = int(time.time())

        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        assert "ttl" in meta
        expected_min = before + 7 * 86400
        expected_max = after + 7 * 86400
        assert expected_min <= int(meta["ttl"]) <= expected_max

    def test_ttl_policy_cached_in_config_reader(self, tracking_store):
        """ConfigReader.get_ttl_policy() caches and returns correct defaults."""
        policy = tracking_store._config.get_ttl_policy()
        assert policy["trace_retention_days"] == 30
        assert policy["soft_deleted_retention_days"] == 90
        assert policy["metric_history_retention_days"] == 365

        # Write custom policy
        tracking_store._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": "TTL_POLICY",
                "trace_retention_days": 7,
            }
        )
        # Still returns cached value (30)
        assert tracking_store._config.get_ttl_policy()["trace_retention_days"] == 30

    def test_trace_has_default_ttl(self, tracking_store):
        """Without CONFIG#TTL_POLICY, default is 30 days."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        before = int(time.time())
        tracking_store.start_trace(trace_info)
        after = int(time.time())

        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        expected_min = before + 30 * 86400
        expected_max = after + 30 * 86400
        assert expected_min <= int(meta["ttl"]) <= expected_max

    def test_trace_lsi_attributes(self, tracking_store):
        """Verifies LSI attrs on META: lsi1sk=timestamp_ms, lsi3sk=status#timestamp_ms,
        lsi4sk=lower(trace_name), lsi5sk=execution_time_ms."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)

        tracking_store.start_trace(trace_info)

        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )

        assert int(meta[LSI1_SK]) == 1709251200000
        # lsi2sk = end_time_ms = request_time + execution_duration
        assert int(meta[LSI2_SK]) == 1709251200000 + 500
        assert meta[LSI3_SK] == "OK#1709251200000"
        assert meta[LSI4_SK] == "my-trace"
        assert int(meta[LSI5_SK]) == 500

    def test_trace_gsi1_entry(self, tracking_store):
        """Verifies GSI1 reverse lookup TRACE#<id> -> EXP#<id>."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)

        tracking_store.start_trace(trace_info)

        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        assert meta[GSI1_PK] == f"{GSI1_TRACE_PREFIX}tr-abc123"
        assert meta[GSI1_SK] == f"{PK_EXPERIMENT_PREFIX}{exp_id}"

    def test_trace_client_request_ptr(self, tracking_store):
        """Verifies CLIENTPTR item for client_request_id."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, client_request_id="client-req-001")

        tracking_store.start_trace(trace_info)

        ptr = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#CLIENTPTR",
        )
        assert ptr is not None
        assert ptr[GSI1_PK] == f"{GSI1_CLIENT_PREFIX}client-req-001"
        assert ptr[GSI1_SK] == f"{GSI1_TRACE_PREFIX}tr-abc123"

    def test_trace_no_client_request_ptr_without_id(self, tracking_store):
        """No CLIENTPTR item when client_request_id is None."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, client_request_id=None)

        tracking_store.start_trace(trace_info)

        ptr = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#CLIENTPTR",
        )
        assert ptr is None

    def test_trace_request_metadata_items(self, tracking_store):
        """Verifies request metadata items are written as separate items."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "custom_key": "custom_val",
            },
        )

        tracking_store.start_trace(trace_info)

        # Check individual metadata items
        rmeta_name = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#RMETA#{TraceTagKey.TRACE_NAME}",
        )
        assert rmeta_name is not None
        assert rmeta_name["value"] == "my-trace"

        rmeta_custom = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#RMETA#custom_key",
        )
        assert rmeta_custom is not None
        assert rmeta_custom["value"] == "custom_val"

    def test_trace_tags_written(self, tracking_store):
        """Verifies that tags are written as separate items and denormalized on META."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            tags={
                "mlflow.user": "alice",
                "custom_tag": "val",
            },
        )

        tracking_store.start_trace(trace_info)

        # Check tag items
        tag_item = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#TAG#mlflow.user",
        )
        assert tag_item is not None
        assert tag_item["value"] == "alice"

        # Check denormalization on META for mlflow.* tags
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        assert meta["tags"]["mlflow.user"] == "alice"


class TestSetTraceTag:
    def test_set_trace_tag_with_denormalization(self, tracking_store):
        """Set tag, verify tag item + denormalization on META."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        tracking_store.set_trace_tag("tr-abc123", "mlflow.note", "important")

        # Verify tag item
        tag_item = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#TAG#mlflow.note",
        )
        assert tag_item is not None
        assert tag_item["value"] == "important"

        # Verify denormalization on META
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        assert meta["tags"]["mlflow.note"] == "important"

    def test_set_trace_tag_inherits_ttl(self, tracking_store):
        """Tag items should inherit the trace's TTL."""
        exp_id = _create_experiment(tracking_store)
        tracking_store._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": "TTL_POLICY",
                "trace_retention_days": 7,
            }
        )
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        tracking_store.set_trace_tag("tr-abc123", "mlflow.note", "important")

        tag_item = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#TAG#mlflow.note",
        )
        assert "ttl" in tag_item
        # TTL should be approximately the same as the trace META
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        assert int(tag_item["ttl"]) == int(meta["ttl"])

    def test_trace_fts_items_have_ttl(self, tracking_store):
        """FTS items written for trace tags must inherit the trace's TTL."""
        exp_id = _create_experiment(tracking_store)
        tracking_store._table.put_item(
            {
                "PK": PK_CONFIG,
                "SK": "TTL_POLICY",
                "trace_retention_days": 7,
            }
        )
        # Enable trigram indexing for trace tag values
        tracking_store._config.set_fts_trigram_fields(["trace_tag_value"])

        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        # Set a tag (triggers FTS item writes)
        tracking_store.set_trace_tag("tr-abc123", "mlflow.note", "important")

        # Read trace META to get the TTL value
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        trace_ttl = int(meta["ttl"])

        # Query FTS forward items for this trace
        from mlflow_dynamodbstore.dynamodb.schema import SK_FTS_PREFIX

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        fts_items = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        # Filter to items for trace entity "T#tr-abc123"
        trace_fts = [i for i in fts_items if "T#tr-abc123" in i["SK"]]
        assert len(trace_fts) > 0, "Expected FTS items to be written for trace tag"
        for fts_item in trace_fts:
            assert "ttl" in fts_item, f"FTS item missing ttl: {fts_item['SK']}"
            assert int(fts_item["ttl"]) == trace_ttl

    def test_trace_tag_overwrite_uses_fts_diff(self, tracking_store):
        """Overwriting a tag value should remove old FTS tokens and add new ones."""
        from mlflow_dynamodbstore.dynamodb.schema import SK_FTS_PREFIX

        exp_id = _create_experiment(tracking_store)
        tracking_store._config.set_fts_trigram_fields(["trace_tag_value"])

        trace_info = _make_trace_info(exp_id, tags={})
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

        # Set initial tag value "alpha"
        tracking_store.set_trace_tag("tr-abc123", "my.tag", "alpha")
        fts_after_first = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        first_fts = {i["SK"] for i in fts_after_first if "T#tr-abc123#my.tag" in i["SK"]}
        assert len(first_fts) > 0

        # Overwrite with "beta" -- old "alpha" tokens should be removed
        tracking_store.set_trace_tag("tr-abc123", "my.tag", "beta")
        fts_after_second = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        second_fts = {i["SK"] for i in fts_after_second if "T#tr-abc123#my.tag" in i["SK"]}

        # "alpha" and "beta" share no tokens, so old tokens should be gone
        assert first_fts.isdisjoint(second_fts), (
            "Old FTS tokens should have been removed on overwrite"
        )
        assert len(second_fts) > 0, "New FTS tokens should have been written"


class TestDeleteTraceTag:
    def test_delete_trace_tag(self, tracking_store):
        """Delete tag, verify removal from both tag item and META."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            tags={"mlflow.note": "important", "other": "val"},
        )
        tracking_store.start_trace(trace_info)

        # Verify tag exists before delete
        tag_item = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#TAG#mlflow.note",
        )
        assert tag_item is not None

        tracking_store.delete_trace_tag("tr-abc123", "mlflow.note")

        # Verify tag item deleted
        tag_item = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#TAG#mlflow.note",
        )
        assert tag_item is None

        # Verify removed from META denormalization
        meta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123",
        )
        assert "mlflow.note" not in meta.get("tags", {})


class TestLinkTracesToRun:
    def test_link_traces_to_run(self, tracking_store):
        """Link trace to run, verify request metadata mlflow.sourceRun."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        # Create a run to link to

        run = tracking_store.create_run(exp_id, "user", 1000, [], "test-run")
        run_id = run.info.run_id

        tracking_store.link_traces_to_run(["tr-abc123"], run_id)

        # Verify request metadata item written
        rmeta = tracking_store._table.get_item(
            pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
            sk=f"{SK_TRACE_PREFIX}tr-abc123#RMETA#{TraceMetadataKey.SOURCE_RUN}",
        )
        assert rmeta is not None
        assert rmeta["value"] == run_id

    def test_link_traces_to_run_multiple(self, tracking_store):
        """Link multiple traces to same run."""
        exp_id = _create_experiment(tracking_store)

        # Create two traces
        t1 = _make_trace_info(exp_id, trace_id="tr-001")
        t2 = _make_trace_info(exp_id, trace_id="tr-002")
        tracking_store.start_trace(t1)
        tracking_store.start_trace(t2)

        run = tracking_store.create_run(exp_id, "user", 1000, [], "test-run")
        run_id = run.info.run_id

        tracking_store.link_traces_to_run(["tr-001", "tr-002"], run_id)

        for tid in ["tr-001", "tr-002"]:
            rmeta = tracking_store._table.get_item(
                pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}",
                sk=f"{SK_TRACE_PREFIX}{tid}#RMETA#{TraceMetadataKey.SOURCE_RUN}",
            )
            assert rmeta is not None
            assert rmeta["value"] == run_id


class TestGetTraceInfo:
    def test_get_trace_info_full(self, tracking_store):
        """get_trace_info returns TraceInfo with tags and request_metadata."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "custom_meta": "meta_val",
            },
            tags={
                "mlflow.user": "bob",
                "custom_tag": "tag_val",
            },
        )
        tracking_store.start_trace(trace_info)

        fetched = tracking_store.get_trace_info("tr-abc123")
        assert fetched.trace_id == "tr-abc123"
        assert fetched.trace_location.mlflow_experiment.experiment_id == exp_id
        assert fetched.request_time == 1709251200000
        assert fetched.execution_duration == 500
        assert fetched.state == TraceState.OK
        assert fetched.trace_metadata[TraceTagKey.TRACE_NAME] == "my-trace"
        assert fetched.trace_metadata["custom_meta"] == "meta_val"
        assert fetched.tags["mlflow.user"] == "bob"
        assert fetched.tags["custom_tag"] == "tag_val"

    def test_get_trace_info_not_found(self, tracking_store):
        """get_trace_info raises for nonexistent trace."""
        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_trace_info("tr-nonexistent")


# ---------------------------------------------------------------------------
# Assessment CRUD tests
# ---------------------------------------------------------------------------


def _setup_trace(tracking_store):
    """Create an experiment and trace, return (experiment_id, trace_id)."""
    exp_id = _create_experiment(tracking_store)
    trace_info = _make_trace_info(exp_id)
    tracking_store.start_trace(trace_info)
    return exp_id, "tr-abc123"


class TestCreateAssessment:
    def test_create_assessment(self, tracking_store):
        """Create assessment with ULID ID, verify FTS tokens, verify TTL inheritance."""
        exp_id, trace_id = _setup_trace(tracking_store)

        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id=trace_id,
            feedback=FeedbackValue(value="good response"),
        )

        result = tracking_store.create_assessment(assessment)

        # Should have a ULID assessment_id assigned
        assert result.assessment_id is not None
        assert len(result.assessment_id) > 0
        assert result.name == "quality"
        assert result.trace_id == trace_id
        assert result.feedback.value == "good response"

        # Verify DynamoDB item exists with correct SK pattern
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        assess_sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{result.assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=assess_sk)
        assert item is not None

        # Verify assessment item inherits trace TTL
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}{trace_id}")
        assert "ttl" in item
        assert int(item["ttl"]) == int(meta["ttl"])

        # Verify FTS tokens exist for the feedback value text
        fts_items = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        assess_fts = [
            i for i in fts_items if f"T#{trace_id}#assess_{result.assessment_id}" in i["SK"]
        ]
        assert len(assess_fts) > 0, "Expected FTS items for assessment value text"

        # FTS items should also have TTL
        for fts_item in assess_fts:
            assert "ttl" in fts_item
            assert int(fts_item["ttl"]) == int(meta["ttl"])

    def test_create_assessment_with_expectation(self, tracking_store):
        """Create assessment with expectation value."""
        exp_id, trace_id = _setup_trace(tracking_store)

        assessment = Assessment(
            name="expected_output",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id=trace_id,
            expectation=ExpectationValue(value="the expected answer"),
        )

        result = tracking_store.create_assessment(assessment)
        assert result.assessment_id is not None
        assert result.expectation.value == "the expected answer"

        # Verify FTS tokens for expectation value text
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        fts_items = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        assess_fts = [
            i for i in fts_items if f"T#{trace_id}#assess_{result.assessment_id}" in i["SK"]
        ]
        assert len(assess_fts) > 0


class TestUpdateAssessment:
    def test_update_assessment(self, tracking_store):
        """Update assessment, verify FTS diff (old tokens removed, new tokens added)."""
        exp_id, trace_id = _setup_trace(tracking_store)

        # Create initial assessment
        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id=trace_id,
            feedback=FeedbackValue(value="alpha"),
        )
        created = tracking_store.create_assessment(assessment)
        assessment_id = created.assessment_id
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

        # Capture FTS tokens after create
        fts_after_create = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        old_fts = {
            i["SK"] for i in fts_after_create if f"T#{trace_id}#assess_{assessment_id}" in i["SK"]
        }
        assert len(old_fts) > 0

        # Update with new feedback value
        updated = tracking_store.update_assessment(
            trace_id=trace_id,
            assessment_id=assessment_id,
            feedback="beta",
        )
        assert updated.feedback.value == "beta"

        # Verify FTS diff: old tokens removed, new tokens added
        fts_after_update = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        new_fts = {
            i["SK"] for i in fts_after_update if f"T#{trace_id}#assess_{assessment_id}" in i["SK"]
        }
        assert old_fts.isdisjoint(new_fts), "Old FTS tokens should be removed on update"
        assert len(new_fts) > 0, "New FTS tokens should be written"

    def test_update_assessment_name(self, tracking_store):
        """Update assessment name."""
        _, trace_id = _setup_trace(tracking_store)

        assessment = Assessment(
            name="original",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id=trace_id,
            feedback=FeedbackValue(value="good"),
        )
        created = tracking_store.create_assessment(assessment)

        updated = tracking_store.update_assessment(
            trace_id=trace_id,
            assessment_id=created.assessment_id,
            name="renamed",
        )
        assert updated.name == "renamed"

        # Verify via get
        fetched = tracking_store.get_assessment(trace_id, created.assessment_id)
        assert fetched.name == "renamed"


class TestDeleteAssessment:
    def test_delete_assessment(self, tracking_store):
        """Delete assessment, verify FTS + FTS_REV cleanup via reverse index query."""
        exp_id, trace_id = _setup_trace(tracking_store)

        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id=trace_id,
            feedback=FeedbackValue(value="good response"),
        )
        created = tracking_store.create_assessment(assessment)
        assessment_id = created.assessment_id
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

        # Verify FTS items exist before delete
        fts_before = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        assess_fts_before = [
            i for i in fts_before if f"T#{trace_id}#assess_{assessment_id}" in i["SK"]
        ]
        assert len(assess_fts_before) > 0

        # Also verify FTS_REV items exist
        fts_rev_before = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_REV_PREFIX)
        assess_rev_before = [
            i for i in fts_rev_before if f"T#{trace_id}#assess_{assessment_id}" in i["SK"]
        ]
        assert len(assess_rev_before) > 0

        # Delete assessment
        tracking_store.delete_assessment(trace_id, assessment_id)

        # Verify assessment item is gone
        assess_sk = f"{SK_TRACE_PREFIX}{trace_id}#ASSESS#{assessment_id}"
        item = tracking_store._table.get_item(pk=pk, sk=assess_sk)
        assert item is None

        # Verify all FTS items cleaned up
        fts_after = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        assess_fts_after = [
            i for i in fts_after if f"T#{trace_id}#assess_{assessment_id}" in i["SK"]
        ]
        assert len(assess_fts_after) == 0

        # Verify all FTS_REV items cleaned up
        fts_rev_after = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_REV_PREFIX)
        assess_rev_after = [
            i for i in fts_rev_after if f"T#{trace_id}#assess_{assessment_id}" in i["SK"]
        ]
        assert len(assess_rev_after) == 0

    def test_delete_nonexistent_assessment(self, tracking_store):
        """Delete nonexistent assessment raises error."""
        _, trace_id = _setup_trace(tracking_store)

        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.delete_assessment(trace_id, "nonexistent-id")


class TestGetAssessment:
    def test_get_assessment(self, tracking_store):
        """Get assessment by ID, verify all fields roundtrip."""
        _, trace_id = _setup_trace(tracking_store)

        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id=trace_id,
            feedback=FeedbackValue(value="good response"),
            rationale="The response was helpful",
            metadata={"key1": "val1"},
        )
        created = tracking_store.create_assessment(assessment)

        fetched = tracking_store.get_assessment(trace_id, created.assessment_id)
        assert fetched.assessment_id == created.assessment_id
        assert fetched.name == "quality"
        assert fetched.trace_id == trace_id
        assert fetched.source.source_type == "HUMAN"
        assert fetched.source.source_id == "user1"
        assert fetched.feedback.value == "good response"
        assert fetched.rationale == "The response was helpful"
        assert fetched.metadata == {"key1": "val1"}

    def test_get_assessment_not_found(self, tracking_store):
        """Get nonexistent assessment raises error."""
        _, trace_id = _setup_trace(tracking_store)

        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_assessment(trace_id, "nonexistent-id")
