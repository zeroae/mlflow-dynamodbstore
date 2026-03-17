"""Tests for trace metadata CRUD and assessment CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

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
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

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

    def test_start_trace_sets_artifact_location_tag(self, tracking_store):
        """start_trace must set mlflow.artifactLocation tag from experiment."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)

        result = tracking_store.start_trace(trace_info)

        assert MLFLOW_ARTIFACT_LOCATION in result.tags
        assert "traces/tr-abc123/artifacts" in result.tags[MLFLOW_ARTIFACT_LOCATION]

    def test_start_trace_with_none_tags(self, tracking_store):
        """start_trace should handle tags=None and still set artifact location."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, tags=None)

        result = tracking_store.start_trace(trace_info)

        assert result.tags is not None
        assert MLFLOW_ARTIFACT_LOCATION in result.tags

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
        assert meta[LSI3_SK] == f"OK#{1709251200000:020d}"
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


# ---------------------------------------------------------------------------
# search_traces tests
# ---------------------------------------------------------------------------


def _create_traces(tracking_store, experiment_id, count=5):
    """Create several traces with varied attributes for search tests.

    Returns a list of trace_ids in creation order.
    """
    traces = []
    base_time = 1709251200000
    states = ["OK", "ERROR", "OK", "OK", "ERROR"]
    names = ["chat-completion", "embedding-gen", "chat-summary", "translate", "error-handler"]
    durations = [500, 100, 1500, 200, 800]

    for i in range(count):
        trace_id = f"tr-search-{i:03d}"
        trace_info = _make_trace_info(
            experiment_id,
            trace_id=trace_id,
            request_time=base_time + i * 1000,
            execution_duration=durations[i],
            state=TraceState(states[i]),
            trace_metadata={
                TraceTagKey.TRACE_NAME: names[i],
                "client_request_id": f"client-{i}",
            },
            tags={
                "user_tag": f"value-{i}",
                "mlflow.user": "alice",
            },
        )
        tracking_store.start_trace(trace_info)
        traces.append(trace_id)
    return traces


class TestSearchTraces:
    def test_search_by_timestamp(self, tracking_store):
        """LSI1 query - search by timestamp range."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        # Search for traces with timestamp > base_time + 2000
        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="attribute.timestamp_ms > 1709251202000",
        )
        # Should find traces 3 and 4 (timestamps 3000 and 4000 above base)
        result_ids = [t.trace_id for t in results]
        assert "tr-search-003" in result_ids
        assert "tr-search-004" in result_ids
        assert "tr-search-000" not in result_ids
        assert "tr-search-001" not in result_ids

    def test_search_by_status(self, tracking_store):
        """LSI3 composite: status#timestamp_ms."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="attribute.status = 'ERROR'",
        )
        result_ids = [t.trace_id for t in results]
        # Traces 1 and 4 have ERROR status
        assert set(result_ids) == {"tr-search-001", "tr-search-004"}

    def test_search_by_execution_time(self, tracking_store):
        """LSI5 duration sort."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            order_by=["execution_time_ms ASC"],
        )
        durations = [t.execution_duration for t in results]
        assert durations == sorted(durations)

    def test_search_by_trace_name(self, tracking_store):
        """LSI4 begins_with for prefix ILIKE."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="attribute.name LIKE 'chat%'",
        )
        result_ids = [t.trace_id for t in results]
        # "chat-completion" and "chat-summary" match
        assert "tr-search-000" in result_ids
        assert "tr-search-002" in result_ids
        assert len(result_ids) == 2

    def test_search_by_tag(self, tracking_store):
        """Denormalized tag -> FilterExpression."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="tag.user_tag = 'value-2'",
        )
        result_ids = [t.trace_id for t in results]
        assert result_ids == ["tr-search-002"]

    def test_search_by_metadata(self, tracking_store):
        """Request metadata filter."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="request_metadata.client_request_id = 'client-3'",
        )
        result_ids = [t.trace_id for t in results]
        assert result_ids == ["tr-search-003"]

    def test_search_fts_keyword(self, tracking_store):
        """FTS word-level search on tag/metadata text."""
        exp_id = _create_experiment(tracking_store)
        # Enable trigram indexing for trace tag values
        tracking_store._config.set_fts_trigram_fields(["trace_tag_value"])

        _create_traces(tracking_store, exp_id)

        # Set a distinctive tag on one trace
        tracking_store.set_trace_tag("tr-search-002", "description", "quantum computing research")

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="tag.description LIKE '%quantum%'",
        )
        result_ids = [t.trace_id for t in results]
        assert "tr-search-002" in result_ids

    def test_search_pagination(self, tracking_store):
        """Pagination with page_token."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        # Get first page of 2
        results1, token1 = tracking_store.search_traces(
            experiment_ids=[exp_id],
            max_results=2,
        )
        assert len(results1) == 2
        assert token1 is not None

        # Get second page
        results2, token2 = tracking_store.search_traces(
            experiment_ids=[exp_id],
            max_results=2,
            page_token=token1,
        )
        assert len(results2) == 2
        assert token2 is not None

        # Get third page
        results3, token3 = tracking_store.search_traces(
            experiment_ids=[exp_id],
            max_results=2,
            page_token=token2,
        )
        assert len(results3) == 1
        assert token3 is None

        # All results should be unique
        all_ids = [t.trace_id for t in results1 + results2 + results3]
        assert len(set(all_ids)) == 5

    def test_search_default_order(self, tracking_store):
        """Default order is by timestamp descending."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
        )
        timestamps = [t.request_time for t in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_search_by_end_time(self, tracking_store):
        """Filter by attribute.end_time_ms (stored as lsi2sk)."""
        exp_id = _create_experiment(tracking_store)
        _create_traces(tracking_store, exp_id)

        # end_time_ms = request_time + execution_duration
        # trace 0: 1709251200000 + 500 = 1709251200500
        # trace 1: 1709251201000 + 100 = 1709251201100
        # trace 2: 1709251202000 + 1500 = 1709251203500
        # trace 3: 1709251203000 + 200 = 1709251203200
        # trace 4: 1709251204000 + 800 = 1709251204800
        # Filter for end_time_ms > 1709251203400 should return traces 2 and 4
        results, token = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="attribute.end_time_ms > 1709251203400",
        )
        result_ids = {t.trace_id for t in results}
        assert result_ids == {"tr-search-002", "tr-search-004"}


# ---------------------------------------------------------------------------
# delete_traces tests
# ---------------------------------------------------------------------------


class TestDeleteTraces:
    def test_delete_traces_removes_meta(self, tracking_store):
        """Verify trace META item is deleted."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        # Verify META exists
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        assert meta is not None

        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-abc123"])

        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        assert meta is None

    def test_delete_traces_removes_tags(self, tracking_store):
        """Verify trace tag items are deleted."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, tags={"mlflow.user": "alice", "env": "prod"})
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        tags_before = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}tr-abc123#TAG#"
        )
        assert len(tags_before) >= 2

        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-abc123"])

        tags_after = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}tr-abc123#TAG#"
        )
        assert len(tags_after) == 0

    def test_delete_traces_removes_request_metadata(self, tracking_store):
        """Verify request metadata items are deleted."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "custom_key": "custom_val",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        rmeta_before = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}tr-abc123#RMETA#"
        )
        assert len(rmeta_before) >= 2

        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-abc123"])

        rmeta_after = tracking_store._table.query(
            pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}tr-abc123#RMETA#"
        )
        assert len(rmeta_after) == 0

    def test_delete_traces_removes_assessments(self, tracking_store):
        """Verify assessment items are deleted."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id="tr-abc123",
            feedback=FeedbackValue(value="good"),
        )
        created = tracking_store.create_assessment(assessment)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        assess_sk = f"{SK_TRACE_PREFIX}tr-abc123#ASSESS#{created.assessment_id}"
        assert tracking_store._table.get_item(pk=pk, sk=assess_sk) is not None

        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-abc123"])

        assert tracking_store._table.get_item(pk=pk, sk=assess_sk) is None

    def test_delete_traces_removes_clientptr(self, tracking_store):
        """Verify CLIENTPTR item is deleted."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, client_request_id="client-req-001")
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        ptr = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123#CLIENTPTR")
        assert ptr is not None

        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-abc123"])

        ptr = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123#CLIENTPTR")
        assert ptr is None

    def test_delete_traces_removes_fts_items(self, tracking_store):
        """Verify FTS forward + FTS_REV items are cleaned up."""
        exp_id = _create_experiment(tracking_store)
        tracking_store._config.set_fts_trigram_fields(["trace_tag_value"])

        trace_info = _make_trace_info(exp_id, tags={})
        tracking_store.start_trace(trace_info)
        tracking_store.set_trace_tag("tr-abc123", "description", "quantum computing")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"

        # Verify FTS items exist
        fts_before = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        trace_fts_before = [i for i in fts_before if "T#tr-abc123" in i["SK"]]
        assert len(trace_fts_before) > 0

        fts_rev_before = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_REV_PREFIX)
        trace_rev_before = [i for i in fts_rev_before if "T#tr-abc123" in i["SK"]]
        assert len(trace_rev_before) > 0

        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-abc123"])

        # All FTS items should be gone
        fts_after = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        trace_fts_after = [i for i in fts_after if "T#tr-abc123" in i["SK"]]
        assert len(trace_fts_after) == 0

        fts_rev_after = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_REV_PREFIX)
        trace_rev_after = [i for i in fts_rev_after if "T#tr-abc123" in i["SK"]]
        assert len(trace_rev_after) == 0

    def test_delete_traces_returns_count(self, tracking_store):
        """Verify returns count of deleted traces."""
        exp_id = _create_experiment(tracking_store)
        for i in range(3):
            trace_info = _make_trace_info(exp_id, trace_id=f"tr-del-{i}")
            tracking_store.start_trace(trace_info)

        count = tracking_store.delete_traces(
            experiment_id=exp_id,
            trace_ids=["tr-del-0", "tr-del-1", "tr-del-2"],
        )
        assert count == 3


# ---------------------------------------------------------------------------
# get_trace tests (X-Ray span proxy + lazy caching)
# ---------------------------------------------------------------------------

SAMPLE_XRAY_TRACE = {
    "Id": "tr-abc123",
    "Segments": [
        {
            "Id": "seg-001",
            "Document": json.dumps(
                {
                    "id": "seg-001",
                    "name": "ChatModel",
                    "trace_id": "tr-abc123",
                    "parent_id": None,
                    "start_time": 1709251200.0,
                    "end_time": 1709251200.5,
                    "annotations": {
                        "mlflow_spanType": "LLM",
                        "mlflow_spanName": "ChatModel",
                        "mlflow_spanStatus": "OK",
                    },
                    "metadata": {
                        "mlflow": {
                            "inputs": {"prompt": "hi"},
                            "outputs": {"response": "hello"},
                        }
                    },
                }
            ),
        }
    ],
}

SAMPLE_XRAY_MULTI_SPAN = {
    "Id": "tr-abc123",
    "Segments": [
        {
            "Id": "seg-001",
            "Document": json.dumps(
                {
                    "id": "seg-001",
                    "name": "ChatModel",
                    "trace_id": "tr-abc123",
                    "parent_id": None,
                    "start_time": 1709251200.0,
                    "end_time": 1709251200.5,
                    "annotations": {
                        "mlflow_spanType": "LLM",
                        "mlflow_spanName": "ChatModel",
                        "mlflow_spanStatus": "OK",
                    },
                    "metadata": {"mlflow": {"inputs": {}, "outputs": {}}},
                }
            ),
        },
        {
            "Id": "seg-002",
            "Document": json.dumps(
                {
                    "id": "seg-002",
                    "name": "Retriever",
                    "trace_id": "tr-abc123",
                    "parent_id": "seg-001",
                    "start_time": 1709251200.1,
                    "end_time": 1709251200.3,
                    "annotations": {
                        "mlflow_spanType": "RETRIEVER",
                        "mlflow_spanName": "Retriever",
                        "mlflow_spanStatus": "OK",
                    },
                    "metadata": {"mlflow": {"inputs": {}, "outputs": {}}},
                }
            ),
        },
    ],
}


class TestGetTraceWithSpans:
    def test_cache_miss_fetches_from_xray(self, tracking_store):
        """First get_trace -> X-Ray fetch -> cache write -> return spans."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_TRACE]

        # Inject mock xray client
        tracking_store._xray_client_instance = mock_xray

        result = tracking_store.get_trace("tr-abc123")

        # Verify X-Ray was called
        mock_xray.batch_get_traces.assert_called_once_with(["tr-abc123"])

        # Verify return type and data
        from mlflow.entities.trace import Trace

        assert isinstance(result, Trace)
        assert result.info.trace_id == "tr-abc123"
        assert len(result.data.spans) == 1

        span = result.data.spans[0]
        assert span.name == "ChatModel"

    def test_cache_hit_skips_xray(self, tracking_store):
        """Second get_trace -> cache hit -> no X-Ray call."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_TRACE]
        tracking_store._xray_client_instance = mock_xray

        # First call populates cache
        tracking_store.get_trace("tr-abc123")
        assert mock_xray.batch_get_traces.call_count == 1

        # Second call should hit cache, no X-Ray call
        result = tracking_store.get_trace("tr-abc123")
        assert mock_xray.batch_get_traces.call_count == 1  # unchanged
        assert len(result.data.spans) == 1

    def test_span_attributes_denormalized_on_cache(self, tracking_store):
        """On cache, span_types/statuses/names sets written to META."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_MULTI_SPAN]
        tracking_store._xray_client_instance = mock_xray

        tracking_store.get_trace("tr-abc123")

        # Read META and check denormalized span attributes
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        assert "span_types" in meta
        assert "span_statuses" in meta
        assert "span_names" in meta

        # Check actual values (DynamoDB stores sets)
        assert {"LLM", "RETRIEVER"} == set(meta["span_types"])
        assert {"OK"} == set(meta["span_statuses"])
        assert {"ChatModel", "Retriever"} == set(meta["span_names"])

    def test_span_name_fts_on_cache(self, tracking_store):
        """On cache, FTS items written for span names."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_TRACE]
        tracking_store._xray_client_instance = mock_xray

        tracking_store.get_trace("tr-abc123")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        fts_items = tracking_store._table.query(pk=pk, sk_prefix=SK_FTS_PREFIX)
        span_fts = [i for i in fts_items if "T#tr-abc123#spans" in i["SK"]]
        assert len(span_fts) > 0, "Expected FTS items for span names"

        # FTS items should have TTL
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        trace_ttl = int(meta["ttl"])
        for fts_item in span_fts:
            assert "ttl" in fts_item
            assert int(fts_item["ttl"]) == trace_ttl

    def test_xray_returns_empty(self, tracking_store):
        """If X-Ray returns nothing (expired), return trace without spans."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = []
        tracking_store._xray_client_instance = mock_xray

        result = tracking_store.get_trace("tr-abc123")

        assert result.info.trace_id == "tr-abc123"
        assert len(result.data.spans) == 0

    def test_spans_cache_item_has_ttl(self, tracking_store):
        """Verify the SPANS cache item inherits the trace's TTL."""
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

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_TRACE]
        tracking_store._xray_client_instance = mock_xray

        tracking_store.get_trace("tr-abc123")

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        spans_item = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123#SPANS")
        assert spans_item is not None
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        assert int(spans_item["ttl"]) == int(meta["ttl"])


class TestHybridSearchTraces:
    """Tests for hybrid DynamoDB + X-Ray search with span-level filters."""

    def _create_trace_and_cache_spans(self, tracking_store, exp_id, trace_id="tr-abc123"):
        """Create a trace, then call get_trace to cache spans and denormalize."""
        trace_info = _make_trace_info(exp_id, trace_id=trace_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.batch_get_traces.return_value = [SAMPLE_XRAY_TRACE]
        tracking_store._xray_client_instance = mock_xray

        # get_trace caches spans and denormalizes span_types/span_names/span_statuses
        tracking_store.get_trace(trace_id)
        return trace_info

    def test_span_type_filter_cached(self, tracking_store):
        """Span type filter on cached traces uses denormalized span_types on META."""
        exp_id = _create_experiment(tracking_store)
        self._create_trace_and_cache_spans(tracking_store, exp_id)

        # Verify denormalized span_types exist on META
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        assert "span_types" in meta
        assert "LLM" in meta["span_types"]

        # Search with span type filter - should find the cached trace via DynamoDB
        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.type = 'LLM'",
        )
        assert len(results) == 1
        assert results[0].trace_id == "tr-abc123"

    def test_span_type_filter_no_match(self, tracking_store):
        """Span type filter that doesn't match returns empty."""
        exp_id = _create_experiment(tracking_store)
        self._create_trace_and_cache_spans(tracking_store, exp_id)

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.type = 'AGENT'",
        )
        assert len(results) == 0

    def test_span_type_filter_uncached_via_xray(self, tracking_store):
        """Span type filter on uncached traces queries X-Ray."""
        exp_id = _create_experiment(tracking_store)

        # Create trace but DON'T call get_trace (no cached spans)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        # Mock X-Ray client to return the trace
        mock_xray = MagicMock()
        mock_xray.get_trace_summaries.return_value = [{"Id": "tr-abc123", "ResponseTime": 0.5}]
        tracking_store._xray_client_instance = mock_xray

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.type = 'LLM'",
        )
        assert len(results) == 1
        assert results[0].trace_id == "tr-abc123"

        # Verify X-Ray was called with the right filter
        mock_xray.get_trace_summaries.assert_called_once()
        call_kwargs = mock_xray.get_trace_summaries.call_args
        assert 'annotation.mlflow_spanType = "LLM"' in call_kwargs.kwargs.get(
            "filter_expression", call_kwargs[1].get("filter_expression", "")
        ) or 'annotation.mlflow_spanType = "LLM"' in str(call_kwargs)

    def test_span_filter_union_dedup(self, tracking_store):
        """Results from DynamoDB and X-Ray are unioned and deduplicated."""
        exp_id = _create_experiment(tracking_store)

        # Create trace1 with cached spans (found via DynamoDB)
        self._create_trace_and_cache_spans(tracking_store, exp_id, trace_id="tr-cached")

        # Create trace2 without cached spans (found via X-Ray)
        trace_info2 = _make_trace_info(exp_id, trace_id="tr-uncached")
        tracking_store.start_trace(trace_info2)

        # Mock X-Ray to return BOTH traces (the cached one + the uncached one)
        mock_xray = MagicMock()
        mock_xray.get_trace_summaries.return_value = [
            {"Id": "tr-cached", "ResponseTime": 0.5},
            {"Id": "tr-uncached", "ResponseTime": 0.3},
        ]
        # Also need batch_get_traces for the cached trace's get_trace call
        # (already called above, so just set up for X-Ray search)
        tracking_store._xray_client_instance = mock_xray

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.type = 'LLM'",
        )

        trace_ids = [r.trace_id for r in results]
        # Both should be present, no duplicates
        assert "tr-cached" in trace_ids
        assert "tr-uncached" in trace_ids
        assert len(trace_ids) == len(set(trace_ids))

    def test_span_name_like_cached(self, tracking_store):
        """span.name LIKE '%Chat%' on cached traces uses denormalized span_names."""
        exp_id = _create_experiment(tracking_store)
        self._create_trace_and_cache_spans(tracking_store, exp_id)

        # Verify denormalized span_names exist
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        meta = tracking_store._table.get_item(pk=pk, sk=f"{SK_TRACE_PREFIX}tr-abc123")
        assert "span_names" in meta
        assert "ChatModel" in meta["span_names"]

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.name LIKE '%Chat%'",
        )
        assert len(results) == 1
        assert results[0].trace_id == "tr-abc123"

    def test_span_name_like_no_match(self, tracking_store):
        """span.name LIKE '%xyz%' on cached traces returns empty when no match."""
        exp_id = _create_experiment(tracking_store)
        self._create_trace_and_cache_spans(tracking_store, exp_id)

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.name LIKE '%nonexistent%'",
        )
        assert len(results) == 0

    def test_no_span_filter_unchanged(self, tracking_store):
        """search_traces without span filters works as before."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="attribute.status = 'OK'",
        )
        assert len(results) == 1
        assert results[0].trace_id == "tr-abc123"

    def test_xray_error_returns_empty(self, tracking_store):
        """If X-Ray call fails, gracefully return DynamoDB-only results."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id)
        tracking_store.start_trace(trace_info)

        mock_xray = MagicMock()
        mock_xray.get_trace_summaries.side_effect = Exception("X-Ray unavailable")
        tracking_store._xray_client_instance = mock_xray

        # Should not raise, just return empty (no cached span data, X-Ray fails)
        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="span.type = 'LLM'",
        )
        assert len(results) == 0


class TestStartTraceSessionTracker:
    """Tests for session tracker upsert in start_trace."""

    def test_trace_with_session_creates_session_tracker(self, tracking_store):
        """start_trace with mlflow.traceSession metadata creates a SESS# item."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-sess-1",
            request_time=1000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "session-abc",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-abc")
        assert item is not None
        assert item["session_id"] == "session-abc"
        assert int(item["trace_count"]) == 1
        assert int(item["first_trace_timestamp_ms"]) == 1000
        assert int(item["last_trace_timestamp_ms"]) == 1000

    def test_trace_without_session_no_session_tracker(self, tracking_store):
        """start_trace without mlflow.traceSession does NOT create a SESS# item."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(exp_id, trace_id="tr-no-sess")
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#any")
        assert item is None

    def test_multiple_traces_same_session_increments(self, tracking_store):
        """Multiple traces in same session increment trace_count and update timestamps."""
        exp_id = _create_experiment(tracking_store)
        for i, (tid, ts) in enumerate(
            [
                ("tr-s1", 1000),
                ("tr-s2", 2000),
                ("tr-s3", 3000),
            ]
        ):
            trace_info = _make_trace_info(
                exp_id,
                trace_id=tid,
                request_time=ts,
                trace_metadata={
                    TraceTagKey.TRACE_NAME: "my-trace",
                    "mlflow.traceSession": "session-xyz",
                },
            )
            tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-xyz")
        assert item is not None
        assert int(item["trace_count"]) == 3
        assert int(item["first_trace_timestamp_ms"]) == 1000
        assert int(item["last_trace_timestamp_ms"]) == 3000

    def test_session_tracker_has_gsi2_attributes(self, tracking_store):
        """Session tracker item has GSI2 PK/SK for find_completed_sessions queries."""
        from mlflow_dynamodbstore.dynamodb.schema import GSI2_PK, GSI2_SK

        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-gsi2",
            request_time=5000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "session-gsi2",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-gsi2")
        assert item[GSI2_PK] == f"SESSIONS#default#{exp_id}"
        # GSI2 SK is zero-padded string for correct lexicographic ordering
        assert item[GSI2_SK] == f"{5000:020d}"

    def test_multiple_traces_updates_gsi2_sk_to_last_timestamp(self, tracking_store):
        """GSI2 SK reflects last_trace_timestamp_ms after multiple traces."""
        from mlflow_dynamodbstore.dynamodb.schema import GSI2_SK

        exp_id = _create_experiment(tracking_store)
        for tid, ts in [("tr-g1", 1000), ("tr-g2", 2000), ("tr-g3", 3000)]:
            trace_info = _make_trace_info(
                exp_id,
                trace_id=tid,
                request_time=ts,
                trace_metadata={
                    TraceTagKey.TRACE_NAME: "my-trace",
                    "mlflow.traceSession": "session-gsi2-multi",
                },
            )
            tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-gsi2-multi")
        assert item[GSI2_SK] == f"{3000:020d}"

    def test_session_tracker_has_ttl(self, tracking_store):
        """Session tracker item inherits trace TTL."""
        exp_id = _create_experiment(tracking_store)
        trace_info = _make_trace_info(
            exp_id,
            trace_id="tr-ttl",
            request_time=1000,
            trace_metadata={
                TraceTagKey.TRACE_NAME: "my-trace",
                "mlflow.traceSession": "session-ttl",
            },
        )
        tracking_store.start_trace(trace_info)

        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-ttl")
        assert "ttl" in item


class TestBatchGetTraceInfos:
    """Tests for batch_get_trace_infos."""

    def _create_traces(self, tracking_store, exp_id, count=3):
        """Helper: create multiple traces, return their IDs."""
        trace_ids = []
        for i in range(count):
            tid = f"tr-batch-info-{i}"
            trace_info = _make_trace_info(
                exp_id,
                trace_id=tid,
                request_time=1000 + i * 100,
            )
            tracking_store.start_trace(trace_info)
            trace_ids.append(tid)
        return trace_ids

    def test_batch_get_single_trace(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos(tids)
        assert len(result) == 1
        assert result[0].trace_id == tids[0]

    def test_batch_get_multiple_traces(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=3)
        result = tracking_store.batch_get_trace_infos(tids)
        assert len(result) == 3
        returned_ids = {t.trace_id for t in result}
        assert returned_ids == set(tids)

    def test_nonexistent_trace_excluded(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos(tids + ["nonexistent-trace"])
        assert len(result) == 1
        assert result[0].trace_id == tids[0]

    def test_empty_list_returns_empty(self, tracking_store):
        result = tracking_store.batch_get_trace_infos([])
        assert result == []

    def test_duplicate_ids_deduplicated(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos([tids[0], tids[0]])
        assert len(result) == 1

    def test_with_location_skips_resolution(self, tracking_store):
        exp_id = _create_experiment(tracking_store)
        tids = self._create_traces(tracking_store, exp_id, count=1)
        result = tracking_store.batch_get_trace_infos(tids, location=exp_id)
        assert len(result) == 1
        assert result[0].trace_id == tids[0]
