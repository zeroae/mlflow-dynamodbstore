"""Tests for trace metadata CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import time

import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceLocationType, TraceState
from mlflow.entities.trace_location import MlflowExperimentLocation
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
        from mlflow.exceptions import MlflowException

        with pytest.raises(MlflowException, match="does not exist"):
            tracking_store.get_trace_info("tr-nonexistent")
