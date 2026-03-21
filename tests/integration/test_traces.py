"""Integration tests for trace lifecycle via moto server."""

from __future__ import annotations

import time

from mlflow.entities import (
    Assessment,
    TraceInfo,
    TraceLocation,
    TraceLocationType,
    TraceState,
)
from mlflow.entities.assessment import AssessmentSource, FeedbackValue
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey


def _make_trace_info(
    experiment_id: str,
    trace_id: str,
    *,
    state: TraceState = TraceState.OK,
    request_time: int | None = None,
    execution_duration: int = 100,
    tags: dict[str, str] | None = None,
    trace_metadata: dict[str, str] | None = None,
    client_request_id: str | None = None,
) -> TraceInfo:
    """Helper to build a TraceInfo for testing."""
    if request_time is None:
        request_time = int(time.time() * 1000)
    if trace_metadata is None:
        trace_metadata = {TraceTagKey.TRACE_NAME: f"trace-{trace_id}"}
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=request_time,
        execution_duration=execution_duration,
        state=state,
        tags=tags or {},
        trace_metadata=trace_metadata,
        client_request_id=client_request_id,
    )


class TestTraceLifecycle:
    def test_start_trace_and_get_info(self, tracking_store):
        """Start trace -> get_trace_info round-trip."""
        exp_id = tracking_store.create_experiment("trace-exp", artifact_location="s3://b")
        ti = _make_trace_info(exp_id, "tr-001", state=TraceState.OK, execution_duration=42)
        tracking_store.start_trace(ti)

        fetched = tracking_store.get_trace_info("tr-001")
        assert fetched.trace_id == "tr-001"
        assert fetched.state == TraceState.OK
        assert fetched.execution_duration == 42
        assert fetched.trace_location.mlflow_experiment.experiment_id == exp_id
        # Request metadata round-trips
        assert fetched.trace_metadata[TraceTagKey.TRACE_NAME] == "trace-tr-001"

    def test_trace_ttl(self, tracking_store):
        """Verify TTL on trace items via direct table read."""
        exp_id = tracking_store.create_experiment("ttl-exp", artifact_location="s3://b")
        ti = _make_trace_info(exp_id, "tr-ttl")
        tracking_store.start_trace(ti)

        meta = tracking_store._table.get_item(pk=f"EXP#{exp_id}", sk="T#tr-ttl")
        assert meta is not None
        ttl_val = int(meta["ttl"])
        # TTL should be in the future (within ~30 days from now)
        now = int(time.time())
        assert ttl_val > now
        assert ttl_val < now + 31 * 86400

    def test_set_and_delete_trace_tag(self, tracking_store):
        """Set tag -> verify denormalization -> delete tag -> verify cleanup."""
        exp_id = tracking_store.create_experiment("tag-exp", artifact_location="s3://b")
        ti = _make_trace_info(exp_id, "tr-tag")
        tracking_store.start_trace(ti)

        # Set a tag
        tracking_store.set_trace_tag("tr-tag", "env", "prod")

        # Verify via get_trace_info
        info = tracking_store.get_trace_info("tr-tag")
        assert info.tags["env"] == "prod"

        # Verify tag item exists in DynamoDB
        tag_item = tracking_store._table.get_item(pk=f"EXP#{exp_id}", sk="T#tr-tag#TAG#env")
        assert tag_item is not None
        assert tag_item["value"] == "prod"

        # Delete the tag
        tracking_store.delete_trace_tag("tr-tag", "env")

        # Verify tag is gone
        info2 = tracking_store.get_trace_info("tr-tag")
        assert "env" not in info2.tags

        tag_item2 = tracking_store._table.get_item(pk=f"EXP#{exp_id}", sk="T#tr-tag#TAG#env")
        assert tag_item2 is None

    def test_tag_fts_lifecycle(self, tracking_store):
        """Set tag -> FTS items created -> delete tag -> FTS items cleaned."""
        exp_id = tracking_store.create_experiment("fts-exp", artifact_location="s3://b")

        # Enable FTS for trace_tag_value
        tracking_store._config.set_fts_trigram_fields(["trace_tag_value"])

        ti = _make_trace_info(exp_id, "tr-fts")
        tracking_store.start_trace(ti)

        tracking_store.set_trace_tag("tr-fts", "description", "hello world")

        # Check that FTS reverse items exist for this tag
        pk = f"EXP#{exp_id}"
        fts_rev_items = tracking_store._table.query(
            pk=pk, sk_prefix="FTS_REV#T#tr-fts#description#"
        )
        assert len(fts_rev_items) > 0, "Expected FTS reverse items for tag value"

        # Delete the tag
        tracking_store.delete_trace_tag("tr-fts", "description")

        # FTS items should be cleaned up
        fts_rev_items_after = tracking_store._table.query(
            pk=pk, sk_prefix="FTS_REV#T#tr-fts#description#"
        )
        assert len(fts_rev_items_after) == 0, (
            "FTS reverse items should be cleaned up after tag delete"
        )

    def test_assessment_lifecycle(self, tracking_store):
        """Create -> update -> delete assessment. FTS maintained throughout."""
        exp_id = tracking_store.create_experiment("assess-exp", artifact_location="s3://b")
        ti = _make_trace_info(exp_id, "tr-assess")
        tracking_store.start_trace(ti)

        # Create assessment
        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="user1"),
            trace_id="tr-assess",
            feedback=FeedbackValue(value="good answer"),
        )
        created = tracking_store.create_assessment(assessment)
        assert created.assessment_id is not None
        aid = created.assessment_id

        # Get assessment
        fetched = tracking_store.get_assessment("tr-assess", aid)
        assert fetched.name == "quality"

        # Update assessment
        updated = tracking_store.update_assessment(
            trace_id="tr-assess",
            assessment_id=aid,
            feedback="excellent answer",
            rationale="very detailed",
        )
        assert updated.rationale == "very detailed"

        # Verify update persisted
        fetched2 = tracking_store.get_assessment("tr-assess", aid)
        assert fetched2.feedback.value == "excellent answer"

        # Delete assessment
        tracking_store.delete_assessment("tr-assess", aid)

        # Verify deletion
        pk = f"EXP#{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk=f"T#tr-assess#ASSESS#{aid}")
        assert item is None

    def test_search_traces_by_status(self, tracking_store):
        """Create traces with different statuses -> search by status."""
        exp_id = tracking_store.create_experiment("search-status", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-ok", state=TraceState.OK, request_time=base_time)
        )
        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-err", state=TraceState.ERROR, request_time=base_time + 1)
        )
        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-ok2", state=TraceState.OK, request_time=base_time + 2)
        )

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="status = 'OK'",
        )
        trace_ids = {t.trace_id for t in results}
        assert "tr-ok" in trace_ids
        assert "tr-ok2" in trace_ids
        assert "tr-err" not in trace_ids

    def test_search_traces_by_timestamp(self, tracking_store):
        """Create traces -> search with timestamp filter -> verify order."""
        exp_id = tracking_store.create_experiment("search-ts", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        for i in range(5):
            tracking_store.start_trace(
                _make_trace_info(
                    exp_id,
                    f"tr-ts-{i}",
                    request_time=base_time + i * 1000,
                )
            )

        # Search with timestamp range (should get traces 2, 3, 4)
        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string=f"timestamp > {base_time + 1500}",
        )
        trace_ids = {t.trace_id for t in results}
        assert "tr-ts-0" not in trace_ids
        assert "tr-ts-1" not in trace_ids
        # Traces 2-4 have request_time >= base_time+2000, which is > base_time+1500
        assert "tr-ts-2" in trace_ids
        assert "tr-ts-3" in trace_ids
        assert "tr-ts-4" in trace_ids

    def test_search_traces_pagination(self, tracking_store):
        """Create many traces -> search with pagination -> verify all found."""
        exp_id = tracking_store.create_experiment("search-page", artifact_location="s3://b")
        base_time = int(time.time() * 1000)
        num_traces = 7

        for i in range(num_traces):
            tracking_store.start_trace(
                _make_trace_info(exp_id, f"tr-pg-{i}", request_time=base_time + i * 1000)
            )

        # Page through with max_results=3
        all_trace_ids: set[str] = set()
        page_token = None
        pages = 0
        while True:
            results, page_token = tracking_store.search_traces(
                experiment_ids=[exp_id],
                max_results=3,
                page_token=page_token,
            )
            all_trace_ids.update(t.trace_id for t in results)
            pages += 1
            if page_token is None:
                break
            # Safety: prevent infinite loop
            if pages > 10:
                break

        assert len(all_trace_ids) == num_traces
        for i in range(num_traces):
            assert f"tr-pg-{i}" in all_trace_ids

    def test_search_traces_by_tag(self, tracking_store):
        """Create traces with tags -> search by tag filter."""
        exp_id = tracking_store.create_experiment("search-tag", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        tracking_store.start_trace(
            _make_trace_info(
                exp_id,
                "tr-tagged-1",
                request_time=base_time,
                tags={"env": "prod"},
            )
        )
        tracking_store.start_trace(
            _make_trace_info(
                exp_id,
                "tr-tagged-2",
                request_time=base_time + 1,
                tags={"env": "staging"},
            )
        )
        tracking_store.start_trace(
            _make_trace_info(
                exp_id,
                "tr-tagged-3",
                request_time=base_time + 2,
                tags={"env": "prod"},
            )
        )

        results, _ = tracking_store.search_traces(
            experiment_ids=[exp_id],
            filter_string="tag.env = 'prod'",
        )
        trace_ids = {t.trace_id for t in results}
        assert "tr-tagged-1" in trace_ids
        assert "tr-tagged-3" in trace_ids
        assert "tr-tagged-2" not in trace_ids

    def test_link_traces_to_run(self, tracking_store):
        """Create trace + run -> link -> verify request metadata."""
        exp_id = tracking_store.create_experiment("link-exp", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        tracking_store.start_trace(_make_trace_info(exp_id, "tr-link", request_time=base_time))

        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=base_time,
            tags=[],
            run_name="linked-run",
        )
        run_id = run.info.run_id

        tracking_store.link_traces_to_run(["tr-link"], run_id)

        # Verify request metadata
        info = tracking_store.get_trace_info("tr-link")
        assert info.trace_metadata[TraceMetadataKey.SOURCE_RUN] == run_id

    def test_delete_traces(self, tracking_store):
        """Create trace with tags/assessments -> delete -> verify all items gone."""
        exp_id = tracking_store.create_experiment("delete-exp", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-del", request_time=base_time, tags={"k": "v"})
        )

        # Add assessment
        assessment = Assessment(
            name="quality",
            source=AssessmentSource(source_type="HUMAN", source_id="tester"),
            trace_id="tr-del",
            feedback=FeedbackValue(value="ok"),
        )
        tracking_store.create_assessment(assessment)

        # Verify trace exists
        info = tracking_store.get_trace_info("tr-del")
        assert info.trace_id == "tr-del"

        # Delete
        deleted_count = tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-del"])
        assert deleted_count == 1

        # Verify all items gone
        pk = f"EXP#{exp_id}"
        remaining = tracking_store._table.query(pk=pk, sk_prefix="T#tr-del")
        assert len(remaining) == 0, f"Expected no trace items, got {len(remaining)}"

        # FTS reverse items should be gone too
        fts_rev = tracking_store._table.query(pk=pk, sk_prefix="FTS_REV#T#tr-del")
        assert len(fts_rev) == 0, "FTS reverse items should be cleaned up"

    def test_clientptr_lifecycle(self, tracking_store):
        """Create trace with client_request_id -> verify CLIENTPTR -> delete -> verify gone."""
        exp_id = tracking_store.create_experiment("client-exp", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        tracking_store.start_trace(
            _make_trace_info(
                exp_id,
                "tr-client",
                request_time=base_time,
                client_request_id="client-req-42",
            )
        )

        pk = f"EXP#{exp_id}"

        # Verify CLIENTPTR item exists
        ptr_item = tracking_store._table.get_item(pk=pk, sk="T#tr-client#CLIENTPTR")
        assert ptr_item is not None
        assert ptr_item["gsi1pk"] == "CLIENT#client-req-42"

        # Verify trace_info has client_request_id
        info = tracking_store.get_trace_info("tr-client")
        assert info.client_request_id == "client-req-42"

        # Delete trace
        tracking_store.delete_traces(experiment_id=exp_id, trace_ids=["tr-client"])

        # CLIENTPTR should be gone
        ptr_after = tracking_store._table.get_item(pk=pk, sk="T#tr-client#CLIENTPTR")
        assert ptr_after is None

        # META should be gone
        meta_after = tracking_store._table.get_item(pk=pk, sk="T#tr-client")
        assert meta_after is None


class TestBatchTraceOperations:
    """Integration tests for batch_get_traces and batch_get_trace_infos."""

    def test_batch_get_trace_infos_round_trip(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-batch-infos", artifact_location="s3://b")
        trace_ids = []
        for i in range(3):
            tid = f"tr-int-batch-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid, request_time=1000 + i))
            trace_ids.append(tid)

        result = tracking_store.batch_get_trace_infos(trace_ids)
        assert len(result) == 3
        assert {t.trace_id for t in result} == set(trace_ids)

    def test_batch_get_traces_without_spans(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-batch-traces", artifact_location="s3://b")
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-int-bt-1"))

        result = tracking_store.batch_get_traces(["tr-int-bt-1"])
        assert len(result) == 1
        assert result[0].info.trace_id == "tr-int-bt-1"
        assert result[0].data.spans == []

    def test_batch_nonexistent_excluded(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-batch-ne", artifact_location="s3://b")
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-int-exists"))

        result = tracking_store.batch_get_trace_infos(["tr-int-exists", "tr-int-ghost"])
        assert len(result) == 1


class TestSessionTracker:
    """Integration tests for session tracker and find_completed_sessions."""

    def test_session_tracker_created(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-sessions", artifact_location="s3://b")
        tracking_store.start_trace(
            _make_trace_info(
                exp_id,
                "tr-int-sess-1",
                request_time=1000,
                trace_metadata={
                    TraceTagKey.TRACE_NAME: "my-trace",
                    TraceMetadataKey.TRACE_SESSION: "session-int-1",
                },
            )
        )

        pk = f"EXP#{exp_id}"
        item = tracking_store._table.get_item(pk=pk, sk="SESS#session-int-1")
        assert item is not None
        assert item["session_id"] == "session-int-1"

    def test_find_completed_sessions_round_trip(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-find-sess", artifact_location="s3://b")
        for i, ts in enumerate([1000, 2000, 3000]):
            tracking_store.start_trace(
                _make_trace_info(
                    exp_id,
                    f"tr-int-fs-{i}",
                    request_time=ts,
                    trace_metadata={
                        TraceTagKey.TRACE_NAME: "my-trace",
                        TraceMetadataKey.TRACE_SESSION: "sess-find-1",
                    },
                )
            )

        result = tracking_store.find_completed_sessions(
            experiment_id=exp_id,
            min_last_trace_timestamp_ms=0,
            max_last_trace_timestamp_ms=9999,
        )
        assert len(result) == 1
        assert result[0].session_id == "sess-find-1"
        assert result[0].first_trace_timestamp_ms == 1000
        assert result[0].last_trace_timestamp_ms == 3000


class TestLinkUnlinkOperations:
    """Integration tests for link/unlink operations."""

    def test_link_prompts_to_trace(self, tracking_store):
        import json

        from mlflow.entities.model_registry import PromptVersion

        exp_id = tracking_store.create_experiment("test-link-prompts", artifact_location="s3://b")
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-int-lp-1"))

        pv = PromptVersion(name="my-prompt", version=1, template="hello {name}")
        tracking_store.link_prompts_to_trace("tr-int-lp-1", [pv])

        fetched = tracking_store.get_trace_info("tr-int-lp-1")
        versions = json.loads(fetched.tags["mlflow.linkedPrompts"])
        assert len(versions) == 1
        assert versions[0]["name"] == "my-prompt"

    def test_unlink_traces_from_run(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-unlink", artifact_location="s3://b")
        base_time = int(time.time() * 1000)
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-int-ul-1", request_time=base_time))

        run = tracking_store.create_run(
            experiment_id=exp_id,
            user_id="user",
            start_time=base_time,
            tags=[],
            run_name="test-run",
        )
        run_id = run.info.run_id

        tracking_store.link_traces_to_run(["tr-int-ul-1"], run_id)
        fetched = tracking_store.get_trace_info("tr-int-ul-1")
        assert TraceMetadataKey.SOURCE_RUN in fetched.trace_metadata

        tracking_store.unlink_traces_from_run(["tr-int-ul-1"], run_id)
        fetched = tracking_store.get_trace_info("tr-int-ul-1")
        assert TraceMetadataKey.SOURCE_RUN not in fetched.trace_metadata


class TestLogSpansIntegration:
    """Integration tests for log_spans."""

    def test_log_spans_creates_item(self, tracking_store):
        import json
        from unittest.mock import MagicMock

        exp_id = tracking_store.create_experiment("test-log-spans", artifact_location="s3://b")
        tracking_store.start_trace(_make_trace_info(exp_id, "tr-int-ls-1"))

        span = MagicMock()
        span.trace_id = "tr-int-ls-1"
        span.to_dict.return_value = {
            "name": "root",
            "trace_id": "tr-int-ls-1",
            "span_id": "span-1",
        }
        tracking_store.log_spans(exp_id, [span])

        pk = f"EXP#{exp_id}"
        cached = tracking_store._table.get_item(pk=pk, sk="T#tr-int-ls-1#SPANS")
        assert cached is not None
        data = json.loads(cached["data"])
        assert len(data) == 1
        assert data[0]["name"] == "root"


class TestCorrelationIntegration:
    """Integration tests for calculate_trace_filter_correlation."""

    def test_basic_correlation(self, tracking_store):
        exp_id = tracking_store.create_experiment("test-corr", artifact_location="s3://b")
        base_time = int(time.time() * 1000)

        # 3 traces: 2 with tag.a=1, 2 with tag.b=2, 1 with both
        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-c-0", request_time=base_time, tags={"a": "1", "b": "2"})
        )
        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-c-1", request_time=base_time + 1, tags={"a": "1"})
        )
        tracking_store.start_trace(
            _make_trace_info(exp_id, "tr-c-2", request_time=base_time + 2, tags={"b": "2"})
        )

        result = tracking_store.calculate_trace_filter_correlation(
            experiment_ids=[exp_id],
            filter_string1="tag.a = '1'",
            filter_string2="tag.b = '2'",
        )
        assert result.total_count == 3
        assert result.filter1_count == 2
        assert result.filter2_count == 2
        assert result.joint_count == 1
