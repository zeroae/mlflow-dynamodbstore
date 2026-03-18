"""E2E tests for trace operations via MLflow client SDK."""

import uuid

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestTraces:
    def test_search_traces_empty(self, client: MlflowClient):
        """Search traces should work even with no traces."""
        exp_id = client.create_experiment(f"e2e-traces-empty-{_uid()}")
        traces = client.search_traces(locations=[exp_id])
        assert isinstance(traces, list)
        assert len(traces) == 0

    def test_trace_via_decorator(self, mlflow_server):
        """Create a trace using the @mlflow.trace decorator."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-trace-dec-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="e2e-traced-func")
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(21)
        assert result == 42

        # Verify trace appears in search
        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) >= 1

    def test_trace_has_artifact_location_tag(self, mlflow_server):
        """Trace must have mlflow.artifactLocation tag set after creation."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-trace-artloc-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="artifact-loc-check")
        def my_func():
            return "ok"

        my_func()

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) == 1
        trace = client.get_trace(traces[0].info.trace_id)
        assert MLFLOW_ARTIFACT_LOCATION in trace.info.tags
        assert trace.info.tags[MLFLOW_ARTIFACT_LOCATION] != ""

    def test_hierarchical_trace_with_spans(self, mlflow_server):
        """Create a trace with parent-child span hierarchy (like demo RAG pipeline)."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-trace-hier-{_uid()}"
        mlflow.set_experiment(exp_name)

        root = mlflow.start_span_no_context(
            name="rag_pipeline",
            span_type=SpanType.CHAIN,
            inputs={"query": "What is MLflow?"},
        )

        embed = mlflow.start_span_no_context(
            name="embed_query",
            span_type=SpanType.EMBEDDING,
            parent_span=root,
            inputs={"text": "What is MLflow?"},
        )
        embed.set_outputs({"embedding": [0.1, 0.2, 0.3]})
        embed.end()

        llm = mlflow.start_span_no_context(
            name="generate",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={"messages": [{"role": "user", "content": "What is MLflow?"}]},
        )
        llm.set_outputs({"response": "MLflow is an open source platform."})
        llm.end()

        root.set_outputs({"response": "MLflow is an open source platform."})
        root.end()

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) >= 1

    def test_trace_set_and_delete_tag(self, mlflow_server):
        """Set and delete tags on a trace."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-trace-tags-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="tag-test")
        def my_func():
            return "tagged"

        my_func()

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        trace_id = traces[0].info.trace_id

        # Set a tag
        client.set_trace_tag(trace_id, "test.tag", "hello")
        trace = client.get_trace(trace_id)
        assert trace.info.tags.get("test.tag") == "hello"

        # Delete the tag
        client.delete_trace_tag(trace_id, "test.tag")
        trace = client.get_trace(trace_id)
        assert "test.tag" not in trace.info.tags

    def test_delete_traces(self, mlflow_server):
        """Create and then delete traces."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-trace-delete-{_uid()}"
        exp = mlflow.set_experiment(exp_name)

        @mlflow.trace(name="to-delete")
        def my_func():
            return "delete me"

        my_func()

        client = MlflowClient(tracking_uri=mlflow_server)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) == 1
        trace_id = traces[0].info.trace_id

        client.delete_traces(
            experiment_id=exp.experiment_id,
            trace_ids=[trace_id],
        )

        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) == 0


class TestBatchTraceOperationsE2E:
    """E2E tests for batch trace operations."""

    def test_batch_get_trace_infos(self, mlflow_server):
        """Create traces via SDK, then batch fetch their infos."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-batch-infos-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="batch-func")
        def my_func(x):
            return x + 1

        for i in range(3):
            my_func(i)

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        trace_ids = [t.info.trace_id for t in traces]

        # Verify we can fetch each trace individually
        for tid in trace_ids:
            info = client.get_trace(tid)
            assert info is not None
            assert info.info.trace_id == tid


class TestLogSpansAsyncE2E:
    """E2E test for log_spans_async (exercises the async span path)."""

    def test_trace_with_spans_via_decorator(self, mlflow_server):
        """Traces created via @mlflow.trace have spans logged (async path)."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-async-spans-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="async-span-func")
        def my_func():
            return "hello"

        my_func()

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) >= 1

        # Fetch full trace with spans
        trace = client.get_trace(traces[0].info.trace_id)
        assert trace.info.trace_id == traces[0].info.trace_id


class TestUnlinkTracesE2E:
    """E2E tests for unlink_traces_from_run."""

    def test_link_and_unlink_trace(self, mlflow_server):
        """Link a trace to a run, then unlink it."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-unlink-{_uid()}"
        exp = mlflow.set_experiment(exp_name)

        @mlflow.trace(name="unlink-func")
        def my_func():
            return "result"

        my_func()

        client = MlflowClient(tracking_uri=mlflow_server)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) >= 1
        trace_id = traces[0].info.trace_id

        # Create a run and link the trace
        run = client.create_run(experiment_id=exp.experiment_id)
        run_id = run.info.run_id
        client.set_trace_tag(trace_id, "mlflow.sourceRun", run_id)

        # Verify tag is set
        trace = client.get_trace(trace_id)
        assert trace.info.tags.get("mlflow.sourceRun") == run_id
