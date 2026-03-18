"""E2E tests for query_trace_metrics.

Since MlflowClient does not expose query_trace_metrics and there is no public
REST endpoint, these tests create traces via the MLflow SDK (which exercises the
full HTTP server stack), then query metrics via the server's internal store.
"""

import uuid

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.constant import SpanMetricKey, TraceMetricKey

pytestmark = pytest.mark.e2e


def _uid():
    return uuid.uuid4().hex[:8]


class TestQueryTraceMetricsE2E:
    def test_trace_count_via_sdk(self, mlflow_server):
        """Create traces via SDK decorator, verify trace_count metric."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-count-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="e2e-count-func")
        def my_func(x):
            return x * 2

        for i in range(3):
            my_func(i)

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)

        # Verify traces were created
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) >= 3

        # Query trace metrics via the server store
        store = _get_tracking_store()
        result = store.query_trace_metrics(
            experiment_ids=[exp.experiment_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] >= 3.0

    def test_span_metrics_via_sdk(self, mlflow_server):
        """Create hierarchical spans via SDK, verify span count metric."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-spans-{_uid()}"
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
        embed.set_outputs({"embedding": [0.1, 0.2]})
        embed.end()

        llm = mlflow.start_span_no_context(
            name="generate",
            span_type=SpanType.LLM,
            parent_span=root,
            inputs={"messages": [{"role": "user", "content": "test"}]},
        )
        llm.set_outputs({"response": "answer"})
        llm.end()

        root.set_outputs({"response": "answer"})
        root.end()

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)
        traces = client.search_traces(locations=[exp.experiment_id])
        assert len(traces) >= 1

        store = _get_tracking_store()
        result = store.query_trace_metrics(
            experiment_ids=[exp.experiment_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
            dimensions=["span_type"],
        )
        types = {dp.dimensions["span_type"]: dp.values["COUNT"] for dp in result}
        assert "CHAIN" in types
        assert "EMBEDDING" in types
        assert "LLM" in types

    def test_latency_avg_via_sdk(self, mlflow_server):
        """Create traces via SDK, verify latency AVG metric is reasonable."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-latency-{_uid()}"
        mlflow.set_experiment(exp_name)

        @mlflow.trace(name="e2e-latency-func")
        def my_func(x):
            return x + 1

        my_func(1)

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)

        store = _get_tracking_store()
        result = store.query_trace_metrics(
            experiment_ids=[exp.experiment_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.LATENCY,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        )
        assert len(result) == 1
        # Latency should be a positive number (milliseconds)
        assert result[0].values["AVG"] >= 0.0
