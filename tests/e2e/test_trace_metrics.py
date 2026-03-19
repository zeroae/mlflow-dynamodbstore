"""E2E tests for query_trace_metrics.

Since MlflowClient does not expose query_trace_metrics and there is no public
REST endpoint, these tests create traces via the MLflow SDK (which exercises the
full HTTP server stack), then query metrics via the server's internal store.
"""

import time
import uuid

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.server.handlers import _get_tracking_store
from mlflow.tracing.constant import TraceMetricKey

pytestmark = pytest.mark.e2e


def _uid():
    return uuid.uuid4().hex[:8]


class TestQueryTraceMetricsE2E:
    def test_trace_count_via_sdk(self, mlflow_server):
        """Create traces via SDK decorator, verify trace_count metric."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-count-{_uid()}"
        mlflow.set_experiment(exp_name)

        start_ms = int(time.time() * 1000)

        @mlflow.trace(name="e2e-count-func")
        def my_func(x):
            return x * 2

        for i in range(3):
            my_func(i)

        end_ms = int(time.time() * 1000) + 1000

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
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] >= 3.0

    def test_latency_avg_via_sdk(self, mlflow_server):
        """Create traces via SDK, verify latency AVG metric is reasonable."""
        mlflow.set_tracking_uri(mlflow_server)
        exp_name = f"e2e-qtm-latency-{_uid()}"
        mlflow.set_experiment(exp_name)

        start_ms = int(time.time() * 1000)

        @mlflow.trace(name="e2e-latency-func")
        def my_func(x):
            return x + 1

        my_func(1)

        end_ms = int(time.time() * 1000) + 1000

        client = MlflowClient(tracking_uri=mlflow_server)
        exp = client.get_experiment_by_name(exp_name)

        store = _get_tracking_store()
        result = store.query_trace_metrics(
            experiment_ids=[exp.experiment_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.LATENCY,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        assert len(result) == 1
        # Latency should be a positive number (milliseconds)
        assert result[0].values["AVG"] >= 0.0
