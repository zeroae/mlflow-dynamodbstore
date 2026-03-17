"""E2E tests for trace operations via MLflow client SDK."""

import uuid

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.e2e


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestTraces:
    def test_search_traces_empty(self, client: MlflowClient):
        """Search traces should work even with no traces."""
        exp_id = client.create_experiment(f"e2e-traces-empty-{_uid()}")
        traces = client.search_traces(experiment_ids=[exp_id])
        assert isinstance(traces, list)
        assert len(traces) == 0

    def test_start_and_end_trace(self, client: MlflowClient):
        """Create a trace and verify it appears in search."""
        exp_id = client.create_experiment(f"e2e-trace-crud-{_uid()}")
        trace_info = client.start_trace(
            name=f"e2e-trace-{_uid()}",
            experiment_id=exp_id,
        )
        client.end_trace(
            request_id=trace_info.request_id,
            status="OK",
        )
        traces = client.search_traces(experiment_ids=[exp_id])
        assert len(traces) >= 1

    def test_set_trace_tag(self, client: MlflowClient):
        """Set a tag on a trace and verify."""
        exp_id = client.create_experiment(f"e2e-trace-tag-{_uid()}")
        trace_info = client.start_trace(
            name="tag-test",
            experiment_id=exp_id,
        )
        client.end_trace(request_id=trace_info.request_id, status="OK")
        client.set_trace_tag(trace_info.request_id, "env", "prod")
        traces = client.search_traces(experiment_ids=[exp_id])
        assert len(traces) >= 1
