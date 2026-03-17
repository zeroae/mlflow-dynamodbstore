"""E2E tests for trace operations via MLflow client SDK."""

import uuid

import mlflow
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

    @pytest.mark.xfail(reason="Trace artifact location tag not propagated through REST layer")
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
        traces = client.search_traces(experiment_ids=[exp.experiment_id])
        assert len(traces) >= 1
