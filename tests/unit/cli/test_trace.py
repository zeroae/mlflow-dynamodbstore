"""Tests for trace CLI commands."""

import click
from click.testing import CliRunner
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import (
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
)
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.constant import TraceTagKey
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.trace import trace  # noqa: E402

_test_cli.add_command(trace)


def _make_trace_info(experiment_id: str, trace_id: str = "tr-cache-test") -> TraceInfo:
    """Helper to build a TraceInfo for tests."""
    return TraceInfo(
        trace_id=trace_id,
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


@mock_aws
def test_trace_cache_no_traces():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["trace", "cache", "--experiment-id", "01ABC"],
    )
    assert result.exit_code == 0
    assert "Cached: 0" in result.output
    assert "Already cached: 0" in result.output


@mock_aws
def test_trace_cache_with_traces():
    ensure_stack_exists(table_name="test", region="us-east-1")
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    store = DynamoDBTrackingStore("dynamodb://us-east-1/test", "/tmp/artifacts")
    exp_id = store.create_experiment("test-exp", artifact_location="s3://b")

    # Create a trace using the correct API
    trace_info = _make_trace_info(exp_id)
    store.start_trace(trace_info)

    runner = CliRunner()
    # get_trace calls X-Ray which won't work under moto, but the CLI handles errors gracefully
    result = runner.invoke(
        _test_cli,
        ["trace", "cache", "--experiment-id", exp_id],
    )
    assert result.exit_code == 0


@mock_aws
def test_trace_cache_multiple_experiments():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        [
            "trace",
            "cache",
            "--experiment-id",
            "exp1",
            "--experiment-id",
            "exp2",
        ],
    )
    assert result.exit_code == 0
    assert "Cached: 0" in result.output
