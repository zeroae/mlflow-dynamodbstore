"""Tests for ttl CLI commands."""

import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.ttl import ttl  # noqa: E402

_test_cli.add_command(ttl)


@mock_aws
def test_ttl_show():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["ttl", "show"])
    assert result.exit_code == 0, result.output
    assert "soft_deleted_retention_days" in result.output
    assert "90 days" in result.output


@mock_aws
def test_ttl_set():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["ttl", "set", "--soft-deleted-retention-days", "60"],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(_test_cli, ["ttl", "show"])
    assert "60 days" in result.output


@mock_aws
def test_ttl_set_no_values():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["ttl", "set"])
    assert result.exit_code == 0, result.output
    assert "No values provided" in result.output


def _create_orphaned_experiment(region: str, table_name: str) -> str:
    """Create an experiment with a run, then delete the META item.

    Returns the experiment ID.
    """
    from mlflow_dynamodbstore.dynamodb.schema import PK_EXPERIMENT_PREFIX, SK_EXPERIMENT_META
    from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    store = DynamoDBTrackingStore(f"dynamodb://{region}/{table_name}", "/tmp/artifacts")
    exp_id = store.create_experiment("orphan-test", artifact_location="s3://bucket")
    store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")

    # Simulate TTL deleting the META item
    table = DynamoDBTable(table_name=table_name, region=region)
    table.delete_item(pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}", sk=SK_EXPERIMENT_META)
    return exp_id


@mock_aws
def test_ttl_cleanup_dry_run():
    from mlflow_dynamodbstore.dynamodb.schema import PK_EXPERIMENT_PREFIX
    from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable

    ensure_stack_exists(table_name="test", region="us-east-1")
    exp_id = _create_orphaned_experiment("us-east-1", "test")

    runner = CliRunner()
    result = runner.invoke(_test_cli, ["ttl", "cleanup", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "orphaned" in result.output.lower()
    assert "Dry run" in result.output

    # Items should NOT have a TTL set
    table = DynamoDBTable(table_name="test", region="us-east-1")
    items = table.query(pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}")
    assert len(items) > 0
    for item in items:
        assert "ttl" not in item


@mock_aws
def test_ttl_cleanup_sets_ttl():
    from mlflow_dynamodbstore.dynamodb.schema import PK_EXPERIMENT_PREFIX
    from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable

    ensure_stack_exists(table_name="test", region="us-east-1")
    exp_id = _create_orphaned_experiment("us-east-1", "test")

    runner = CliRunner()
    result = runner.invoke(_test_cli, ["ttl", "cleanup"])
    assert result.exit_code == 0, result.output
    assert "Set TTL on" in result.output

    # Verify children now have a TTL attribute
    table = DynamoDBTable(table_name="test", region="us-east-1")
    items = table.query(pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}")
    assert len(items) > 0
    for item in items:
        assert "ttl" in item


@mock_aws
def test_ttl_cleanup_skips_live_experiments():
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    ensure_stack_exists(table_name="test", region="us-east-1")

    store = DynamoDBTrackingStore("dynamodb://us-east-1/test", "/tmp/artifacts")
    store.create_experiment("alive-test", artifact_location="s3://bucket")

    runner = CliRunner()
    result = runner.invoke(_test_cli, ["ttl", "cleanup", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry run: 0 orphaned items found" in result.output
