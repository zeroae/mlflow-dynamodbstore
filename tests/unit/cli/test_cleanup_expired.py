"""Tests for cleanup-expired CLI command."""

from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import PK_EXPERIMENT_PREFIX, SK_EXPERIMENT_META
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


def _create_orphaned_experiment(region: str, table_name: str) -> str:
    """Create an experiment with a run, then delete the META item.

    Returns the experiment ID.
    """
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    store = DynamoDBTrackingStore(f"dynamodb://{region}/{table_name}", "/tmp/artifacts")
    exp_id = store.create_experiment("orphan-test", artifact_location="s3://bucket")
    store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")

    # Simulate TTL deleting the META item
    table = DynamoDBTable(table_name=table_name, region=region)
    table.delete_item(pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}", sk=SK_EXPERIMENT_META)
    return exp_id


@mock_aws
def test_cleanup_expired_dry_run():
    ensure_stack_exists(table_name="test", region="us-east-1")
    exp_id = _create_orphaned_experiment("us-east-1", "test")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["cleanup-expired", "--table", "test", "--region", "us-east-1", "--dry-run"],
    )
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
def test_cleanup_expired_sets_ttl():
    ensure_stack_exists(table_name="test", region="us-east-1")
    exp_id = _create_orphaned_experiment("us-east-1", "test")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["cleanup-expired", "--table", "test", "--region", "us-east-1"],
    )
    assert result.exit_code == 0, result.output
    assert "Set TTL on" in result.output

    # Verify children now have a TTL attribute
    table = DynamoDBTable(table_name="test", region="us-east-1")
    items = table.query(pk=f"{PK_EXPERIMENT_PREFIX}{exp_id}")
    assert len(items) > 0
    for item in items:
        assert "ttl" in item


@mock_aws
def test_cleanup_expired_skips_live_experiments():
    ensure_stack_exists(table_name="test", region="us-east-1")

    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    store = DynamoDBTrackingStore("dynamodb://us-east-1/test", "/tmp/artifacts")
    exp_id = store.create_experiment("alive-test", artifact_location="s3://bucket")
    store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["cleanup-expired", "--table", "test", "--region", "us-east-1", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run: 0 orphaned items found" in result.output
