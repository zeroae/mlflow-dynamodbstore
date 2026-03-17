"""Tests for workspace CLI commands."""

import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import PK_WORKSPACE_PREFIX, SK_WORKSPACE_META
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.workspace import workspace  # noqa: E402

_test_cli.add_command(workspace)


@mock_aws
def test_delete_workspace_rejects_default():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["workspace", "delete", "--workspace", "default"],
    )
    assert result.exit_code != 0


@mock_aws
def test_delete_workspace_not_found():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["workspace", "delete", "--workspace", "nonexistent"],
    )
    assert result.exit_code != 0


@mock_aws
def test_delete_workspace_soft():
    ensure_stack_exists(table_name="test", region="us-east-1")
    table = DynamoDBTable(table_name="test", region="us-east-1")
    table.put_item(
        {
            "PK": f"{PK_WORKSPACE_PREFIX}test-ws",
            "SK": SK_WORKSPACE_META,
            "name": "test-ws",
            "status": "active",
        }
    )

    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["workspace", "delete", "--workspace", "test-ws", "--mode", "soft"],
    )
    assert result.exit_code == 0
    assert "deleted" in result.output.lower()

    # Verify status changed
    item = table.get_item(pk=f"{PK_WORKSPACE_PREFIX}test-ws", sk=SK_WORKSPACE_META)
    assert item is not None
    assert item["status"] == "deleted"


@mock_aws
def test_delete_workspace_cascade_requires_confirmation():
    ensure_stack_exists(table_name="test", region="us-east-1")
    table = DynamoDBTable(table_name="test", region="us-east-1")
    table.put_item(
        {
            "PK": f"{PK_WORKSPACE_PREFIX}test-ws",
            "SK": SK_WORKSPACE_META,
            "name": "test-ws",
            "status": "active",
        }
    )

    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["workspace", "delete", "--workspace", "test-ws", "--mode", "cascade"],
        input="n\n",
    )
    # Should abort when user answers no
    assert result.exit_code != 0


@mock_aws
def test_delete_workspace_cascade_with_yes():
    ensure_stack_exists(table_name="test", region="us-east-1")
    table = DynamoDBTable(table_name="test", region="us-east-1")
    table.put_item(
        {
            "PK": f"{PK_WORKSPACE_PREFIX}test-ws",
            "SK": SK_WORKSPACE_META,
            "name": "test-ws",
            "status": "active",
        }
    )

    runner = CliRunner()
    result = runner.invoke(
        _test_cli,
        ["workspace", "delete", "--workspace", "test-ws", "--mode", "cascade", "--yes"],
    )
    assert result.exit_code == 0
    assert "Deleted" in result.output

    # Verify workspace META is gone
    item = table.get_item(pk=f"{PK_WORKSPACE_PREFIX}test-ws", sk=SK_WORKSPACE_META)
    assert item is None
