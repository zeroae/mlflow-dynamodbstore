"""Tests for fts CLI commands."""

import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.fts import fts  # noqa: E402

_test_cli.add_command(fts)


@mock_aws
def test_fts_list():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["fts", "list"])
    assert result.exit_code == 0
    assert "experiment_name" in result.output
    assert "No additional fields configured." in result.output


@mock_aws
def test_fts_add():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["fts", "add", "run_param_value"])
    assert result.exit_code == 0
    assert "Added field: run_param_value" in result.output

    result = runner.invoke(_test_cli, ["fts", "list"])
    assert result.exit_code == 0
    assert "run_param_value" in result.output
    assert "Additional fields:" in result.output


@mock_aws
def test_fts_add_idempotent():
    """Adding the same field twice should not duplicate it."""
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    runner.invoke(_test_cli, ["fts", "add", "my_field"])
    runner.invoke(_test_cli, ["fts", "add", "my_field"])
    result = runner.invoke(_test_cli, ["fts", "list"])
    assert result.exit_code == 0
    # Should appear only once
    assert result.output.count("my_field") == 1
