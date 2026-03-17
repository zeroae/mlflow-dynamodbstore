"""Tests for tag CLI commands."""

import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.tag import tag  # noqa: E402

_test_cli.add_command(tag)


@mock_aws
def test_tag_list():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["tag", "list"])
    assert result.exit_code == 0
    assert "mlflow.*" in result.output


@mock_aws
def test_tag_add():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["tag", "add", "team.*"])
    assert result.exit_code == 0
    # Verify it was added
    result = runner.invoke(_test_cli, ["tag", "list"])
    assert "team.*" in result.output


@mock_aws
def test_tag_remove():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    # Add then remove
    runner.invoke(_test_cli, ["tag", "add", "team.*"])
    result = runner.invoke(_test_cli, ["tag", "remove", "team.*"])
    assert result.exit_code == 0
    result = runner.invoke(_test_cli, ["tag", "list"])
    assert "team.*" not in result.output


@mock_aws
def test_tag_add_per_experiment():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(_test_cli, ["tag", "add", "--experiment-id", "01ABC", "team.*"])
    assert result.exit_code == 0
    # Verify effective patterns include both global and per-experiment
    result = runner.invoke(_test_cli, ["tag", "list", "--experiment-id", "01ABC"])
    assert result.exit_code == 0
    assert "mlflow.*" in result.output
    assert "team.*" in result.output


@mock_aws
def test_tag_remove_per_experiment():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    # Add per-experiment pattern
    runner.invoke(_test_cli, ["tag", "add", "--experiment-id", "01ABC", "team.*"])
    # Remove it
    result = runner.invoke(_test_cli, ["tag", "remove", "--experiment-id", "01ABC", "team.*"])
    assert result.exit_code == 0
    # Verify it's gone from effective patterns
    result = runner.invoke(_test_cli, ["tag", "list", "--experiment-id", "01ABC"])
    assert "team.*" not in result.output


@mock_aws
def test_tag_add_duplicate():
    """Adding a pattern that already exists should not create duplicates."""
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    runner.invoke(_test_cli, ["tag", "add", "team.*"])
    runner.invoke(_test_cli, ["tag", "add", "team.*"])
    result = runner.invoke(_test_cli, ["tag", "list"])
    # Count occurrences - should appear exactly once
    assert result.output.strip().split("\n").count("team.*") == 1
