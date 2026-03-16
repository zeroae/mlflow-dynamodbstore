"""Tests for denormalize-tags CLI commands."""

from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@mock_aws
def test_denormalize_tags_list():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["denormalize-tags", "list", "--table", "test", "--region", "us-east-1"]
    )
    assert result.exit_code == 0
    assert "mlflow.*" in result.output


@mock_aws
def test_denormalize_tags_add():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["denormalize-tags", "add", "--table", "test", "--region", "us-east-1", "team.*"]
    )
    assert result.exit_code == 0
    # Verify it was added
    result = runner.invoke(
        cli, ["denormalize-tags", "list", "--table", "test", "--region", "us-east-1"]
    )
    assert "team.*" in result.output


@mock_aws
def test_denormalize_tags_remove():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    # Add then remove
    runner.invoke(
        cli, ["denormalize-tags", "add", "--table", "test", "--region", "us-east-1", "team.*"]
    )
    result = runner.invoke(
        cli, ["denormalize-tags", "remove", "--table", "test", "--region", "us-east-1", "team.*"]
    )
    assert result.exit_code == 0
    result = runner.invoke(
        cli, ["denormalize-tags", "list", "--table", "test", "--region", "us-east-1"]
    )
    assert "team.*" not in result.output


@mock_aws
def test_denormalize_tags_add_per_experiment():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "denormalize-tags",
            "add",
            "--table",
            "test",
            "--region",
            "us-east-1",
            "--experiment-id",
            "01ABC",
            "team.*",
        ],
    )
    assert result.exit_code == 0
    # Verify effective patterns include both global and per-experiment
    result = runner.invoke(
        cli,
        [
            "denormalize-tags",
            "list",
            "--table",
            "test",
            "--region",
            "us-east-1",
            "--experiment-id",
            "01ABC",
        ],
    )
    assert result.exit_code == 0
    assert "mlflow.*" in result.output
    assert "team.*" in result.output


@mock_aws
def test_denormalize_tags_remove_per_experiment():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    # Add per-experiment pattern
    runner.invoke(
        cli,
        [
            "denormalize-tags",
            "add",
            "--table",
            "test",
            "--region",
            "us-east-1",
            "--experiment-id",
            "01ABC",
            "team.*",
        ],
    )
    # Remove it
    result = runner.invoke(
        cli,
        [
            "denormalize-tags",
            "remove",
            "--table",
            "test",
            "--region",
            "us-east-1",
            "--experiment-id",
            "01ABC",
            "team.*",
        ],
    )
    assert result.exit_code == 0
    # Verify it's gone from effective patterns
    result = runner.invoke(
        cli,
        [
            "denormalize-tags",
            "list",
            "--table",
            "test",
            "--region",
            "us-east-1",
            "--experiment-id",
            "01ABC",
        ],
    )
    assert "team.*" not in result.output


@mock_aws
def test_denormalize_tags_add_duplicate():
    """Adding a pattern that already exists should not create duplicates."""
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    runner.invoke(
        cli, ["denormalize-tags", "add", "--table", "test", "--region", "us-east-1", "team.*"]
    )
    runner.invoke(
        cli, ["denormalize-tags", "add", "--table", "test", "--region", "us-east-1", "team.*"]
    )
    result = runner.invoke(
        cli, ["denormalize-tags", "list", "--table", "test", "--region", "us-east-1"]
    )
    # Count occurrences - should appear exactly once
    assert result.output.strip().split("\n").count("team.*") == 1
