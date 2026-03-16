"""Tests for fts-trigrams CLI commands."""

from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@mock_aws
def test_fts_trigrams_list():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["fts-trigrams", "list", "--table", "test", "--region", "us-east-1"]
    )
    assert result.exit_code == 0
    assert "experiment_name" in result.output
    assert "No additional fields configured." in result.output


@mock_aws
def test_fts_trigrams_add():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["fts-trigrams", "add", "--table", "test", "--region", "us-east-1", "run_param_value"],
    )
    assert result.exit_code == 0
    assert "Added field: run_param_value" in result.output

    result = runner.invoke(
        cli, ["fts-trigrams", "list", "--table", "test", "--region", "us-east-1"]
    )
    assert result.exit_code == 0
    assert "run_param_value" in result.output
    assert "Additional fields:" in result.output


@mock_aws
def test_fts_trigrams_add_idempotent():
    """Adding the same field twice should not duplicate it."""
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    runner.invoke(
        cli,
        ["fts-trigrams", "add", "--table", "test", "--region", "us-east-1", "my_field"],
    )
    runner.invoke(
        cli,
        ["fts-trigrams", "add", "--table", "test", "--region", "us-east-1", "my_field"],
    )
    result = runner.invoke(
        cli, ["fts-trigrams", "list", "--table", "test", "--region", "us-east-1"]
    )
    assert result.exit_code == 0
    # Should appear only once
    assert result.output.count("my_field") == 1
