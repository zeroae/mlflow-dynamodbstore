"""Tests for ttl-policy CLI commands."""

from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@mock_aws
def test_ttl_policy_show():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(cli, ["ttl-policy", "show", "--table", "test", "--region", "us-east-1"])
    assert result.exit_code == 0
    assert "soft_deleted_retention_days" in result.output
    assert "90 days" in result.output


@mock_aws
def test_ttl_policy_set():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ttl-policy",
            "set",
            "--table",
            "test",
            "--region",
            "us-east-1",
            "--soft-deleted-retention-days",
            "60",
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(cli, ["ttl-policy", "show", "--table", "test", "--region", "us-east-1"])
    assert "60 days" in result.output


@mock_aws
def test_ttl_policy_set_no_values():
    ensure_stack_exists(table_name="test", region="us-east-1")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ttl-policy", "set", "--table", "test", "--region", "us-east-1"],
    )
    assert result.exit_code == 0
    assert "No values provided" in result.output
