import boto3
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


class TestDestroy:
    @mock_aws
    def test_destroy_with_yes(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--name", "test-table", "--region", "us-east-1", "destroy", "--yes"]
        )
        assert result.exit_code == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" not in tables

    @mock_aws
    def test_destroy_prompts_confirmation(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--name", "test-table", "--region", "us-east-1", "destroy"], input="y\n"
        )
        assert result.exit_code == 0

    @mock_aws
    def test_destroy_aborts_on_no(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--name", "test-table", "--region", "us-east-1", "destroy"], input="n\n"
        )
        assert result.exit_code != 0

    @mock_aws
    def test_destroy_retain_keeps_table(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--name", "test-table", "--region", "us-east-1", "destroy", "--yes", "--retain"]
        )
        assert result.exit_code == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" in tables
