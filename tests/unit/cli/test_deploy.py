import boto3
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli


class TestDeploy:
    @mock_aws
    def test_deploy_creates_stack(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        assert result.exit_code == 0
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        assert any(s["StackName"] == "test-table" for s in stacks)

    @mock_aws
    def test_deploy_idempotent(self):
        runner = CliRunner()
        runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        result = runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        assert result.exit_code == 0

    @mock_aws
    def test_deploy_seeds_default_data(self):
        runner = CliRunner()
        runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "WORKSPACE#default"}, "SK": {"S": "META"}},
        )
        assert "Item" in result
