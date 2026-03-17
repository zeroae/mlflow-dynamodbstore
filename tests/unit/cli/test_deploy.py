import boto3
import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test-table", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.deploy import deploy  # noqa: E402

_test_cli.add_command(deploy)


class TestDeploy:
    @mock_aws
    def test_deploy_creates_stack(self):
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["deploy"])
        assert result.exit_code == 0
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        assert any(s["StackName"] == "test-table" for s in stacks)

    @mock_aws
    def test_deploy_idempotent(self):
        runner = CliRunner()
        runner.invoke(_test_cli, ["deploy"])
        result = runner.invoke(_test_cli, ["deploy"])
        assert result.exit_code == 0

    @mock_aws
    def test_deploy_seeds_default_data(self):
        runner = CliRunner()
        runner.invoke(_test_cli, ["deploy"])
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "WORKSPACE#default"}, "SK": {"S": "META"}},
        )
        assert "Item" in result
