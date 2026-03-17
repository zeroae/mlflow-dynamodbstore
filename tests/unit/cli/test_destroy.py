import boto3
import click
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import CliContext
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.group()
@click.pass_context
def _test_cli(ctx):
    ctx.obj = CliContext(name="test-table", region="us-east-1", endpoint_url=None)


from mlflow_dynamodbstore.cli.destroy import destroy  # noqa: E402

_test_cli.add_command(destroy)


class TestDestroy:
    @mock_aws
    def test_destroy_with_yes(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy", "--yes"])
        assert result.exit_code == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" not in tables

    @mock_aws
    def test_destroy_prompts_confirmation(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy"], input="y\n")
        assert result.exit_code == 0

    @mock_aws
    def test_destroy_aborts_on_no(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy"], input="n\n")
        assert result.exit_code != 0

    @mock_aws
    def test_destroy_retain_keeps_table(self):
        ensure_stack_exists("test-table", "us-east-1")
        runner = CliRunner()
        result = runner.invoke(_test_cli, ["destroy", "--yes", "--retain"])
        assert result.exit_code == 0
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" in tables
