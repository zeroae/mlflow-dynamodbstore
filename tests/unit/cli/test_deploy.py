from unittest.mock import patch

import boto3
from click.testing import CliRunner
from moto import mock_aws

from mlflow_dynamodbstore.cli import cli

_NO_BUCKET = patch("mlflow_dynamodbstore.cli.deploy._resolve_bucket_name", return_value=None)


class TestDeploy:
    @mock_aws
    def test_deploy_creates_stack(self):
        runner = CliRunner()
        with _NO_BUCKET:
            result = runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        assert result.exit_code == 0
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        assert any(s["StackName"] == "test-table" for s in stacks)

    @mock_aws
    def test_deploy_idempotent(self):
        runner = CliRunner()
        with _NO_BUCKET:
            runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
            result = runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        assert result.exit_code == 0

    @mock_aws
    def test_deploy_seeds_default_data(self):
        runner = CliRunner()
        with _NO_BUCKET:
            runner.invoke(cli, ["--name", "test-table", "--region", "us-east-1", "deploy"])
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "WORKSPACE#default"}, "SK": {"S": "META"}},
        )
        assert "Item" in result

    @mock_aws
    def test_deploy_passes_iam_params(self):
        """CLI --iam-format and --permission-boundary are forwarded to ensure_stack_exists."""
        runner = CliRunner()
        with patch("mlflow_dynamodbstore.cli.deploy.ensure_stack_exists") as mock_ensure:
            result = runner.invoke(
                cli,
                [
                    "--name",
                    "test-table",
                    "--region",
                    "us-east-1",
                    "--iam-format",
                    "PowerUserPB-{}",
                    "--permission-boundary",
                    "arn:aws:iam::aws:policy/PowerUserAccess",
                    "deploy",
                ],
            )
            assert result.exit_code == 0
            _, kwargs = mock_ensure.call_args
            assert kwargs["iam_format"] == "PowerUserPB-{}"
            assert kwargs["permission_boundary"] == "arn:aws:iam::aws:policy/PowerUserAccess"
            assert kwargs["bucket_name"] is not None  # auto-resolved from STS

    @mock_aws
    def test_deploy_resolves_bucket_from_sts(self):
        """When --bucket is not provided, bucket name is auto-resolved from STS account ID."""
        runner = CliRunner()
        with patch("mlflow_dynamodbstore.cli.deploy.ensure_stack_exists") as mock_ensure:
            result = runner.invoke(
                cli,
                ["--name", "test-table", "--region", "us-east-1", "deploy"],
            )
            assert result.exit_code == 0
            _, kwargs = mock_ensure.call_args
            assert kwargs["bucket_name"].startswith("test-table-artifacts-")
