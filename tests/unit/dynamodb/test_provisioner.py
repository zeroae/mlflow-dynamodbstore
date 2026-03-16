import boto3
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists, get_stack_name


class TestProvisioner:
    @mock_aws
    def test_stack_name(self):
        assert get_stack_name("my-table") == "mlflow-dynamodbstore-my-table"

    @mock_aws
    def test_creates_stack_if_not_exists(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        stacks = cfn.list_stacks(StackStatusFilter=["CREATE_COMPLETE"])["StackSummaries"]
        stack_names = [s["StackName"] for s in stacks]
        assert "mlflow-dynamodbstore-test-table" in stack_names

    @mock_aws
    def test_table_created_by_stack(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" in tables

    @mock_aws
    def test_table_has_5_lsis(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        desc = ddb.describe_table(TableName="test-table")["Table"]
        lsis = desc.get("LocalSecondaryIndexes", [])
        assert len(lsis) == 5

    @mock_aws
    def test_table_has_5_gsis(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        desc = ddb.describe_table(TableName="test-table")["Table"]
        gsis = desc.get("GlobalSecondaryIndexes", [])
        assert len(gsis) == 5

    @mock_aws
    def test_idempotent_if_stack_exists(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ensure_stack_exists(table_name="test-table", region="us-east-1")  # no error

    @mock_aws
    def test_creates_default_workspace(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "WORKSPACE#default"}, "SK": {"S": "META"}},
        )
        assert "Item" in result

    @mock_aws
    def test_creates_default_experiment(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        result = ddb.get_item(
            TableName="test-table",
            Key={"PK": {"S": "EXP#0"}, "SK": {"S": "E#META"}},
        )
        assert "Item" in result
        assert result["Item"]["name"]["S"] == "Default"

    @mock_aws
    def test_delete_stack_removes_table(self):
        ensure_stack_exists(table_name="test-table", region="us-east-1")
        cfn = boto3.client("cloudformation", region_name="us-east-1")
        cfn.delete_stack(StackName="mlflow-dynamodbstore-test-table")
        ddb = boto3.client("dynamodb", region_name="us-east-1")
        tables = ddb.list_tables()["TableNames"]
        assert "test-table" not in tables
