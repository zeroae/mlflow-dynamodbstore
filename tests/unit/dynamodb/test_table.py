import boto3
import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


def _create_test_table(region: str = "us-east-1", table_name: str = "test"):
    """Create a minimal DynamoDB table for testing."""
    ddb = boto3.client("dynamodb", region_name=region)
    ddb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "lsi1sk", "AttributeType": "S"},
            {"AttributeName": "gsi1pk", "AttributeType": "S"},
            {"AttributeName": "gsi1sk", "AttributeType": "S"},
        ],
        LocalSecondaryIndexes=[
            {
                "IndexName": "lsi1",
                "KeySchema": [
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": "lsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "gsi1",
                "KeySchema": [
                    {"AttributeName": "gsi1pk", "KeyType": "HASH"},
                    {"AttributeName": "gsi1sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        BillingMode="PAY_PER_REQUEST",
    )


class TestDynamoDBTable:
    @mock_aws
    def test_put_and_get_item(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#123", "SK": "E#META", "name": "test"})
        item = table.get_item("EXP#123", "E#META")
        assert item is not None
        assert item["name"] == "test"

    @mock_aws
    def test_get_item_not_found(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        assert table.get_item("EXP#999", "E#META") is None

    @mock_aws
    def test_query_by_pk_and_sk_prefix(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "R#aaa", "status": "RUNNING"})
        table.put_item({"PK": "EXP#1", "SK": "R#bbb", "status": "FINISHED"})
        table.put_item({"PK": "EXP#1", "SK": "E#META", "name": "exp"})
        items = table.query(pk="EXP#1", sk_prefix="R#")
        assert len(items) == 2

    @mock_aws
    def test_query_with_sk_between(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "R#aaa"})
        table.put_item({"PK": "EXP#1", "SK": "R#bbb"})
        table.put_item({"PK": "EXP#1", "SK": "R#ccc"})
        items = table.query(pk="EXP#1", sk_gte="R#aaa", sk_lte="R#bbb")
        assert len(items) == 2

    @mock_aws
    def test_query_with_limit(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        for i in range(10):
            table.put_item({"PK": "EXP#1", "SK": f"R#{i:04d}"})
        items = table.query(pk="EXP#1", sk_prefix="R#", limit=3)
        assert len(items) == 3

    @mock_aws
    def test_query_reverse(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "R#aaa"})
        table.put_item({"PK": "EXP#1", "SK": "R#bbb"})
        items = table.query(pk="EXP#1", sk_prefix="R#", scan_forward=False)
        assert items[0]["SK"] == "R#bbb"

    @mock_aws
    def test_query_on_gsi(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "R#run1", "gsi1pk": "RUN#run1", "gsi1sk": "EXP#1"})
        items = table.query(pk="RUN#run1", index_name="gsi1")
        assert len(items) == 1
        assert items[0]["gsi1sk"] == "EXP#1"

    @mock_aws
    def test_delete_item(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "E#META"})
        table.delete_item("EXP#1", "E#META")
        assert table.get_item("EXP#1", "E#META") is None

    @mock_aws
    def test_update_item(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "E#META", "name": "old"})
        table.update_item("EXP#1", "E#META", updates={"name": "new", "status": "active"})
        item = table.get_item("EXP#1", "E#META")
        assert item["name"] == "new"
        assert item["status"] == "active"

    @mock_aws
    def test_update_item_remove_attributes(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "E#META", "name": "test", "ttl": 12345})
        table.update_item("EXP#1", "E#META", removes=["ttl"])
        item = table.get_item("EXP#1", "E#META")
        assert "ttl" not in item

    @mock_aws
    def test_batch_write_item(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        items = [{"PK": "EXP#1", "SK": f"R#run{i}"} for i in range(30)]
        table.batch_write(items)  # should chunk into 25 + 5
        result = table.query(pk="EXP#1", sk_prefix="R#")
        assert len(result) == 30

    @mock_aws
    def test_conditional_put_item(self):
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        table.put_item({"PK": "EXP#1", "SK": "E#META"}, condition="attribute_not_exists(PK)")
        with pytest.raises(Exception):  # ConditionalCheckFailedException
            table.put_item({"PK": "EXP#1", "SK": "E#META"}, condition="attribute_not_exists(PK)")
