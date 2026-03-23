import boto3
import pytest
from moto import mock_aws


@pytest.fixture
def s3_bucket():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield "test-bucket"


def test_overflow_write_and_read(s3_bucket):
    from mlflow_dynamodbstore.dynamodb.overflow import overflow_read, overflow_write

    large_value = "x" * 400_000
    ref = overflow_write(s3_bucket, "EXP#123", "D#n#d", "schema", large_value, region="us-east-1")
    result = overflow_read(ref, region="us-east-1")
    assert result == large_value


def test_overflow_key_url_encodes():
    from mlflow_dynamodbstore.dynamodb.overflow import _overflow_key

    key = _overflow_key("EXP#123", "R#abc#INPUT#xyz", "schema")
    assert "EXP%23123" in key


def test_prepare_item_overflows_large_fields(s3_bucket):
    from mlflow_dynamodbstore.dynamodb.overflow import prepare_item_for_write

    item = {"PK": "EXP#1", "SK": "D#n#d", "schema": "x" * 400_000, "name": "small"}
    prepared = prepare_item_for_write(item, s3_bucket, region="us-east-1")
    assert isinstance(prepared["schema"], dict) and "_s3_ref" in prepared["schema"]
    assert prepared["name"] == "small"


def test_resolve_item_fetches_overflow(s3_bucket):
    from mlflow_dynamodbstore.dynamodb.overflow import (
        prepare_item_for_write,
        resolve_item_overflows,
    )

    item = {"PK": "EXP#1", "SK": "D#n#d", "schema": "x" * 400_000}
    prepared = prepare_item_for_write(item, s3_bucket, region="us-east-1")
    resolved = resolve_item_overflows(prepared, region="us-east-1")
    assert resolved["schema"] == "x" * 400_000
