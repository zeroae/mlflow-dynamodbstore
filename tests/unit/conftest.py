import json
from typing import Any

import boto3
import pytest
from moto import mock_aws

_MOTO_BUCKET = "test-table-artifacts"


def _setup_stack_moto(
    table_name: str,
    region: str = "us-east-1",
) -> None:
    """Create a CloudFormation stack under moto without S3/Lambda resources.

    Uses bucket_name=None to skip all S3 resources which hang under moto.
    Creates the S3 bucket directly via boto3 so the store can use it for overflow.
    """
    from mlflow_dynamodbstore.dynamodb.provisioner import _build_template, _seed_initial_data

    template = _build_template(table_name)
    kwargs: dict[str, Any] = {"region_name": region}
    cfn = boto3.client("cloudformation", **kwargs)
    cfn.create_stack(
        StackName=table_name,
        TemplateBody=json.dumps(template),
    )
    cfn.get_waiter("stack_create_complete").wait(StackName=table_name)
    _seed_initial_data(table_name, region=region)

    # Create S3 bucket directly (bypassing CloudFormation Lambda issue)
    s3 = boto3.client("s3", **kwargs)
    try:
        s3.create_bucket(Bucket=_MOTO_BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass


@pytest.fixture
def mock_dynamodb():
    with mock_aws():
        yield


@pytest.fixture
def tracking_store(mock_dynamodb):
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    _setup_stack_moto("test-table")
    store = DynamoDBTrackingStore(
        store_uri=f"dynamodb://us-east-1/test-table?deploy=false&bucket={_MOTO_BUCKET}",
        artifact_uri="/tmp/artifacts",
    )
    return store


@pytest.fixture
def registry_store(mock_dynamodb):
    from mlflow_dynamodbstore.registry_store import DynamoDBRegistryStore

    _setup_stack_moto("test-table")
    store = DynamoDBRegistryStore(
        store_uri="dynamodb://us-east-1/test-table?deploy=false",
    )
    return store


@pytest.fixture
def workspace_store(mock_dynamodb):
    from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore

    _setup_stack_moto("test-table")
    store = DynamoDBWorkspaceStore(
        store_uri="dynamodb://us-east-1/test-table?deploy=false",
    )
    return store
