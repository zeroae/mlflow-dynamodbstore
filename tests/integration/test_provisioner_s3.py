"""Integration tests for S3 bucket provisioning, store discovery, and overflow.

moto limitations
----------------
- ``AWS::Lambda::Function`` paired with a ``Custom::*`` CloudFormation resource
  causes moto to invoke the Lambda handler and then wait for a callback to a
  pre-signed S3 URL.  That callback never arrives in the test context so
  ``cfn.get_waiter("stack_create_complete")`` hangs indefinitely.

  Tests that call ``ensure_stack_exists`` (which builds a template that includes
  the ``BucketCleanupCustomResource``) are therefore **skipped** rather than
  marked as xfail, because xfail still executes the test body and would hang.

- The key tests (store autodiscovery and fallback) avoid the Custom Resource by
  setting up AWS resources directly via boto3 using a ``retain_bucket=True``
  template — which omits ``BucketCleanupCustomResource`` and completes
  synchronously under moto.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.provisioner import (
    destroy_stack,
    get_stack_outputs,
)

_MOTO_LAMBDA_CUSTOM_RESOURCE = pytest.mark.skip(
    reason=(
        "moto does not complete CloudFormation stacks that include a "
        "Lambda-backed Custom Resource — the waiter hangs indefinitely. "
        "Template structure is covered by unit tests in tests/unit/test_provisioner.py."
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_stack_with_bucket(table_name: str, bucket_name: str, region: str = "us-east-1") -> None:
    """Create a CloudFormation stack with a DynamoDB table and S3 bucket.

    Uses ``retain_bucket=True`` so ``BucketCleanupCustomResource`` is excluded
    from the template, allowing the CloudFormation waiter to complete
    synchronously under moto.
    """
    import json

    from mlflow_dynamodbstore.dynamodb.provisioner import (
        _build_template,
        _seed_initial_data,
    )

    template = _build_template(table_name, bucket_name=bucket_name, retain_bucket=True)
    cfn = boto3.client("cloudformation", region_name=region)
    cfn.create_stack(
        StackName=table_name,
        TemplateBody=json.dumps(template),
        Capabilities=["CAPABILITY_NAMED_IAM"],
    )
    cfn.get_waiter("stack_create_complete").wait(StackName=table_name)
    _seed_initial_data(table_name, region=region)


# ---------------------------------------------------------------------------
# Provisioner tests
# ---------------------------------------------------------------------------


@_MOTO_LAMBDA_CUSTOM_RESOURCE
@mock_aws
def test_deploy_creates_bucket_and_table():
    """ensure_stack_exists provisions both the DynamoDB table and the S3 bucket."""
    from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists

    ensure_stack_exists("test", region="us-east-1", bucket_name="test-artifacts")
    outputs = get_stack_outputs("test", region="us-east-1")
    assert outputs["ArtifactBucketName"] == "test-artifacts"
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    assert "test" in ddb.list_tables()["TableNames"]


@_MOTO_LAMBDA_CUSTOM_RESOURCE
@mock_aws
def test_deploy_default_bucket_name():
    """When no bucket_name is supplied the default derives from account ID (123456789012 in moto).

    moto account ID is always 123456789012.
    """
    from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists

    ensure_stack_exists("mystack", region="us-east-1")
    outputs = get_stack_outputs("mystack", region="us-east-1")
    # moto returns account ID 123456789012
    assert outputs["ArtifactBucketName"] == "mystack-artifacts-123456789012"


@mock_aws
def test_deploy_with_retain_bucket_creates_stack():
    """A retain_bucket=True stack completes synchronously under moto.

    Verifies that the DynamoDB table and S3 bucket are provisioned correctly
    when the Custom Resource is absent — the scenario produced by
    ``_setup_stack_with_bucket`` which is used by the store tests below.
    """
    _setup_stack_with_bucket("test", "test-artifacts")
    outputs = get_stack_outputs("test", region="us-east-1")
    assert outputs["ArtifactBucketName"] == "test-artifacts"
    ddb = boto3.client("dynamodb", region_name="us-east-1")
    assert "test" in ddb.list_tables()["TableNames"]


@_MOTO_LAMBDA_CUSTOM_RESOURCE
@mock_aws
def test_destroy_with_retain():
    """destroy_stack(retain=True) keeps the S3 bucket after stack deletion."""
    from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists

    ensure_stack_exists("test", region="us-east-1", bucket_name="test-artifacts")
    destroy_stack("test", region="us-east-1", retain=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert "test-artifacts" in buckets


@mock_aws
def test_destroy_with_retain_via_helper():
    """destroy_stack(retain=True) keeps the S3 bucket (using the moto-safe helper stack)."""
    _setup_stack_with_bucket("test", "test-artifacts")
    destroy_stack("test", region="us-east-1", retain=True)
    s3 = boto3.client("s3", region_name="us-east-1")
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    assert "test-artifacts" in buckets


# ---------------------------------------------------------------------------
# Store autodiscovery tests  (KEY TESTS)
# ---------------------------------------------------------------------------


@mock_aws
def test_store_autodiscover_bucket_from_uri():
    """When bucket= is in the URI the store uses it directly without stack lookup."""
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    # Provision infrastructure via the moto-safe helper.
    _setup_stack_with_bucket("test", "test-artifacts")

    store = DynamoDBTrackingStore("dynamodb://us-east-1/test?bucket=test-artifacts")
    assert store._overflow_bucket == "test-artifacts"
    assert store._artifact_uri == "s3://test-artifacts"


@mock_aws
def test_store_autodiscover_bucket_from_stack():
    """When no bucket= is in the URI the store discovers the bucket from stack outputs."""
    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    # Provision infrastructure with a known bucket in CFn outputs.
    _setup_stack_with_bucket("test", "test-artifacts")

    # deploy=false: skip ensure_stack_exists (avoids the Custom Resource), but
    # the store still reads stack outputs to discover the bucket name.
    store = DynamoDBTrackingStore("dynamodb://us-east-1/test?deploy=false")
    assert store._overflow_bucket == "test-artifacts"
    assert store._artifact_uri == "s3://test-artifacts"


@mock_aws
def test_store_fallback_without_stack():
    """With deploy=false and no pre-existing stack the store falls back to ./mlartifacts.

    ConfigReader.reconcile() would write to DynamoDB (which doesn't exist when
    deploy=false with no provisioned table), so reconcile() is patched to a no-op
    to allow inspection of the bucket-discovery attributes that are set earlier
    in __init__.
    """
    from unittest.mock import patch

    from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore

    with patch("mlflow_dynamodbstore.dynamodb.config.ConfigReader.reconcile", return_value=None):
        store = DynamoDBTrackingStore("dynamodb://us-east-1/test?deploy=false")

    assert store._artifact_uri == "./mlartifacts"
    assert store._overflow_bucket is None
