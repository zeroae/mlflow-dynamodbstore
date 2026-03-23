"""CloudFormation auto-provisioner for the mlflow-dynamodbstore DynamoDB table."""

from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

_BUCKET_CLEANUP_CODE = """\
import json
import urllib.request
import boto3

def handler(event, context):
    response_url = event["ResponseURL"]
    try:
        if event["RequestType"] == "Delete":
            bucket = event["ResourceProperties"]["BucketName"]
            s3 = boto3.resource("s3")
            b = s3.Bucket(bucket)
            b.object_versions.all().delete()
            b.objects.all().delete()
        _send(response_url, event, "SUCCESS")
    except Exception as e:
        _send(response_url, event, "FAILED", str(e))

def _send(url, event, status, reason=""):
    body = json.dumps({
        "Status": status,
        "Reason": reason,
        "PhysicalResourceId": event.get("PhysicalResourceId", event["LogicalResourceId"]),
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ""})
    req.add_header("Content-Length", str(len(body)))
    urllib.request.urlopen(req)
"""


def _add_s3_resources(
    template: dict[str, Any],
    bucket_name: str,
    retain_bucket: bool,
    iam_format: str,
    permission_boundary: str | None,
) -> None:
    """Add S3 bucket, Lambda cleanup function, IAM role, and custom resource."""
    resources = template["Resources"]

    # S3 Bucket
    bucket_resource: dict[str, Any] = {
        "Type": "AWS::S3::Bucket",
        "Properties": {
            "BucketName": bucket_name,
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [
                    {
                        "ServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256",
                        },
                    },
                ],
            },
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
        },
    }
    if retain_bucket:
        bucket_resource["DeletionPolicy"] = "Retain"
    else:
        bucket_resource["DeletionPolicy"] = "Delete"
    resources["ArtifactBucket"] = bucket_resource

    # IAM Role for Lambda cleanup
    role_name_sub = iam_format.format("${AWS::StackName}-BucketCleanup")
    policy_name_sub = iam_format.format("${AWS::StackName}-BucketCleanupPolicy")

    role_props: dict[str, Any] = {
        "RoleName": {"Fn::Sub": role_name_sub},
        "AssumeRolePolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                },
            ],
        },
        "Policies": [
            {
                "PolicyName": {"Fn::Sub": policy_name_sub},
                "PolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "s3:ListBucket",
                                "s3:DeleteObject",
                                "s3:ListBucketVersions",
                                "s3:DeleteObjectVersion",
                            ],
                            "Resource": [
                                f"arn:aws:s3:::{bucket_name}",
                                f"arn:aws:s3:::{bucket_name}/*",
                            ],
                        },
                        {
                            "Effect": "Allow",
                            "Action": [
                                "logs:CreateLogGroup",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                            ],
                            "Resource": {
                                "Fn::Sub": (
                                    "arn:aws:logs:${AWS::Region}:${AWS::AccountId}"
                                    ":log-group:/aws/lambda/*"
                                ),
                            },
                        },
                    ],
                },
            },
        ],
    }

    if permission_boundary:
        role_props["PermissionsBoundary"] = {
            "Fn::Sub": f"arn:aws:iam::${{AWS::AccountId}}:policy/{permission_boundary}",
        }

    resources["BucketCleanupRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": role_props,
    }

    # Lambda Function
    resources["BucketCleanupFunction"] = {
        "Type": "AWS::Lambda::Function",
        "Properties": {
            "FunctionName": {"Fn::Sub": "${AWS::StackName}-BucketCleanup"},
            "Runtime": "python3.12",
            "Handler": "index.handler",
            "Timeout": 300,
            "Role": {"Fn::GetAtt": ["BucketCleanupRole", "Arn"]},
            "Code": {"ZipFile": _BUCKET_CLEANUP_CODE},
        },
    }

    # Custom Resource (only when not retaining the bucket)
    if not retain_bucket:
        resources["BucketCleanupCustomResource"] = {
            "Type": "Custom::BucketCleanup",
            "DependsOn": ["ArtifactBucket"],
            "Properties": {
                "ServiceToken": {"Fn::GetAtt": ["BucketCleanupFunction", "Arn"]},
                "BucketName": bucket_name,
            },
        }

    # Outputs
    template["Outputs"] = {
        "ArtifactBucketName": {"Value": bucket_name},
        "ArtifactBucketArn": {"Value": {"Fn::GetAtt": ["ArtifactBucket", "Arn"]}},
    }


def _build_template(
    table_name: str,
    retain_table: bool = False,
    retain_bucket: bool = False,
    bucket_name: str | None = None,
    iam_format: str = "{}",
    permission_boundary: str | None = None,
) -> dict[str, Any]:
    """Build the CloudFormation template as a Python dict."""
    # All attribute definitions needed for keys
    attr_defs = [
        {"AttributeName": "PK", "AttributeType": "S"},
        {"AttributeName": "SK", "AttributeType": "S"},
    ]
    # lsi2sk stores timestamps (numeric); all others are string sort keys
    _lsi_types = {"lsi2sk": "N"}
    for i in range(1, 6):
        attr_name = f"lsi{i}sk"
        attr_defs.append(
            {"AttributeName": attr_name, "AttributeType": _lsi_types.get(attr_name, "S")}
        )
    for i in range(1, 6):
        attr_defs.append({"AttributeName": f"gsi{i}pk", "AttributeType": "S"})
        attr_defs.append({"AttributeName": f"gsi{i}sk", "AttributeType": "S"})

    # LSI definitions
    lsis = []
    for i in range(1, 6):
        lsis.append(
            {
                "IndexName": f"lsi{i}",
                "KeySchema": [
                    {"AttributeName": "PK", "KeyType": "HASH"},
                    {"AttributeName": f"lsi{i}sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        )

    # GSI definitions
    gsis = []
    for i in range(1, 6):
        gsis.append(
            {
                "IndexName": f"gsi{i}",
                "KeySchema": [
                    {"AttributeName": f"gsi{i}pk", "KeyType": "HASH"},
                    {"AttributeName": f"gsi{i}sk", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        )

    mlflow_table_resource: dict[str, Any] = {
        "Type": "AWS::DynamoDB::Table",
        "Properties": {
            "TableName": table_name,
            "AttributeDefinitions": attr_defs,
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
            "LocalSecondaryIndexes": lsis,
            "GlobalSecondaryIndexes": gsis,
            "PointInTimeRecoverySpecification": {
                "PointInTimeRecoveryEnabled": True,
            },
            "TimeToLiveSpecification": {
                "AttributeName": "ttl",
                "Enabled": True,
            },
        },
    }
    if retain_table:
        mlflow_table_resource["DeletionPolicy"] = "Retain"

    template: dict[str, Any] = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"MLflow DynamoDB store table: {table_name}",
        "Resources": {
            "MlflowTable": mlflow_table_resource,
        },
    }

    if bucket_name is not None:
        _add_s3_resources(template, bucket_name, retain_bucket, iam_format, permission_boundary)

    return template


def _boto_kwargs(
    region: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    """Build kwargs for boto3 client/resource calls."""
    kwargs: dict[str, Any] = {}
    if region:
        kwargs["region_name"] = region
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return kwargs


def _seed_initial_data(
    table_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> None:
    """Seed default workspace, experiment, and config items."""
    kwargs = _boto_kwargs(region, endpoint_url)

    ddb = boto3.resource("dynamodb", **kwargs)
    table = ddb.Table(table_name)

    now_ms = int(time.time() * 1000)

    seed_items: list[dict[str, Any]] = [
        # Default workspace
        {
            "PK": "WORKSPACE#default",
            "SK": "META",
            "name": "default",
            "description": "Default workspace for legacy resources",
            "gsi2pk": "WORKSPACES",
            "gsi2sk": "default",
        },
        # Default experiment
        {
            "PK": "EXP#0",
            "SK": "E#META",
            "name": "Default",
            "lifecycle_stage": "active",
            "artifact_location": "",
            "creation_time": now_ms,
            "last_update_time": now_ms,
            "workspace": "default",
            "gsi2pk": "EXPERIMENTS#default#active",
            "gsi2sk": "0",
            "gsi3pk": "EXP_NAME#default#Default",
            "gsi3sk": "0",
            "gsi5pk": "EXP_NAMES#default",
            "gsi5sk": "Default#0",
        },
        # Config: denormalize tags
        {
            "PK": "CONFIG",
            "SK": "DENORMALIZE_TAGS",
            "patterns": ["mlflow.*"],
        },
        # Config: TTL policy
        {
            "PK": "CONFIG",
            "SK": "TTL_POLICY",
            "soft_deleted_retention_days": 90,
            "trace_retention_days": 30,
            "metric_history_retention_days": 365,
        },
        # Config: FTS trigram fields
        {
            "PK": "CONFIG",
            "SK": "FTS_TRIGRAM_FIELDS",
            "fields": [],
        },
    ]

    for item in seed_items:
        try:
            table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(PK)",
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                # Item already exists, skip
                pass
            else:
                raise


def _stack_exists(cfn: Any, stack_name: str) -> bool:
    """Check if a CloudFormation stack already exists and is in a good state."""
    try:
        response = cfn.describe_stacks(StackName=stack_name)
    except ClientError as exc:
        msg = exc.response["Error"]["Message"]
        if "does not exist" in msg:
            return False
        raise
    stacks = response.get("Stacks", [])
    if not stacks:
        return False
    status = stacks[0]["StackStatus"]
    return status in ("CREATE_COMPLETE", "UPDATE_COMPLETE")


def ensure_stack_exists(
    table_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> None:
    """Ensure the CloudFormation stack and DynamoDB table exist.

    Creates the stack if it does not exist, then seeds initial data
    (default workspace, default experiment, config items).

    Idempotent: safe to call multiple times.
    """
    stack_name = table_name

    cfn = boto3.client("cloudformation", **_boto_kwargs(region, endpoint_url))

    if not _stack_exists(cfn, stack_name):
        template = _build_template(table_name)
        cfn.create_stack(
            StackName=stack_name,
            TemplateBody=json.dumps(template),
        )
        cfn.get_waiter("stack_create_complete").wait(StackName=stack_name)

    _seed_initial_data(table_name, region, endpoint_url)


def destroy_stack(
    table_name: str,
    region: str | None = None,
    endpoint_url: str | None = None,
    retain: bool = False,
) -> None:
    """Delete the CloudFormation stack for a given table.

    Args:
        table_name: The DynamoDB table name (also the stack name).
        region: AWS region (omit to use boto3 default chain).
        endpoint_url: Optional custom endpoint URL.
        retain: If True, retain the DynamoDB table resource when deleting the stack.

    Raises:
        ClientError: If the stack does not exist or deletion fails.
    """
    cfn = boto3.client("cloudformation", **_boto_kwargs(region, endpoint_url))

    # Verify the stack exists first (raises if not)
    cfn.describe_stacks(StackName=table_name)

    if retain:
        # Update the stack to set DeletionPolicy=Retain on the table, then delete
        retain_template = _build_template(table_name, retain_table=True)
        cfn.update_stack(
            StackName=table_name,
            TemplateBody=json.dumps(retain_template),
        )
        cfn.get_waiter("stack_update_complete").wait(StackName=table_name)

    cfn.delete_stack(StackName=table_name)
    cfn.get_waiter("stack_delete_complete").wait(StackName=table_name)
