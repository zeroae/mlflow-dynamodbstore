"""CloudFormation auto-provisioner for the mlflow-dynamodbstore DynamoDB table."""

from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

_STACK_PREFIX = "mlflow-dynamodbstore-"


def get_stack_name(table_name: str) -> str:
    """Return the CloudFormation stack name for a given table."""
    return f"{_STACK_PREFIX}{table_name}"


def _build_template(table_name: str) -> dict[str, Any]:
    """Build the CloudFormation template as a Python dict."""
    # All attribute definitions needed for keys
    attr_defs = [
        {"AttributeName": "PK", "AttributeType": "S"},
        {"AttributeName": "SK", "AttributeType": "S"},
    ]
    for i in range(1, 6):
        attr_defs.append({"AttributeName": f"lsi{i}sk", "AttributeType": "S"})
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

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"MLflow DynamoDB store table: {table_name}",
        "Resources": {
            "MlflowTable": {
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
            },
        },
    }


def _seed_initial_data(
    table_name: str,
    region: str,
    endpoint_url: str | None = None,
) -> None:
    """Seed default workspace, experiment, and config items."""
    kwargs: dict[str, Any] = {"region_name": region}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    ddb = boto3.resource("dynamodb", **kwargs)
    table = ddb.Table(table_name)

    now_ms = int(time.time() * 1000)

    seed_items: list[dict[str, Any]] = [
        # Default workspace
        {
            "PK": "WORKSPACE#default",
            "SK": "META",
            "name": "default",
            "description": "Default workspace",
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
    region: str = "us-east-1",
    endpoint_url: str | None = None,
) -> None:
    """Ensure the CloudFormation stack and DynamoDB table exist.

    Creates the stack if it does not exist, then seeds initial data
    (default workspace, default experiment, config items).

    Idempotent: safe to call multiple times.
    """
    stack_name = get_stack_name(table_name)

    kwargs: dict[str, Any] = {"region_name": region}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    cfn = boto3.client("cloudformation", **kwargs)

    if not _stack_exists(cfn, stack_name):
        template = _build_template(table_name)
        cfn.create_stack(
            StackName=stack_name,
            TemplateBody=json.dumps(template),
        )
        # In moto, stack creation is synchronous.
        # In real AWS, wait for completion:
        # cfn.get_waiter('stack_create_complete').wait(StackName=stack_name)

    _seed_initial_data(table_name, region, endpoint_url)
