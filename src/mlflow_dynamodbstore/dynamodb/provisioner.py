"""CloudFormation auto-provisioner for the mlflow-dynamodbstore DynamoDB table."""

from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError


def _build_template(table_name: str, retain_table: bool = False) -> dict[str, Any]:
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

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"MLflow DynamoDB store table: {table_name}",
        "Resources": {
            "MlflowTable": mlflow_table_resource,
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
    stack_name = table_name

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
        cfn.get_waiter("stack_create_complete").wait(StackName=stack_name)

    _seed_initial_data(table_name, region, endpoint_url)


def destroy_stack(
    table_name: str,
    region: str = "us-east-1",
    endpoint_url: str | None = None,
    retain: bool = False,
) -> None:
    """Delete the CloudFormation stack for a given table.

    Args:
        table_name: The DynamoDB table name (also the stack name).
        region: AWS region.
        endpoint_url: Optional custom endpoint URL.
        retain: If True, retain the DynamoDB table resource when deleting the stack.

    Raises:
        ClientError: If the stack does not exist or deletion fails.
    """
    kwargs: dict[str, Any] = {"region_name": region}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    cfn = boto3.client("cloudformation", **kwargs)

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
