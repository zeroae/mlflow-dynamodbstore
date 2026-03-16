"""cleanup-expired CLI command."""

from __future__ import annotations

import time

import boto3
import click

from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_META,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.command("cleanup-expired")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.option("--dry-run", is_flag=True, help="Report orphans without setting TTL")
def cleanup_expired(table: str, region: str, dry_run: bool) -> None:
    """Find and expire orphaned children of TTL-deleted experiments.

    Scans for experiment partitions whose META item has been removed by
    DynamoDB TTL, then sets ``ttl = now`` on all remaining children so
    DynamoDB will garbage-collect them.
    """
    ddb_table = DynamoDBTable(table_name=table, region=region)

    # Scan the main table for all unique EXP# partition keys.
    # We use a raw boto3 scan with a projection to minimise read cost.
    resource = boto3.resource("dynamodb", region_name=region)
    raw_table = resource.Table(table)

    exp_pks: set[str] = set()
    scan_kwargs: dict[str, object] = {
        "FilterExpression": "begins_with(PK, :prefix)",
        "ExpressionAttributeValues": {":prefix": PK_EXPERIMENT_PREFIX},
        "ProjectionExpression": "PK",
    }

    while True:
        response = raw_table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            exp_pks.add(item["PK"])
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    orphan_count = 0
    for pk in sorted(exp_pks):
        # Check whether the META item still exists
        meta = ddb_table.get_item(pk=pk, sk=SK_EXPERIMENT_META)
        if meta is not None:
            continue  # Experiment is alive — nothing to do

        # META is gone; all remaining items under this PK are orphans
        items = ddb_table.query(pk=pk)
        if not items:
            continue

        exp_id = pk[len(PK_EXPERIMENT_PREFIX) :]
        click.echo(f"Experiment {exp_id}: {len(items)} orphaned items")
        orphan_count += len(items)

        if not dry_run:
            now = int(time.time())
            for item in items:
                ddb_table.update_item(
                    pk=item["PK"],
                    sk=item["SK"],
                    updates={"ttl": now},
                )

    if dry_run:
        click.echo(f"Dry run: {orphan_count} orphaned items found")
    else:
        click.echo(f"Set TTL on {orphan_count} orphaned items")
