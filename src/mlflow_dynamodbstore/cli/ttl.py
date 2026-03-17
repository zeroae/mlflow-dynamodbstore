"""ttl CLI commands."""

from __future__ import annotations

import time

import boto3
import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_EXPERIMENT_META,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group("ttl")
def ttl() -> None:
    """Manage TTL retention policies."""
    pass


@ttl.command("show")
@pass_context
def show(ctx: CliContext) -> None:
    """Show current TTL policy."""
    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)
    config = ConfigReader(table=ddb_table)
    policy = config.get_ttl_policy()
    for key, value in sorted(policy.items()):
        status = "disabled" if value == 0 else f"{value} days"
        click.echo(f"{key}: {status}")


@ttl.command("set")
@click.option("--soft-deleted-retention-days", type=int, default=None)
@click.option("--trace-retention-days", type=int, default=None)
@click.option("--metric-history-retention-days", type=int, default=None)
@pass_context
def set_(
    ctx: CliContext,
    soft_deleted_retention_days: int | None,
    trace_retention_days: int | None,
    metric_history_retention_days: int | None,
) -> None:
    """Set TTL policy values."""
    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)
    config = ConfigReader(table=ddb_table)
    kwargs: dict[str, int] = {}
    if soft_deleted_retention_days is not None:
        kwargs["soft_deleted_retention_days"] = soft_deleted_retention_days
    if trace_retention_days is not None:
        kwargs["trace_retention_days"] = trace_retention_days
    if metric_history_retention_days is not None:
        kwargs["metric_history_retention_days"] = metric_history_retention_days
    if not kwargs:
        click.echo("No values provided. Use --help for options.")
        return
    config.set_ttl_policy(**kwargs)
    click.echo("TTL policy updated.")


@ttl.command("cleanup")
@click.option("--dry-run", is_flag=True, help="Report orphans without setting TTL")
@pass_context
def cleanup(ctx: CliContext, dry_run: bool) -> None:
    """Find and expire orphaned children of TTL-deleted experiments.

    Scans for experiment partitions whose META item has been removed by
    DynamoDB TTL, then sets ``ttl = now`` on all remaining children so
    DynamoDB will garbage-collect them.
    """
    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)

    # Scan the main table for all unique EXP# partition keys.
    # We use a raw boto3 scan with a projection to minimise read cost.
    resource = boto3.resource(
        "dynamodb",
        region_name=ctx.region,
        endpoint_url=ctx.endpoint_url,
    )
    raw_table = resource.Table(ctx.name)

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
