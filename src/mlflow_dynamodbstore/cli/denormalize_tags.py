"""denormalize-tags CLI commands."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group("denormalize-tags")
def denormalize_tags() -> None:
    """Manage tag denormalization patterns."""
    pass


@denormalize_tags.command("list")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.option("--experiment-id", default=None, help="Show effective patterns for an experiment")
def list_patterns(table: str, region: str, experiment_id: str | None) -> None:
    """List denormalization patterns."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)
    if experiment_id:
        patterns = config.get_effective_denormalize_patterns(experiment_id)
    else:
        patterns = config.get_denormalize_patterns()
    for p in patterns:
        click.echo(p)


@denormalize_tags.command("add")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.option("--experiment-id", default=None, help="Add per-experiment pattern")
@click.argument("pattern")
def add_pattern(table: str, region: str, experiment_id: str | None, pattern: str) -> None:
    """Add a denormalization pattern."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)
    if experiment_id:
        patterns = config.get_experiment_denormalize_patterns(experiment_id)
        if pattern not in patterns:
            patterns.append(pattern)
        config.set_experiment_denormalize_patterns(experiment_id, patterns)
    else:
        patterns = config.get_denormalize_patterns()
        if pattern not in patterns:
            patterns.append(pattern)
        config.set_denormalize_patterns(patterns)
    click.echo(f"Added pattern: {pattern}")


@denormalize_tags.command("remove")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.option("--experiment-id", default=None, help="Remove per-experiment pattern")
@click.argument("pattern")
def remove_pattern(table: str, region: str, experiment_id: str | None, pattern: str) -> None:
    """Remove a denormalization pattern."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)
    if experiment_id:
        patterns = config.get_experiment_denormalize_patterns(experiment_id)
        if pattern in patterns:
            patterns.remove(pattern)
        config.set_experiment_denormalize_patterns(experiment_id, patterns)
    else:
        patterns = config.get_denormalize_patterns()
        if pattern in patterns:
            patterns.remove(pattern)
        config.set_denormalize_patterns(patterns)
    click.echo(f"Removed pattern: {pattern}")


@denormalize_tags.command("backfill")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.option("--experiment-id", default=None, help="Backfill only this experiment")
def backfill(table: str, region: str, experiment_id: str | None) -> None:
    """Backfill denormalized tags onto META items.

    Scans tag items, matches against configured patterns, and updates META items
    with denormalized tag attributes.
    """
    from mlflow_dynamodbstore.dynamodb.schema import PK_CONFIG  # noqa: F401

    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)

    # Query all tag items (SK begins with "TAG#")
    # We need to scan since tags are spread across experiments
    import boto3

    resource = boto3.resource("dynamodb", region_name=region)
    raw_table = resource.Table(table)

    scan_kwargs: dict[str, object] = {
        "FilterExpression": "begins_with(SK, :tag_prefix)",
        "ExpressionAttributeValues": {":tag_prefix": "TAG#"},
    }

    updated_count = 0
    scanned_count = 0

    while True:
        response = raw_table.scan(**scan_kwargs)
        items = response.get("Items", [])

        for item in items:
            scanned_count += 1
            pk = item["PK"]
            sk = item["SK"]
            # Extract tag key from SK: TAG#<key>
            tag_key = sk[4:]  # strip "TAG#"
            tag_value = item.get("value", "")

            # Determine experiment_id from PK for pattern matching
            # PK format varies; use None for global patterns if experiment_id not provided
            exp_id = experiment_id
            if config.should_denormalize(exp_id, tag_key):
                # Update the META item for this entity
                meta_sk = "META"
                attr_name = f"tag.{tag_key}"
                ddb_table.update_item(pk=pk, sk=meta_sk, updates={attr_name: tag_value})
                updated_count += 1

        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    click.echo(
        f"Backfill complete: scanned {scanned_count} tags, updated {updated_count} META items."
    )
