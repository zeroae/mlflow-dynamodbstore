"""ttl-policy CLI commands."""

import click

from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group("ttl-policy")
def ttl_policy() -> None:
    """Manage TTL retention policies."""
    pass


@ttl_policy.command("show")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
def show_policy(table: str, region: str) -> None:
    """Show current TTL policy."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)
    policy = config.get_ttl_policy()
    for key, value in sorted(policy.items()):
        status = "disabled" if value == 0 else f"{value} days"
        click.echo(f"{key}: {status}")


@ttl_policy.command("set")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.option("--soft-deleted-retention-days", type=int, default=None)
@click.option("--trace-retention-days", type=int, default=None)
@click.option("--metric-history-retention-days", type=int, default=None)
def set_policy(
    table: str,
    region: str,
    soft_deleted_retention_days: int | None,
    trace_retention_days: int | None,
    metric_history_retention_days: int | None,
) -> None:
    """Set TTL policy values."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
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
