"""fts CLI commands."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group("fts")
def fts() -> None:
    """Manage FTS trigram field configuration."""
    pass


@fts.command("list")
@pass_context
def list_(ctx: CliContext) -> None:
    """List FTS trigram fields."""
    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)
    config = ConfigReader(table=ddb_table)
    click.echo("Always enabled: experiment_name, run_name, model_name")
    fields = config.get_fts_trigram_fields()
    if fields:
        click.echo("Additional fields:")
        for f in fields:
            click.echo(f"  {f}")
    else:
        click.echo("No additional fields configured.")


@fts.command("add")
@click.argument("field")
@pass_context
def add(ctx: CliContext, field: str) -> None:
    """Add a trigram field."""
    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)
    config = ConfigReader(table=ddb_table)
    fields = config.get_fts_trigram_fields()
    if field not in fields:
        fields.append(field)
    config.set_fts_trigram_fields(fields)
    click.echo(f"Added field: {field}")
