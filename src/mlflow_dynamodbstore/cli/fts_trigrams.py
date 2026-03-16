"""fts-trigrams CLI commands."""

import click

from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group("fts-trigrams")
def fts_trigrams() -> None:
    """Manage FTS trigram field configuration."""
    pass


@fts_trigrams.command("list")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
def list_fields(table: str, region: str) -> None:
    """List FTS trigram fields."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)
    click.echo("Always enabled: experiment_name, run_name, model_name")
    fields = config.get_fts_trigram_fields()
    if fields:
        click.echo("Additional fields:")
        for f in fields:
            click.echo(f"  {f}")
    else:
        click.echo("No additional fields configured.")


@fts_trigrams.command("add")
@click.option("--table", required=True, help="DynamoDB table name")
@click.option("--region", required=True, help="AWS region")
@click.argument("field")
def add_field(table: str, region: str, field: str) -> None:
    """Add a trigram field."""
    ddb_table = DynamoDBTable(table_name=table, region=region)
    config = ConfigReader(table=ddb_table)
    fields = config.get_fts_trigram_fields()
    if field not in fields:
        fields.append(field)
    config.set_fts_trigram_fields(fields)
    click.echo(f"Added field: {field}")
