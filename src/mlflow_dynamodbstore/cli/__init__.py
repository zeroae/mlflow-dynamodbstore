"""Admin CLI for mlflow-dynamodbstore."""

import click

from mlflow_dynamodbstore.cli.fts_trigrams import fts_trigrams
from mlflow_dynamodbstore.cli.ttl_policy import ttl_policy


@click.group()
def cli() -> None:
    """mlflow-dynamodbstore admin commands."""
    pass


cli.add_command(fts_trigrams)
cli.add_command(ttl_policy)
