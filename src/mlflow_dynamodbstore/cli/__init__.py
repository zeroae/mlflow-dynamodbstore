"""Admin CLI for mlflow-dynamodbstore."""

import click

from mlflow_dynamodbstore.cli.cache_spans import cache_spans
from mlflow_dynamodbstore.cli.cleanup_expired import cleanup_expired
from mlflow_dynamodbstore.cli.delete_workspace import delete_workspace
from mlflow_dynamodbstore.cli.denormalize_tags import denormalize_tags
from mlflow_dynamodbstore.cli.fts_trigrams import fts_trigrams
from mlflow_dynamodbstore.cli.ttl_policy import ttl_policy


class CliContext:
    """Shared context for all CLI commands."""

    def __init__(self, name: str, region: str, endpoint_url: str | None) -> None:
        self.name = name
        self.region = region
        self.endpoint_url = endpoint_url


pass_context = click.make_pass_decorator(CliContext)


@click.group()
def cli() -> None:
    """mlflow-dynamodbstore admin commands."""
    pass


cli.add_command(cache_spans)
cli.add_command(cleanup_expired)
cli.add_command(delete_workspace)
cli.add_command(denormalize_tags)
cli.add_command(fts_trigrams)
cli.add_command(ttl_policy)
