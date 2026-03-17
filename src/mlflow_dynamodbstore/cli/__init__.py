"""Admin CLI for mlflow-dynamodbstore."""

import click


class CliContext:
    """Shared context for all CLI commands."""

    def __init__(self, name: str, region: str, endpoint_url: str | None) -> None:
        self.name = name
        self.region = region
        self.endpoint_url = endpoint_url


pass_context = click.make_pass_decorator(CliContext)

from mlflow_dynamodbstore.cli.cache_spans import cache_spans  # noqa: E402
from mlflow_dynamodbstore.cli.delete_workspace import delete_workspace  # noqa: E402
from mlflow_dynamodbstore.cli.fts_trigrams import fts_trigrams  # noqa: E402


@click.group()
def cli() -> None:
    """mlflow-dynamodbstore admin commands."""
    pass


cli.add_command(cache_spans)
cli.add_command(delete_workspace)
cli.add_command(fts_trigrams)


def _register_tag_command() -> None:
    """Register tag command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.tag import tag

    cli.add_command(tag)


def _register_ttl_command() -> None:
    """Register ttl command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.ttl import ttl

    cli.add_command(ttl)


_register_tag_command()
_register_ttl_command()
