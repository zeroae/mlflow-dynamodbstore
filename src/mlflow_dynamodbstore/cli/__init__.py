"""Admin CLI for mlflow-dynamodbstore."""

import click


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


def _register_workspace_command() -> None:
    """Register workspace command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.workspace import workspace

    cli.add_command(workspace)


def _register_trace_command() -> None:
    """Register trace command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.trace import trace

    cli.add_command(trace)


def _register_fts_command() -> None:
    """Register fts command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.fts import fts

    cli.add_command(fts)


def _register_tag_command() -> None:
    """Register tag command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.tag import tag

    cli.add_command(tag)


def _register_ttl_command() -> None:
    """Register ttl command. Imported lazily to avoid circular imports."""
    from mlflow_dynamodbstore.cli.ttl import ttl

    cli.add_command(ttl)


_register_fts_command()
_register_tag_command()
_register_ttl_command()
_register_trace_command()
_register_workspace_command()
