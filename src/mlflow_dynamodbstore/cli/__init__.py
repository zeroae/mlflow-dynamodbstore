"""Admin CLI for mlflow-dynamodbstore."""

from __future__ import annotations

import os

import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context

__all__ = ["CliContext", "pass_context", "cli"]


def _default_region() -> str | None:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


@click.group()
@click.option("--name", default="mlflow-dynamodbstore", show_default=True, help="Stack/table name")
@click.option(
    "--region",
    default=_default_region,
    help="AWS region (defaults to AWS_REGION or AWS_DEFAULT_REGION)",
)
@click.option("--endpoint-url", default=None, help="Custom endpoint URL (for LocalStack/testing)")
@click.pass_context
def cli(ctx: click.Context, name: str, region: str | None, endpoint_url: str | None) -> None:
    """mlflow-dynamodbstore admin commands."""
    if not region:
        raise click.UsageError(
            "Missing --region. Set AWS_REGION or AWS_DEFAULT_REGION, or pass --region explicitly."
        )
    ctx.ensure_object(dict)
    ctx.obj = CliContext(name=name, region=region, endpoint_url=endpoint_url)


from mlflow_dynamodbstore.cli.deploy import deploy  # noqa: E402
from mlflow_dynamodbstore.cli.destroy import destroy  # noqa: E402
from mlflow_dynamodbstore.cli.fts import fts  # noqa: E402
from mlflow_dynamodbstore.cli.tag import tag  # noqa: E402
from mlflow_dynamodbstore.cli.trace import trace  # noqa: E402
from mlflow_dynamodbstore.cli.ttl import ttl  # noqa: E402
from mlflow_dynamodbstore.cli.workspace import workspace  # noqa: E402

cli.add_command(deploy)
cli.add_command(destroy)
cli.add_command(fts)
cli.add_command(tag)
cli.add_command(trace)
cli.add_command(ttl)
cli.add_command(workspace)
