"""deploy CLI command."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.command()
@pass_context
def deploy(ctx: CliContext) -> None:
    """Create the CloudFormation stack and seed initial data."""
    ensure_stack_exists(ctx.name, ctx.region, ctx.endpoint_url)
    click.echo(f"Stack '{ctx.name}' deployed successfully.")
