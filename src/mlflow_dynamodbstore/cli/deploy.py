"""deploy CLI command."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


@click.command()
@pass_context
def deploy(ctx: CliContext) -> None:
    """Create the CloudFormation stack and seed initial data."""
    ensure_stack_exists(
        ctx.name,
        ctx.region,
        ctx.endpoint_url,
        bucket_name=ctx.bucket,
        iam_format=ctx.iam_format,
        permission_boundary=ctx.permission_boundary,
    )
    click.echo(f"Stack '{ctx.name}' deployed successfully.")
