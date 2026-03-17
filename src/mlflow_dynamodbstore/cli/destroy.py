"""destroy CLI command."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.provisioner import destroy_stack


@click.command()
@click.option("--yes", "confirmed", is_flag=True, help="Skip confirmation prompt")
@click.option("--retain", is_flag=True, help="Delete stack but keep the DynamoDB table")
@pass_context
def destroy(ctx: CliContext, confirmed: bool, retain: bool) -> None:
    """Delete the CloudFormation stack."""
    if not confirmed:
        click.confirm(
            f"Destroy stack '{ctx.name}'? This will delete the CloudFormation stack"
            + (" (table will be retained)" if retain else " and the DynamoDB table"),
            abort=True,
        )
    destroy_stack(ctx.name, ctx.region, ctx.endpoint_url, retain=retain)
    click.echo(f"Stack '{ctx.name}' destroyed.")
