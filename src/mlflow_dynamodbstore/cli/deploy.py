"""deploy CLI command."""

from __future__ import annotations

import boto3
import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists


def _resolve_bucket_name(name: str, region: str | None, endpoint_url: str | None) -> str:
    """Derive a default bucket name from the stack name and AWS account ID."""
    sts = boto3.client("sts", region_name=region, endpoint_url=endpoint_url)
    account_id = sts.get_caller_identity()["Account"]
    return f"{name}-artifacts-{account_id}"


@click.command()
@pass_context
def deploy(ctx: CliContext) -> None:
    """Create the CloudFormation stack and seed initial data."""
    bucket_name = ctx.bucket
    if bucket_name is None:
        bucket_name = _resolve_bucket_name(ctx.name, ctx.region, ctx.endpoint_url)

    ensure_stack_exists(
        ctx.name,
        ctx.region,
        ctx.endpoint_url,
        bucket_name=bucket_name,
        iam_format=ctx.iam_format,
        permission_boundary=ctx.permission_boundary,
    )
    click.echo(f"Stack '{ctx.name}' deployed successfully.")
