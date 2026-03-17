"""Admin CLI for mlflow-dynamodbstore."""

from __future__ import annotations

import cloup

from mlflow_dynamodbstore.cli._context import CliContext, pass_context

__all__ = ["CliContext", "pass_context", "cli"]


@cloup.group()
@cloup.option("--name", default="mlflow", show_default=True, help="Stack/table name")
@cloup.option("--region", default=None, help="AWS region (omit to use boto3 default chain)")
@cloup.option("--endpoint-url", default=None, help="Custom endpoint URL (for LocalStack/testing)")
@cloup.pass_context
def cli(ctx: cloup.Context, name: str, region: str | None, endpoint_url: str | None) -> None:
    """mlflow-dynamodbstore admin commands."""
    ctx.ensure_object(dict)
    ctx.obj = CliContext(name=name, region=region, endpoint_url=endpoint_url)


from mlflow_dynamodbstore.cli.deploy import deploy  # noqa: E402
from mlflow_dynamodbstore.cli.destroy import destroy  # noqa: E402
from mlflow_dynamodbstore.cli.fts import fts  # noqa: E402
from mlflow_dynamodbstore.cli.tag import tag  # noqa: E402
from mlflow_dynamodbstore.cli.trace import trace  # noqa: E402
from mlflow_dynamodbstore.cli.ttl import ttl  # noqa: E402
from mlflow_dynamodbstore.cli.workspace import workspace  # noqa: E402

cli.section(
    "Stack Lifecycle",
    deploy,
    destroy,
)
cli.section(
    "Configuration",
    tag,
    ttl,
    fts,
    trace,
    workspace,
)
