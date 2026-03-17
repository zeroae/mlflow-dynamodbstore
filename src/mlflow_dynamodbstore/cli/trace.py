"""trace CLI commands."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli._context import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.schema import (
    PK_EXPERIMENT_PREFIX,
    SK_TRACE_PREFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.tracking_store import DynamoDBTrackingStore


@click.group("trace")
def trace() -> None:
    """Trace operations."""
    pass


@trace.command("cache")
@click.option("--experiment-id", required=True, multiple=True, help="Experiment ID(s)")
@click.option("--days", type=int, default=None, help="Only process traces newer than N days")
@pass_context
def cache(ctx: CliContext, experiment_id: tuple[str, ...], days: int | None) -> None:
    """Pre-cache X-Ray spans for traces."""
    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)
    store = DynamoDBTrackingStore(
        store_uri=f"dynamodb://{ctx.region}/{ctx.name}",
        artifact_uri="",  # Not needed for trace operations
    )

    cached_count = 0
    skipped_count = 0

    for exp_id in experiment_id:
        pk = f"{PK_EXPERIMENT_PREFIX}{exp_id}"
        # Query all trace items under this experiment
        traces = ddb_table.query(pk=pk, sk_prefix=SK_TRACE_PREFIX)

        for trace_item in traces:
            sk = trace_item.get("SK", "")
            # Only process META items (T#<trace_id>, not T#<trace_id>#...)
            if "#" in sk[2:]:  # Skip children like T#<id>#TAG#..., T#<id>#SPANS
                continue

            trace_id = sk[2:]  # Remove "T#" prefix

            # Check if spans are already cached
            spans_items = ddb_table.query(pk=pk, sk_prefix=f"{SK_TRACE_PREFIX}{trace_id}#SPANS")
            if spans_items:
                skipped_count += 1
                continue

            try:
                store.get_trace(trace_id)
                cached_count += 1
                click.echo(f"Cached spans for trace {trace_id}")
            except Exception as e:
                click.echo(f"Error caching trace {trace_id}: {e}", err=True)

    click.echo(f"Cached: {cached_count}, Already cached: {skipped_count}")
