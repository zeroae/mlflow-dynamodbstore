"""workspace CLI commands."""

from __future__ import annotations

import click

from mlflow_dynamodbstore.cli import CliContext, pass_context
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_EXPERIMENTS_PREFIX,
    GSI2_MODELS_PREFIX,
    PK_EXPERIMENT_PREFIX,
    PK_MODEL_PREFIX,
    PK_WORKSPACE_PREFIX,
    SK_WORKSPACE_META,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


@click.group("workspace")
def workspace() -> None:
    """Workspace operations."""
    pass


@workspace.command("delete")
@click.option("--workspace", "workspace_name", required=True, help="Workspace name to delete")
@click.option(
    "--mode",
    type=click.Choice(["soft", "cascade"]),
    default="soft",
    help="Deletion mode",
)
@click.option("--yes", "confirmed", is_flag=True, help="Skip confirmation for cascade")
@pass_context
def delete(ctx: CliContext, workspace_name: str, mode: str, confirmed: bool) -> None:
    """Delete a workspace."""
    if workspace_name == "default":
        click.echo("Error: Cannot delete the default workspace.", err=True)
        raise SystemExit(1)

    ddb_table = DynamoDBTable(ctx.name, ctx.region, ctx.endpoint_url)
    pk = f"{PK_WORKSPACE_PREFIX}{workspace_name}"

    # Verify workspace exists
    meta = ddb_table.get_item(pk=pk, sk=SK_WORKSPACE_META)
    if not meta:
        click.echo(f"Error: Workspace '{workspace_name}' not found.", err=True)
        raise SystemExit(1)

    if mode == "soft":
        ddb_table.update_item(pk=pk, sk=SK_WORKSPACE_META, updates={"status": "deleted"})
        click.echo(f"Workspace '{workspace_name}' marked as deleted.")

    elif mode == "cascade":
        if not confirmed:
            click.confirm(
                f"Delete workspace '{workspace_name}' and ALL its experiments and models?",
                abort=True,
            )

        deleted_experiments = 0
        deleted_models = 0

        # Delete all experiments in workspace
        for lifecycle in ("active", "deleted"):
            gsi2pk = f"{GSI2_EXPERIMENTS_PREFIX}{workspace_name}#{lifecycle}"
            results = ddb_table.query(pk=gsi2pk, index_name="gsi2")
            for item in results:
                exp_pk = item.get("PK", "")
                if exp_pk.startswith(PK_EXPERIMENT_PREFIX):
                    exp_items = ddb_table.query(pk=exp_pk)
                    for exp_item in exp_items:
                        ddb_table.delete_item(pk=exp_item["PK"], sk=exp_item["SK"])
                    deleted_experiments += 1

        # Delete all models in workspace
        for lifecycle in ("active", "deleted"):
            gsi2pk = f"{GSI2_MODELS_PREFIX}{workspace_name}#{lifecycle}"
            results = ddb_table.query(pk=gsi2pk, index_name="gsi2")
            for item in results:
                model_pk = item.get("PK", "")
                if model_pk.startswith(PK_MODEL_PREFIX):
                    model_items = ddb_table.query(pk=model_pk)
                    for model_item in model_items:
                        ddb_table.delete_item(pk=model_item["PK"], sk=model_item["SK"])
                    deleted_models += 1

        # Delete workspace META
        ddb_table.delete_item(pk=pk, sk=SK_WORKSPACE_META)
        click.echo(
            f"Deleted workspace '{workspace_name}': "
            f"{deleted_experiments} experiments, {deleted_models} models removed."
        )
