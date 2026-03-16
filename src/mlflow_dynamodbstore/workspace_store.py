"""DynamoDB-backed MLflow workspace store (duck-typed, no ABC)."""

from __future__ import annotations

from typing import Any

from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_PK,
    GSI2_SK,
    GSI2_WORKSPACES,
    PK_WORKSPACE_PREFIX,
    SK_WORKSPACE_META,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


class DynamoDBWorkspaceStore:
    """Workspace provider backed by DynamoDB.

    This class is duck-typed — it does not inherit from any MLflow abstract
    base class.  Workspace items are stored as:

        PK = WORKSPACE#<name>
        SK = META

    They are indexed on GSI2 with:

        gsi2pk = WORKSPACES
        gsi2sk = <name>
    """

    def __init__(self, store_uri: str) -> None:
        uri = parse_dynamodb_uri(store_uri)
        ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)
        # Ensure the default workspace exists on startup
        self._ensure_default_workspace()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_default_workspace(self) -> None:
        """Create the default workspace if it does not already exist."""
        existing = self._table.get_item(
            pk=f"{PK_WORKSPACE_PREFIX}default",
            sk=SK_WORKSPACE_META,
        )
        if existing is None:
            self._put_workspace("default", description="", default_artifact_root="")

    def _put_workspace(
        self,
        name: str,
        description: str = "",
        default_artifact_root: str = "",
        condition: str | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "PK": f"{PK_WORKSPACE_PREFIX}{name}",
            "SK": SK_WORKSPACE_META,
            "name": name,
            "description": description,
            "default_artifact_root": default_artifact_root,
            # GSI2: list all workspaces
            GSI2_PK: GSI2_WORKSPACES,
            GSI2_SK: name,
        }
        self._table.put_item(item, condition=condition)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_workspace(
        self,
        name: str,
        description: str = "",
        default_artifact_root: str = "",
    ) -> None:
        """Create a new workspace.

        Raises:
            ClientError: If a workspace with the given name already exists
                (condition expression violation).
        """
        self._put_workspace(
            name,
            description=description,
            default_artifact_root=default_artifact_root,
            condition="attribute_not_exists(PK)",
        )

    def get_workspace(self, name: str) -> dict[str, Any] | None:
        """Return workspace dict or None if not found."""
        return self._table.get_item(
            pk=f"{PK_WORKSPACE_PREFIX}{name}",
            sk=SK_WORKSPACE_META,
        )

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Return all workspace dicts."""
        return self._table.query(
            pk=GSI2_WORKSPACES,
            index_name="gsi2",
        )

    def update_workspace(
        self,
        name: str,
        description: str | None = None,
        default_artifact_root: str | None = None,
    ) -> None:
        """Update mutable workspace attributes."""
        updates: dict[str, Any] = {}
        if description is not None:
            updates["description"] = description
        if default_artifact_root is not None:
            updates["default_artifact_root"] = default_artifact_root

        if updates:
            self._table.update_item(
                pk=f"{PK_WORKSPACE_PREFIX}{name}",
                sk=SK_WORKSPACE_META,
                updates=updates,
            )

    def delete_workspace(self, name: str) -> None:
        """Delete a workspace.

        Raises:
            ValueError: If attempting to delete the built-in 'default' workspace.
        """
        if name == "default":
            raise ValueError("Cannot delete the default workspace.")
        self._table.delete_item(
            pk=f"{PK_WORKSPACE_PREFIX}{name}",
            sk=SK_WORKSPACE_META,
        )

    def resolve_artifact_root(self, workspace_name: str, artifact_uri: str | None = None) -> str:
        """Return the effective artifact root for a workspace.

        If *artifact_uri* is provided it takes precedence; otherwise the
        workspace's ``default_artifact_root`` is returned.
        """
        if artifact_uri:
            return artifact_uri
        ws = self.get_workspace(workspace_name)
        if ws is None:
            return ""
        return str(ws.get("default_artifact_root", ""))
