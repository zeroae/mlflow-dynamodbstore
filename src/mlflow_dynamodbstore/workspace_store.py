"""DynamoDB-backed MLflow workspace store."""

from __future__ import annotations

from typing import Any

from mlflow.entities import Workspace
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.workspace.abstract_store import AbstractStore

from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_EXPERIMENTS_PREFIX,
    GSI2_MODELS_PREFIX,
    GSI2_PK,
    GSI2_SK,
    GSI2_WORKSPACES,
    PK_WORKSPACE_PREFIX,
    SK_WORKSPACE_META,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


class DynamoDBWorkspaceStore(AbstractStore):
    """Workspace provider backed by DynamoDB.

    Implements the MLflow ``AbstractStore`` interface for workspaces.
    Workspace items are stored as:

        PK = WORKSPACE#<name>
        SK = META

    They are indexed on GSI2 with:

        gsi2pk = WORKSPACES
        gsi2sk = <name>
    """

    def __init__(self, workspace_uri: str | None = None, store_uri: str | None = None) -> None:
        resolved_uri = workspace_uri or store_uri
        if resolved_uri is None:
            raise TypeError("DynamoDBWorkspaceStore requires 'workspace_uri' argument")
        uri = parse_dynamodb_uri(resolved_uri)
        if uri.deploy:
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
            self._put_workspace(
                "default",
                description="Default workspace for legacy resources",
                default_artifact_root="",
            )

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

    @staticmethod
    def _item_to_workspace(item: dict[str, Any]) -> Workspace:
        return Workspace(
            name=item["name"],
            description=item.get("description") or None,
            default_artifact_root=item.get("default_artifact_root") or None,
        )

    def _check_workspace_empty(self, workspace_name: str) -> None:
        """Raise if the workspace still contains experiments or models."""
        # Check for active or deleted experiments
        for lifecycle in ("active", "deleted"):
            items = self._table.query(
                pk=f"{GSI2_EXPERIMENTS_PREFIX}{workspace_name}#{lifecycle}",
                index_name="gsi2",
                limit=1,
            )
            if items:
                raise MlflowException(
                    f"Cannot delete workspace '{workspace_name}': "
                    f"'experiments' still contains resource(s). "
                    "Remove or reassign them before deleting the workspace.",
                    error_code=INVALID_STATE,
                )
        # Check for registered models
        items = self._table.query(
            pk=f"{GSI2_MODELS_PREFIX}{workspace_name}",
            index_name="gsi2",
            limit=1,
        )
        if items:
            raise MlflowException(
                f"Cannot delete workspace '{workspace_name}': "
                f"'registered_models' still contains resource(s). "
                "Remove or reassign them before deleting the workspace.",
                error_code=INVALID_STATE,
            )

    # ------------------------------------------------------------------
    # AbstractStore interface
    # ------------------------------------------------------------------

    def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create a new workspace.

        Raises:
            MlflowException: If a workspace with the given name already exists.
        """
        from botocore.exceptions import ClientError
        from mlflow.store.workspace.abstract_store import WorkspaceNameValidator

        WorkspaceNameValidator.validate(workspace.name)
        try:
            self._put_workspace(
                workspace.name,
                description=workspace.description or "",
                default_artifact_root=workspace.default_artifact_root or "",
                condition="attribute_not_exists(PK)",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise MlflowException(
                    f"Workspace '{workspace.name}' already exists.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from None
            raise
        return workspace

    def get_workspace(self, workspace_name: str) -> Workspace:
        """Return a Workspace entity.

        Raises:
            MlflowException: If the workspace does not exist.
        """
        item = self._table.get_item(
            pk=f"{PK_WORKSPACE_PREFIX}{workspace_name}",
            sk=SK_WORKSPACE_META,
        )
        if item is None:
            raise MlflowException(
                f"Workspace '{workspace_name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return self._item_to_workspace(item)

    def list_workspaces(self) -> list[Workspace]:
        """Return all workspaces as Workspace entities."""
        items = self._table.query(
            pk=GSI2_WORKSPACES,
            index_name="gsi2",
        )
        return [self._item_to_workspace(item) for item in items]

    def update_workspace(self, workspace: Workspace) -> Workspace:
        """Update mutable workspace attributes.

        An empty string for ``default_artifact_root`` signals "clear this field".
        """
        updates: dict[str, Any] = {}
        if workspace.description is not None:
            updates["description"] = workspace.description
        if workspace.default_artifact_root is not None:
            # Empty string means "clear" — store empty, _item_to_workspace returns None
            updates["default_artifact_root"] = workspace.default_artifact_root

        if updates:
            self._table.update_item(
                pk=f"{PK_WORKSPACE_PREFIX}{workspace.name}",
                sk=SK_WORKSPACE_META,
                updates=updates,
            )
        return self.get_workspace(workspace.name)

    def delete_workspace(
        self,
        workspace_name: str,
        mode: WorkspaceDeletionMode = WorkspaceDeletionMode.RESTRICT,
    ) -> None:
        """Delete a workspace.

        Raises:
            MlflowException: If attempting to delete the built-in 'default' workspace.
        """
        if workspace_name == "default":
            raise MlflowException(
                f"Cannot delete the reserved '{workspace_name}' workspace",
                error_code=INVALID_STATE,
            )
        if mode == WorkspaceDeletionMode.RESTRICT:
            self._check_workspace_empty(workspace_name)
        self._table.delete_item(
            pk=f"{PK_WORKSPACE_PREFIX}{workspace_name}",
            sk=SK_WORKSPACE_META,
        )

    def get_default_workspace(self) -> Workspace:
        """Return the default workspace."""
        return self.get_workspace("default")

    def resolve_artifact_root(
        self, default_artifact_root: str | None, workspace_name: str
    ) -> tuple[str | None, bool]:
        """Allow per-workspace artifact storage roots.

        Returns the workspace's ``default_artifact_root`` if configured, otherwise
        falls back to the server's ``default_artifact_root``.
        """
        try:
            ws = self.get_workspace(workspace_name)
            if ws.default_artifact_root:
                return ws.default_artifact_root, False
        except MlflowException:
            pass
        return default_artifact_root, True
