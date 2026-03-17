"""DynamoDB-backed MLflow auth store."""

from __future__ import annotations

from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.server.auth.entities import (
    ExperimentPermission,
    RegisteredModelPermission,
    ScorerPermission,
    User,
    WorkspacePermission,
)
from werkzeug.security import check_password_hash, generate_password_hash

from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_AUTH_USERS,
    GSI2_NAME,
    GSI2_PK,
    GSI2_SK,
    GSI4_NAME,
    GSI4_PERM_PREFIX,
    GSI4_PK,
    GSI4_SK,
    PK_USER_PREFIX,
    SK_USER_META,
    SK_USER_PERM_PREFIX,
)
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable
from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


def _user_id_from_username(username: str) -> int:
    """Derive a stable integer ID from a username (DynamoDB has no auto-increment)."""
    return hash(username) & 0x7FFFFFFF


def _item_to_user(item: dict[str, Any]) -> User:
    """Convert a DynamoDB item to an MLflow User entity."""
    username = item["PK"].removeprefix(PK_USER_PREFIX)
    return User(  # type: ignore[no-untyped-call]
        id_=_user_id_from_username(username),
        username=username,
        password_hash=item.get("password_hash", ""),
        is_admin=item.get("is_admin", False),
        experiment_permissions=[],
        registered_model_permissions=[],
    )


class DynamoDBAuthStore:
    """Auth store backed by DynamoDB.

    User items are stored as:

        PK = USER#<username>
        SK = U#META

    They are indexed on GSI2 with:

        gsi2pk = AUTH_USERS
        gsi2sk = <username>
    """

    def __init__(self, store_uri: str) -> None:
        uri = parse_dynamodb_uri(store_uri)
        if uri.deploy:
            ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)

    def init_db(self, *args: Any, **kwargs: Any) -> None:
        """No-op — table is created by the provisioner."""
        pass

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    def create_user(self, username: str, password: str, is_admin: bool = False) -> User:
        """Create a new user.

        Raises MlflowException if a user with the given username already exists.
        """
        password_hash = generate_password_hash(password)
        item: dict[str, Any] = {
            "PK": f"{PK_USER_PREFIX}{username}",
            "SK": SK_USER_META,
            "password_hash": password_hash,
            "is_admin": is_admin,
            # GSI2: list all users
            GSI2_PK: GSI2_AUTH_USERS,
            GSI2_SK: username,
        }
        try:
            self._table.put_item(item, condition="attribute_not_exists(PK)")
        except Exception as exc:
            if "ConditionalCheckFailedException" in str(exc):
                raise MlflowException(  # type: ignore[no-untyped-call]
                    f"User '{username}' already exists.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from exc
            raise

        return User(  # type: ignore[no-untyped-call]
            id_=_user_id_from_username(username),
            username=username,
            password_hash=password_hash,
            is_admin=is_admin,
            experiment_permissions=[],
            registered_model_permissions=[],
        )

    def authenticate_user(self, username: str, password: str) -> bool:
        """Verify a user's password. Returns True if valid, False otherwise."""
        item = self._table.get_item(
            pk=f"{PK_USER_PREFIX}{username}",
            sk=SK_USER_META,
            consistent=True,
        )
        if item is None:
            return False
        return check_password_hash(item.get("password_hash", ""), password)

    def has_user(self, username: str) -> bool:
        """Return True if the user exists."""
        item = self._table.get_item(
            pk=f"{PK_USER_PREFIX}{username}",
            sk=SK_USER_META,
        )
        return item is not None

    def get_user(self, username: str) -> User:
        """Return a User entity.

        Raises MlflowException if the user does not exist.
        """
        item = self._table.get_item(
            pk=f"{PK_USER_PREFIX}{username}",
            sk=SK_USER_META,
            consistent=True,
        )
        if item is None:
            raise MlflowException(  # type: ignore[no-untyped-call]
                f"User '{username}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return _item_to_user(item)

    def list_users(self) -> list[User]:
        """Return all users."""
        items = self._table.query(
            pk=GSI2_AUTH_USERS,
            index_name=GSI2_NAME,
        )
        return [_item_to_user(item) for item in items]

    def update_user(
        self,
        username: str,
        password: str | None = None,
        is_admin: bool | None = None,
    ) -> None:
        """Update a user's password and/or admin status."""
        updates: dict[str, Any] = {}
        if password is not None:
            updates["password_hash"] = generate_password_hash(password)
        if is_admin is not None:
            updates["is_admin"] = is_admin

        if updates:
            self._table.update_item(
                pk=f"{PK_USER_PREFIX}{username}",
                sk=SK_USER_META,
                updates=updates,
            )

    def delete_user(self, username: str) -> None:
        """Delete a user and all associated items in the USER# partition."""
        pk = f"{PK_USER_PREFIX}{username}"
        # Query all items in this user's partition and delete them
        items = self._table.query(pk=pk)
        for item in items:
            self._table.delete_item(pk=pk, sk=item["SK"])

    # ------------------------------------------------------------------
    # Experiment Permission CRUD
    # ------------------------------------------------------------------

    def create_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> ExperimentPermission:
        """Create an experiment permission for a user.

        Raises MlflowException if the permission already exists.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}EXP#{experiment_id}"
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "permission": permission,
            # GSI4: look up all users with permissions on an experiment
            GSI4_PK: f"{GSI4_PERM_PREFIX}EXP#{experiment_id}",
            GSI4_SK: f"{PK_USER_PREFIX}{username}",
        }
        try:
            self._table.put_item(item, condition="attribute_not_exists(PK)")
        except Exception as exc:
            if "ConditionalCheckFailedException" in str(exc):
                raise MlflowException(  # type: ignore[no-untyped-call]
                    f"Permission for experiment '{experiment_id}' "
                    f"and user '{username}' already exists.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from exc
            raise

        return ExperimentPermission(  # type: ignore[no-untyped-call]
            experiment_id=experiment_id,
            user_id=_user_id_from_username(username),
            permission=permission,
        )

    def get_experiment_permission(self, experiment_id: str, username: str) -> ExperimentPermission:
        """Return an experiment permission.

        Raises MlflowException if the permission does not exist.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}EXP#{experiment_id}"
        item = self._table.get_item(pk=pk, sk=sk, consistent=True)
        if item is None:
            raise MlflowException(  # type: ignore[no-untyped-call]
                f"Permission for experiment '{experiment_id}' and user '{username}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return ExperimentPermission(  # type: ignore[no-untyped-call]
            experiment_id=experiment_id,
            user_id=_user_id_from_username(username),
            permission=item["permission"],
        )

    def list_experiment_permissions(self, username: str) -> list[ExperimentPermission]:
        """Return all experiment permissions for a user."""
        pk = f"{PK_USER_PREFIX}{username}"
        items = self._table.query(pk=pk, sk_prefix=f"{SK_USER_PERM_PREFIX}EXP#")
        user_id = _user_id_from_username(username)
        return [
            ExperimentPermission(  # type: ignore[no-untyped-call]
                experiment_id=item["SK"].removeprefix(f"{SK_USER_PERM_PREFIX}EXP#"),
                user_id=user_id,
                permission=item["permission"],
            )
            for item in items
        ]

    def update_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> None:
        """Update an experiment permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}EXP#{experiment_id}"
        self._table.update_item(pk=pk, sk=sk, updates={"permission": permission})

    def delete_experiment_permission(self, experiment_id: str, username: str) -> None:
        """Delete an experiment permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}EXP#{experiment_id}"
        self._table.delete_item(pk=pk, sk=sk)

    # ------------------------------------------------------------------
    # Registered Model Permission CRUD
    # ------------------------------------------------------------------

    def create_registered_model_permission(
        self, name: str, username: str, permission: str
    ) -> RegisteredModelPermission:
        """Create a registered model permission for a user.

        Raises MlflowException if the permission already exists.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}MODEL#default#{name}"
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "permission": permission,
            GSI4_PK: f"{GSI4_PERM_PREFIX}MODEL#default#{name}",
            GSI4_SK: f"{PK_USER_PREFIX}{username}",
        }
        try:
            self._table.put_item(item, condition="attribute_not_exists(PK)")
        except Exception as exc:
            if "ConditionalCheckFailedException" in str(exc):
                raise MlflowException(  # type: ignore[no-untyped-call]
                    f"Permission for model '{name}' and user '{username}' already exists.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from exc
            raise

        return RegisteredModelPermission(  # type: ignore[no-untyped-call]
            name=name,
            user_id=_user_id_from_username(username),
            permission=permission,
        )

    def get_registered_model_permission(
        self, name: str, username: str
    ) -> RegisteredModelPermission:
        """Return a registered model permission.

        Raises MlflowException if the permission does not exist.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}MODEL#default#{name}"
        item = self._table.get_item(pk=pk, sk=sk, consistent=True)
        if item is None:
            raise MlflowException(  # type: ignore[no-untyped-call]
                f"Permission for model '{name}' and user '{username}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return RegisteredModelPermission(  # type: ignore[no-untyped-call]
            name=name,
            user_id=_user_id_from_username(username),
            permission=item["permission"],
        )

    def list_registered_model_permissions(self, username: str) -> list[RegisteredModelPermission]:
        """Return all registered model permissions for a user."""
        pk = f"{PK_USER_PREFIX}{username}"
        items = self._table.query(pk=pk, sk_prefix=f"{SK_USER_PERM_PREFIX}MODEL#default#")
        user_id = _user_id_from_username(username)
        return [
            RegisteredModelPermission(  # type: ignore[no-untyped-call]
                name=item["SK"].removeprefix(f"{SK_USER_PERM_PREFIX}MODEL#default#"),
                user_id=user_id,
                permission=item["permission"],
            )
            for item in items
        ]

    def update_registered_model_permission(self, name: str, username: str, permission: str) -> None:
        """Update a registered model permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}MODEL#default#{name}"
        self._table.update_item(pk=pk, sk=sk, updates={"permission": permission})

    def delete_registered_model_permission(self, name: str, username: str) -> None:
        """Delete a registered model permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}MODEL#default#{name}"
        self._table.delete_item(pk=pk, sk=sk)

    def delete_registered_model_permissions(self, name: str) -> None:
        """Bulk-delete all permissions for a registered model (all users)."""
        gsi4pk = f"{GSI4_PERM_PREFIX}MODEL#default#{name}"
        items = self._table.query(pk=gsi4pk, index_name=GSI4_NAME)
        for item in items:
            self._table.delete_item(pk=item["PK"], sk=item["SK"])

    def rename_registered_model_permissions(self, old_name: str, new_name: str) -> None:
        """Rename all permissions from old_name to new_name."""
        gsi4pk = f"{GSI4_PERM_PREFIX}MODEL#default#{old_name}"
        items = self._table.query(pk=gsi4pk, index_name=GSI4_NAME)
        for item in items:
            # Delete old
            self._table.delete_item(pk=item["PK"], sk=item["SK"])
            # Write new
            username = item["PK"].removeprefix(PK_USER_PREFIX)
            new_sk = f"{SK_USER_PERM_PREFIX}MODEL#default#{new_name}"
            new_item: dict[str, Any] = {
                "PK": item["PK"],
                "SK": new_sk,
                "permission": item["permission"],
                GSI4_PK: f"{GSI4_PERM_PREFIX}MODEL#default#{new_name}",
                GSI4_SK: f"{PK_USER_PREFIX}{username}",
            }
            self._table.put_item(new_item)

    # ------------------------------------------------------------------
    # Workspace Permission CRUD
    # ------------------------------------------------------------------

    def set_workspace_permission(self, workspace: str, username: str, permission: str) -> None:
        """Create or update a workspace permission (upsert)."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}WORKSPACE#{workspace}"
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "permission": permission,
            GSI4_PK: f"{GSI4_PERM_PREFIX}WORKSPACE#{workspace}",
            GSI4_SK: f"{PK_USER_PREFIX}{username}",
        }
        self._table.put_item(item)  # No condition — upsert

    def get_workspace_permission(self, workspace: str, username: str) -> WorkspacePermission:
        """Return a workspace permission.

        Raises MlflowException if the permission does not exist.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}WORKSPACE#{workspace}"
        item = self._table.get_item(pk=pk, sk=sk, consistent=True)
        if item is None:
            raise MlflowException(  # type: ignore[no-untyped-call]
                f"Permission for workspace '{workspace}' and user '{username}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return WorkspacePermission(  # type: ignore[no-untyped-call]
            workspace=workspace,
            user_id=_user_id_from_username(username),
            permission=item["permission"],
        )

    def list_workspace_permissions(self, workspace: str) -> list[WorkspacePermission]:
        """Return all permissions for a workspace (all users)."""
        gsi4pk = f"{GSI4_PERM_PREFIX}WORKSPACE#{workspace}"
        items = self._table.query(pk=gsi4pk, index_name=GSI4_NAME)
        return [
            WorkspacePermission(  # type: ignore[no-untyped-call]
                workspace=workspace,
                user_id=_user_id_from_username(item[GSI4_SK].removeprefix(PK_USER_PREFIX)),
                permission=item["permission"],
            )
            for item in items
        ]

    def list_user_workspace_permissions(self, username: str) -> list[WorkspacePermission]:
        """Return all workspace permissions for a user."""
        pk = f"{PK_USER_PREFIX}{username}"
        items = self._table.query(pk=pk, sk_prefix=f"{SK_USER_PERM_PREFIX}WORKSPACE#")
        user_id = _user_id_from_username(username)
        return [
            WorkspacePermission(  # type: ignore[no-untyped-call]
                workspace=item["SK"].removeprefix(f"{SK_USER_PERM_PREFIX}WORKSPACE#"),
                user_id=user_id,
                permission=item["permission"],
            )
            for item in items
        ]

    def delete_workspace_permission(self, workspace: str, username: str) -> None:
        """Delete a workspace permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}WORKSPACE#{workspace}"
        self._table.delete_item(pk=pk, sk=sk)

    def delete_workspace_permissions_for_workspace(self, workspace: str) -> None:
        """Bulk-delete all permissions for a workspace (all users)."""
        gsi4pk = f"{GSI4_PERM_PREFIX}WORKSPACE#{workspace}"
        items = self._table.query(pk=gsi4pk, index_name=GSI4_NAME)
        for item in items:
            self._table.delete_item(pk=item["PK"], sk=item["SK"])

    def list_accessible_workspace_names(self, username: str) -> list[str]:
        """Return workspace names that the user has any permission on."""
        pk = f"{PK_USER_PREFIX}{username}"
        items = self._table.query(pk=pk, sk_prefix=f"{SK_USER_PERM_PREFIX}WORKSPACE#")
        return [item["SK"].removeprefix(f"{SK_USER_PERM_PREFIX}WORKSPACE#") for item in items]

    # ------------------------------------------------------------------
    # Scorer Permission CRUD
    # ------------------------------------------------------------------

    def create_scorer_permission(
        self,
        experiment_id: str,
        scorer_name: str,
        username: str,
        permission: str,
    ) -> ScorerPermission:
        """Create a scorer permission for a user.

        Raises MlflowException if the permission already exists.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}SCORER#{experiment_id}#{scorer_name}"
        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "permission": permission,
            GSI4_PK: f"{GSI4_PERM_PREFIX}SCORER#{experiment_id}#{scorer_name}",
            GSI4_SK: f"{PK_USER_PREFIX}{username}",
        }
        try:
            self._table.put_item(item, condition="attribute_not_exists(PK)")
        except Exception as exc:
            if "ConditionalCheckFailedException" in str(exc):
                raise MlflowException(  # type: ignore[no-untyped-call]
                    f"Permission for scorer '{scorer_name}' in experiment "
                    f"'{experiment_id}' and user '{username}' already exists.",
                    error_code=RESOURCE_ALREADY_EXISTS,
                ) from exc
            raise

        return ScorerPermission(  # type: ignore[no-untyped-call]
            experiment_id=experiment_id,
            scorer_name=scorer_name,
            user_id=_user_id_from_username(username),
            permission=permission,
        )

    def get_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str
    ) -> ScorerPermission:
        """Return a scorer permission.

        Raises MlflowException if the permission does not exist.
        """
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}SCORER#{experiment_id}#{scorer_name}"
        item = self._table.get_item(pk=pk, sk=sk, consistent=True)
        if item is None:
            raise MlflowException(  # type: ignore[no-untyped-call]
                f"Permission for scorer '{scorer_name}' in experiment "
                f"'{experiment_id}' and user '{username}' not found.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return ScorerPermission(  # type: ignore[no-untyped-call]
            experiment_id=experiment_id,
            scorer_name=scorer_name,
            user_id=_user_id_from_username(username),
            permission=item["permission"],
        )

    def list_scorer_permissions(self, username: str) -> list[ScorerPermission]:
        """Return all scorer permissions for a user."""
        pk = f"{PK_USER_PREFIX}{username}"
        items = self._table.query(pk=pk, sk_prefix=f"{SK_USER_PERM_PREFIX}SCORER#")
        user_id = _user_id_from_username(username)
        result = []
        for item in items:
            suffix = item["SK"].removeprefix(f"{SK_USER_PERM_PREFIX}SCORER#")
            # suffix is "<experiment_id>#<scorer_name>"
            experiment_id, scorer_name = suffix.split("#", 1)
            result.append(
                ScorerPermission(  # type: ignore[no-untyped-call]
                    experiment_id=experiment_id,
                    scorer_name=scorer_name,
                    user_id=user_id,
                    permission=item["permission"],
                )
            )
        return result

    def update_scorer_permission(
        self,
        experiment_id: str,
        scorer_name: str,
        username: str,
        permission: str,
    ) -> None:
        """Update a scorer permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}SCORER#{experiment_id}#{scorer_name}"
        self._table.update_item(pk=pk, sk=sk, updates={"permission": permission})

    def delete_scorer_permission(self, experiment_id: str, scorer_name: str, username: str) -> None:
        """Delete a scorer permission."""
        pk = f"{PK_USER_PREFIX}{username}"
        sk = f"{SK_USER_PERM_PREFIX}SCORER#{experiment_id}#{scorer_name}"
        self._table.delete_item(pk=pk, sk=sk)

    def delete_scorer_permissions_for_scorer(self, experiment_id: str, scorer_name: str) -> None:
        """Bulk-delete all permissions for a scorer (all users)."""
        gsi4pk = f"{GSI4_PERM_PREFIX}SCORER#{experiment_id}#{scorer_name}"
        items = self._table.query(pk=gsi4pk, index_name=GSI4_NAME)
        for item in items:
            self._table.delete_item(pk=item["PK"], sk=item["SK"])

    # ------------------------------------------------------------------
    # Gateway permissions (not supported — Databricks-specific)
    # ------------------------------------------------------------------

    def create_gateway_secret_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway secret permissions are Databricks-specific.")

    def get_gateway_secret_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway secret permissions are Databricks-specific.")

    def list_gateway_secret_permissions(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway secret permissions are Databricks-specific.")

    def update_gateway_secret_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway secret permissions are Databricks-specific.")

    def delete_gateway_secret_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway secret permissions are Databricks-specific.")

    def create_gateway_endpoint_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway endpoint permissions are Databricks-specific.")

    def get_gateway_endpoint_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway endpoint permissions are Databricks-specific.")

    def list_gateway_endpoint_permissions(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway endpoint permissions are Databricks-specific.")

    def update_gateway_endpoint_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway endpoint permissions are Databricks-specific.")

    def delete_gateway_endpoint_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway endpoint permissions are Databricks-specific.")

    def create_gateway_model_definition_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway model definition permissions are Databricks-specific.")

    def get_gateway_model_definition_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway model definition permissions are Databricks-specific.")

    def list_gateway_model_definition_permissions(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway model definition permissions are Databricks-specific.")

    def update_gateway_model_definition_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway model definition permissions are Databricks-specific.")

    def delete_gateway_model_definition_permission(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Gateway model definition permissions are Databricks-specific.")
