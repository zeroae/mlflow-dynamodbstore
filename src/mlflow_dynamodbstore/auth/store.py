"""DynamoDB-backed MLflow auth store — user CRUD methods."""

from __future__ import annotations

from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.server.auth.entities import User
from werkzeug.security import check_password_hash, generate_password_hash

from mlflow_dynamodbstore.dynamodb.provisioner import ensure_stack_exists
from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_AUTH_USERS,
    GSI2_NAME,
    GSI2_PK,
    GSI2_SK,
    PK_USER_PREFIX,
    SK_USER_META,
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
        ensure_stack_exists(uri.table_name, uri.region, uri.endpoint_url)
        self._table = DynamoDBTable(uri.table_name, uri.region, uri.endpoint_url)

    def init_db(self) -> None:
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
