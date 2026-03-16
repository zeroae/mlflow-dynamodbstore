"""Integration tests for DynamoDBAuthStore — full auth lifecycle via moto server."""

from __future__ import annotations

import pytest
from mlflow.exceptions import MlflowException


class TestAuthLifecycle:
    def test_create_and_authenticate_user(self, auth_store):
        """Create user -> authenticate with correct/wrong password."""
        user = auth_store.create_user("alice", "s3cret")
        assert user.username == "alice"
        assert user.is_admin is False

        # Correct password
        assert auth_store.authenticate_user("alice", "s3cret") is True

        # Wrong password
        assert auth_store.authenticate_user("alice", "wrong") is False

        # Non-existent user
        assert auth_store.authenticate_user("nobody", "s3cret") is False

    def test_create_user_duplicate_raises(self, auth_store):
        """Creating the same user twice raises RESOURCE_ALREADY_EXISTS."""
        auth_store.create_user("alice", "pass1")
        with pytest.raises(MlflowException, match="already exists") as exc_info:
            auth_store.create_user("alice", "pass2")
        assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"

    def test_create_update_delete_user(self, auth_store):
        """Full user lifecycle: create -> update password -> update admin -> delete."""
        auth_store.create_user("bob", "original")
        assert auth_store.has_user("bob") is True

        # Update password
        auth_store.update_user("bob", password="newpass")
        assert auth_store.authenticate_user("bob", "newpass") is True
        assert auth_store.authenticate_user("bob", "original") is False

        # Update admin status
        auth_store.update_user("bob", is_admin=True)
        user = auth_store.get_user("bob")
        assert user.is_admin is True

        # Delete
        auth_store.delete_user("bob")
        assert auth_store.has_user("bob") is False
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_user("bob")

    def test_list_users(self, auth_store):
        """Create multiple users -> list -> verify count and names."""
        auth_store.create_user("user_a", "p1")
        auth_store.create_user("user_b", "p2")
        auth_store.create_user("user_c", "p3")

        users = auth_store.list_users()
        usernames = {u.username for u in users}
        assert usernames == {"user_a", "user_b", "user_c"}

    def test_experiment_permission_lifecycle(self, auth_store):
        """Create -> get -> update -> delete experiment permission."""
        auth_store.create_user("alice", "pass")

        # Create
        perm = auth_store.create_experiment_permission("exp-1", "alice", "READ")
        assert perm.experiment_id == "exp-1"
        assert perm.permission == "READ"

        # Get
        perm = auth_store.get_experiment_permission("exp-1", "alice")
        assert perm.permission == "READ"

        # Update
        auth_store.update_experiment_permission("exp-1", "alice", "MANAGE")
        perm = auth_store.get_experiment_permission("exp-1", "alice")
        assert perm.permission == "MANAGE"

        # List
        perms = auth_store.list_experiment_permissions("alice")
        assert len(perms) == 1
        assert perms[0].experiment_id == "exp-1"

        # Delete
        auth_store.delete_experiment_permission("exp-1", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_experiment_permission("exp-1", "alice")

    def test_experiment_permission_duplicate_raises(self, auth_store):
        """Creating the same experiment permission twice raises RESOURCE_ALREADY_EXISTS."""
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp-1", "alice", "READ")
        with pytest.raises(MlflowException, match="already exists") as exc_info:
            auth_store.create_experiment_permission("exp-1", "alice", "MANAGE")
        assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"

    def test_model_permission_lifecycle(self, auth_store):
        """Create -> get -> update -> delete model permission."""
        auth_store.create_user("alice", "pass")

        # Create
        perm = auth_store.create_registered_model_permission("my-model", "alice", "READ")
        assert perm.name == "my-model"
        assert perm.permission == "READ"

        # Get
        perm = auth_store.get_registered_model_permission("my-model", "alice")
        assert perm.permission == "READ"

        # Update
        auth_store.update_registered_model_permission("my-model", "alice", "MANAGE")
        perm = auth_store.get_registered_model_permission("my-model", "alice")
        assert perm.permission == "MANAGE"

        # List
        perms = auth_store.list_registered_model_permissions("alice")
        assert len(perms) == 1
        assert perms[0].name == "my-model"

        # Delete
        auth_store.delete_registered_model_permission("my-model", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_registered_model_permission("my-model", "alice")

    def test_model_permission_bulk_delete(self, auth_store):
        """Create perms for multiple users -> bulk delete -> verify all gone."""
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")

        auth_store.create_registered_model_permission("shared-model", "alice", "READ")
        auth_store.create_registered_model_permission("shared-model", "bob", "MANAGE")

        # Bulk delete all permissions for the model
        auth_store.delete_registered_model_permissions("shared-model")

        # Both should be gone
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_registered_model_permission("shared-model", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_registered_model_permission("shared-model", "bob")

    def test_model_permission_rename(self, auth_store):
        """Create perms -> rename_registered_model_permissions -> verify old gone, new exists."""
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")

        auth_store.create_registered_model_permission("old-model", "alice", "READ")
        auth_store.create_registered_model_permission("old-model", "bob", "MANAGE")

        # Rename
        auth_store.rename_registered_model_permissions("old-model", "new-model")

        # Old permissions are gone
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_registered_model_permission("old-model", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_registered_model_permission("old-model", "bob")

        # New permissions exist with same permission levels
        perm_a = auth_store.get_registered_model_permission("new-model", "alice")
        assert perm_a.permission == "READ"
        perm_b = auth_store.get_registered_model_permission("new-model", "bob")
        assert perm_b.permission == "MANAGE"

    def test_workspace_permission_lifecycle(self, auth_store):
        """Set -> get -> list -> delete workspace permission."""
        auth_store.create_user("alice", "pass")

        # Set (upsert)
        auth_store.set_workspace_permission("ws-1", "alice", "READ")
        perm = auth_store.get_workspace_permission("ws-1", "alice")
        assert perm.workspace == "ws-1"
        assert perm.permission == "READ"

        # Update via set
        auth_store.set_workspace_permission("ws-1", "alice", "MANAGE")
        perm = auth_store.get_workspace_permission("ws-1", "alice")
        assert perm.permission == "MANAGE"

        # List
        auth_store.create_user("bob", "pass")
        auth_store.set_workspace_permission("ws-1", "bob", "READ")
        perms = auth_store.list_workspace_permissions("ws-1")
        assert len(perms) == 2

        # Delete
        auth_store.delete_workspace_permission("ws-1", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_workspace_permission("ws-1", "alice")

    def test_list_accessible_workspace_names(self, auth_store):
        """Grant workspace perms -> list_accessible_workspace_names -> verify."""
        auth_store.create_user("alice", "pass")

        auth_store.set_workspace_permission("ws-alpha", "alice", "READ")
        auth_store.set_workspace_permission("ws-beta", "alice", "MANAGE")

        names = auth_store.list_accessible_workspace_names("alice")
        assert set(names) == {"ws-alpha", "ws-beta"}

    def test_scorer_permission_lifecycle(self, auth_store):
        """Create -> get -> update -> delete scorer permission."""
        auth_store.create_user("alice", "pass")

        # Create
        perm = auth_store.create_scorer_permission("exp-1", "accuracy", "alice", "READ")
        assert perm.experiment_id == "exp-1"
        assert perm.scorer_name == "accuracy"
        assert perm.permission == "READ"

        # Get
        perm = auth_store.get_scorer_permission("exp-1", "accuracy", "alice")
        assert perm.permission == "READ"

        # Update
        auth_store.update_scorer_permission("exp-1", "accuracy", "alice", "MANAGE")
        perm = auth_store.get_scorer_permission("exp-1", "accuracy", "alice")
        assert perm.permission == "MANAGE"

        # List
        perms = auth_store.list_scorer_permissions("alice")
        assert len(perms) == 1
        assert perms[0].scorer_name == "accuracy"

        # Delete
        auth_store.delete_scorer_permission("exp-1", "accuracy", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_scorer_permission("exp-1", "accuracy", "alice")

    def test_scorer_permission_bulk_delete(self, auth_store):
        """Create perms for scorer -> delete_scorer_permissions_for_scorer -> verify all gone."""
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")

        auth_store.create_scorer_permission("exp-1", "accuracy", "alice", "READ")
        auth_store.create_scorer_permission("exp-1", "accuracy", "bob", "MANAGE")

        # Bulk delete
        auth_store.delete_scorer_permissions_for_scorer("exp-1", "accuracy")

        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_scorer_permission("exp-1", "accuracy", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_scorer_permission("exp-1", "accuracy", "bob")

    def test_delete_user_cascades_all_permissions(self, auth_store):
        """Create user with all perm types -> delete user -> verify all gone."""
        auth_store.create_user("alice", "pass")

        # Grant various permissions
        auth_store.create_experiment_permission("exp-1", "alice", "READ")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        auth_store.set_workspace_permission("ws-1", "alice", "READ")
        auth_store.create_scorer_permission("exp-1", "accuracy", "alice", "READ")

        # Delete user (should cascade-delete all items in the user partition)
        auth_store.delete_user("alice")

        assert auth_store.has_user("alice") is False

        # All permissions should be gone
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_experiment_permission("exp-1", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_registered_model_permission("my-model", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_workspace_permission("ws-1", "alice")
        with pytest.raises(MlflowException, match="not found"):
            auth_store.get_scorer_permission("exp-1", "accuracy", "alice")
