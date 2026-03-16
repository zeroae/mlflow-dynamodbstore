import pytest


class TestAuthStoreUsers:
    def test_create_user(self, auth_store):
        user = auth_store.create_user("alice", "password123")
        assert user.username == "alice"
        assert user.is_admin is False

    def test_authenticate_user(self, auth_store):
        auth_store.create_user("alice", "password123")
        assert auth_store.authenticate_user("alice", "password123") is True
        assert auth_store.authenticate_user("alice", "wrong") is False

    def test_get_user(self, auth_store):
        auth_store.create_user("alice", "password123")
        user = auth_store.get_user("alice")
        assert user.username == "alice"

    def test_list_users(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.create_user("bob", "pass2")
        users = auth_store.list_users()
        assert len(users) == 2

    def test_update_user_password(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.update_user("alice", password="pass2")
        assert auth_store.authenticate_user("alice", "pass2") is True

    def test_update_user_admin(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.update_user("alice", is_admin=True)
        user = auth_store.get_user("alice")
        assert user.is_admin is True

    def test_delete_user(self, auth_store):
        auth_store.create_user("alice", "pass1")
        auth_store.delete_user("alice")
        assert auth_store.has_user("alice") is False

    def test_create_duplicate_user_raises(self, auth_store):
        auth_store.create_user("alice", "pass1")
        with pytest.raises(Exception):
            auth_store.create_user("alice", "pass2")

    def test_has_user(self, auth_store):
        assert auth_store.has_user("alice") is False
        auth_store.create_user("alice", "pass1")
        assert auth_store.has_user("alice") is True


class TestExperimentPermissions:
    def test_create_experiment_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        perm = auth_store.create_experiment_permission("exp1", "alice", "READ")
        assert perm.experiment_id == "exp1"
        assert perm.permission == "READ"

    def test_create_duplicate_permission_raises(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        with pytest.raises(Exception):
            auth_store.create_experiment_permission("exp1", "alice", "MANAGE")

    def test_get_experiment_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        perm = auth_store.get_experiment_permission("exp1", "alice")
        assert perm.permission == "READ"

    def test_get_experiment_permission_not_found(self, auth_store):
        auth_store.create_user("alice", "pass")
        with pytest.raises(Exception):
            auth_store.get_experiment_permission("exp1", "alice")

    def test_list_experiment_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        auth_store.create_experiment_permission("exp2", "alice", "MANAGE")
        perms = auth_store.list_experiment_permissions("alice")
        assert len(perms) == 2

    def test_update_experiment_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        auth_store.update_experiment_permission("exp1", "alice", "MANAGE")
        perm = auth_store.get_experiment_permission("exp1", "alice")
        assert perm.permission == "MANAGE"

    def test_delete_experiment_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        auth_store.delete_experiment_permission("exp1", "alice")
        with pytest.raises(Exception):
            auth_store.get_experiment_permission("exp1", "alice")

    def test_delete_user_cascades_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_experiment_permission("exp1", "alice", "READ")
        auth_store.delete_user("alice")
        assert auth_store.has_user("alice") is False
        # Permission should be gone too — listing should return empty
        perms = auth_store.list_experiment_permissions("alice")
        assert len(perms) == 0
