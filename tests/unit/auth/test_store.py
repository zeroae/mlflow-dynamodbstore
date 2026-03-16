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
