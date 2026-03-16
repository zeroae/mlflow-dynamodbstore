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


class TestRegisteredModelPermissions:
    def test_create_registered_model_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        perm = auth_store.create_registered_model_permission("my-model", "alice", "READ")
        assert perm.name == "my-model"
        assert perm.permission == "READ"

    def test_create_duplicate_model_permission_raises(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        with pytest.raises(Exception):
            auth_store.create_registered_model_permission("my-model", "alice", "MANAGE")

    def test_get_registered_model_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        perm = auth_store.get_registered_model_permission("my-model", "alice")
        assert perm.permission == "READ"
        assert perm.name == "my-model"

    def test_get_registered_model_permission_not_found(self, auth_store):
        auth_store.create_user("alice", "pass")
        with pytest.raises(Exception):
            auth_store.get_registered_model_permission("my-model", "alice")

    def test_list_registered_model_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_registered_model_permission("model-a", "alice", "READ")
        auth_store.create_registered_model_permission("model-b", "alice", "MANAGE")
        perms = auth_store.list_registered_model_permissions("alice")
        assert len(perms) == 2
        names = {p.name for p in perms}
        assert names == {"model-a", "model-b"}

    def test_update_registered_model_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        auth_store.update_registered_model_permission("my-model", "alice", "MANAGE")
        perm = auth_store.get_registered_model_permission("my-model", "alice")
        assert perm.permission == "MANAGE"

    def test_delete_registered_model_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        auth_store.delete_registered_model_permission("my-model", "alice")
        with pytest.raises(Exception):
            auth_store.get_registered_model_permission("my-model", "alice")

    def test_delete_registered_model_permissions_bulk(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        auth_store.create_registered_model_permission("my-model", "bob", "MANAGE")
        auth_store.delete_registered_model_permissions("my-model")
        # Both should be gone
        with pytest.raises(Exception):
            auth_store.get_registered_model_permission("my-model", "alice")
        with pytest.raises(Exception):
            auth_store.get_registered_model_permission("my-model", "bob")

    def test_rename_registered_model_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")
        auth_store.create_registered_model_permission("old-model", "alice", "READ")
        auth_store.create_registered_model_permission("old-model", "bob", "MANAGE")
        auth_store.rename_registered_model_permissions("old-model", "new-model")
        # Old permissions should be gone
        with pytest.raises(Exception):
            auth_store.get_registered_model_permission("old-model", "alice")
        # New permissions should exist with same values
        perm_a = auth_store.get_registered_model_permission("new-model", "alice")
        assert perm_a.permission == "READ"
        perm_b = auth_store.get_registered_model_permission("new-model", "bob")
        assert perm_b.permission == "MANAGE"

    def test_delete_user_cascades_model_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_registered_model_permission("my-model", "alice", "READ")
        auth_store.delete_user("alice")
        perms = auth_store.list_registered_model_permissions("alice")
        assert len(perms) == 0


class TestWorkspacePermissions:
    def test_set_and_get_workspace_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        perm = auth_store.get_workspace_permission("ws1", "alice")
        assert perm.workspace == "ws1"
        assert perm.permission == "READ"

    def test_set_workspace_permission_upsert(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.set_workspace_permission("ws1", "alice", "MANAGE")
        perm = auth_store.get_workspace_permission("ws1", "alice")
        assert perm.permission == "MANAGE"

    def test_get_workspace_permission_not_found(self, auth_store):
        auth_store.create_user("alice", "pass")
        with pytest.raises(Exception):
            auth_store.get_workspace_permission("ws1", "alice")

    def test_list_workspace_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.set_workspace_permission("ws1", "bob", "MANAGE")
        perms = auth_store.list_workspace_permissions("ws1")
        assert len(perms) == 2

    def test_list_user_workspace_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.set_workspace_permission("ws2", "alice", "MANAGE")
        perms = auth_store.list_user_workspace_permissions("alice")
        assert len(perms) == 2
        workspaces = {p.workspace for p in perms}
        assert workspaces == {"ws1", "ws2"}

    def test_delete_workspace_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.delete_workspace_permission("ws1", "alice")
        with pytest.raises(Exception):
            auth_store.get_workspace_permission("ws1", "alice")

    def test_delete_workspace_permissions_for_workspace(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.set_workspace_permission("ws1", "bob", "MANAGE")
        auth_store.delete_workspace_permissions_for_workspace("ws1")
        with pytest.raises(Exception):
            auth_store.get_workspace_permission("ws1", "alice")
        with pytest.raises(Exception):
            auth_store.get_workspace_permission("ws1", "bob")

    def test_list_accessible_workspace_names(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.set_workspace_permission("ws2", "alice", "MANAGE")
        names = auth_store.list_accessible_workspace_names("alice")
        assert set(names) == {"ws1", "ws2"}

    def test_list_accessible_workspace_names_empty(self, auth_store):
        auth_store.create_user("alice", "pass")
        names = auth_store.list_accessible_workspace_names("alice")
        assert names == []

    def test_delete_user_cascades_workspace_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.set_workspace_permission("ws1", "alice", "READ")
        auth_store.delete_user("alice")
        names = auth_store.list_accessible_workspace_names("alice")
        assert names == []


class TestScorerPermissions:
    def test_create_scorer_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        perm = auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        assert perm.experiment_id == "exp1"
        assert perm.scorer_name == "accuracy"
        assert perm.permission == "READ"

    def test_create_duplicate_scorer_permission_raises(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        with pytest.raises(Exception):
            auth_store.create_scorer_permission("exp1", "accuracy", "alice", "MANAGE")

    def test_get_scorer_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        perm = auth_store.get_scorer_permission("exp1", "accuracy", "alice")
        assert perm.permission == "READ"
        assert perm.scorer_name == "accuracy"

    def test_get_scorer_permission_not_found(self, auth_store):
        auth_store.create_user("alice", "pass")
        with pytest.raises(Exception):
            auth_store.get_scorer_permission("exp1", "accuracy", "alice")

    def test_list_scorer_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        auth_store.create_scorer_permission("exp1", "f1_score", "alice", "MANAGE")
        perms = auth_store.list_scorer_permissions("alice")
        assert len(perms) == 2
        scorer_names = {p.scorer_name for p in perms}
        assert scorer_names == {"accuracy", "f1_score"}

    def test_update_scorer_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        auth_store.update_scorer_permission("exp1", "accuracy", "alice", "MANAGE")
        perm = auth_store.get_scorer_permission("exp1", "accuracy", "alice")
        assert perm.permission == "MANAGE"

    def test_delete_scorer_permission(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        auth_store.delete_scorer_permission("exp1", "accuracy", "alice")
        with pytest.raises(Exception):
            auth_store.get_scorer_permission("exp1", "accuracy", "alice")

    def test_delete_scorer_permissions_for_scorer(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_user("bob", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        auth_store.create_scorer_permission("exp1", "accuracy", "bob", "MANAGE")
        auth_store.delete_scorer_permissions_for_scorer("exp1", "accuracy")
        with pytest.raises(Exception):
            auth_store.get_scorer_permission("exp1", "accuracy", "alice")
        with pytest.raises(Exception):
            auth_store.get_scorer_permission("exp1", "accuracy", "bob")

    def test_delete_user_cascades_scorer_permissions(self, auth_store):
        auth_store.create_user("alice", "pass")
        auth_store.create_scorer_permission("exp1", "accuracy", "alice", "READ")
        auth_store.delete_user("alice")
        perms = auth_store.list_scorer_permissions("alice")
        assert len(perms) == 0


class TestGatewayPermissionStubs:
    """Gateway permission methods should raise NotImplementedError."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "create_gateway_secret_permission",
            "get_gateway_secret_permission",
            "list_gateway_secret_permissions",
            "update_gateway_secret_permission",
            "delete_gateway_secret_permission",
            "create_gateway_endpoint_permission",
            "get_gateway_endpoint_permission",
            "list_gateway_endpoint_permissions",
            "update_gateway_endpoint_permission",
            "delete_gateway_endpoint_permission",
            "create_gateway_model_definition_permission",
            "get_gateway_model_definition_permission",
            "list_gateway_model_definition_permissions",
            "update_gateway_model_definition_permission",
            "delete_gateway_model_definition_permission",
        ],
    )
    def test_gateway_stubs_raise(self, auth_store, method_name):
        method = getattr(auth_store, method_name)
        with pytest.raises(NotImplementedError):
            method()
