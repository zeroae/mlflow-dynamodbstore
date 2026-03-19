"""Phase 2a: MLflow workspace store tests run against DynamoDB.

MLflow's workspace tests use a `workspace_store` fixture.
Our conftest provides this backed by DynamoDB (moto).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "mlflow"))

from tests.store.workspace.test_sqlalchemy_store import (  # noqa: E402, F401
    test_create_workspace_duplicate_raises,
    test_create_workspace_invalid_name_raises,
    test_create_workspace_persists_record,
    test_delete_default_workspace_rejected,
    test_delete_workspace_removes_empty_workspace,
    test_delete_workspace_restrict_allows_empty_workspace,
    test_delete_workspace_restrict_blocks_when_resources_exist,
    test_get_workspace_not_found,
    test_get_workspace_success,
    test_list_workspaces_returns_all,
    test_update_workspace_can_clear_default_artifact_root,
    test_update_workspace_changes_description,
    test_update_workspace_sets_default_artifact_root,
)
