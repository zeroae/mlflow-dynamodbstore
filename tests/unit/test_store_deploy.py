import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.uri import parse_dynamodb_uri


class TestDeployFlag:
    def test_deploy_false_parsed_from_uri(self):
        uri = parse_dynamodb_uri("dynamodb://us-east-1/my-table?deploy=false")
        assert uri.deploy is False

    @mock_aws
    def test_workspace_store_deploy_false_skips_provisioning(self):
        """With deploy=false, store init should not create a CFn stack."""
        from mlflow_dynamodbstore.workspace_store import DynamoDBWorkspaceStore

        with pytest.raises(Exception):
            DynamoDBWorkspaceStore(workspace_uri="dynamodb://us-east-1/nonexistent?deploy=false")
