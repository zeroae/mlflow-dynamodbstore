from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from mlflow_dynamodbstore.dynamodb.config import ConfigReader
from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable


def _create_test_table(region: str = "us-east-1", table_name: str = "test") -> None:
    """Create a minimal DynamoDB table for testing."""
    ddb = boto3.client("dynamodb", region_name=region)
    ddb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def config_reader():
    with mock_aws():
        _create_test_table()
        table = DynamoDBTable(table_name="test", region="us-east-1")
        yield ConfigReader(table=table)


class TestConfigReader:
    def test_denormalize_patterns_default(self, config_reader):
        patterns = config_reader.get_denormalize_patterns()
        assert "mlflow.*" in patterns

    def test_denormalize_patterns_per_experiment(self, config_reader):
        # Write per-experiment config, verify merge
        config_reader.set_experiment_denormalize_patterns("01JQXYZ", ["team.*"])
        patterns = config_reader.get_effective_denormalize_patterns("01JQXYZ")
        assert "mlflow.*" in patterns  # global
        assert "team.*" in patterns  # experiment-specific

    def test_should_denormalize(self, config_reader):
        assert config_reader.should_denormalize(None, "mlflow.user") is True
        assert config_reader.should_denormalize(None, "custom_tag") is False

    def test_fts_trigram_fields_default(self, config_reader):
        fields = config_reader.get_fts_trigram_fields()
        assert isinstance(fields, list)

    def test_should_trigram_entity_names_always_true(self, config_reader):
        assert config_reader.should_trigram("experiment_name") is True
        assert config_reader.should_trigram("run_name") is True
        assert config_reader.should_trigram("model_name") is True

    def test_should_trigram_other_fields_configurable(self, config_reader):
        # Default: no extra trigram fields
        assert config_reader.should_trigram("run_param_value") is False

    def test_reconcile_from_env(self, config_reader, monkeypatch):
        monkeypatch.setenv("MLFLOW_DYNAMODB_DENORMALIZE_TAGS", "mlflow.*,env,team.*")
        config_reader.reconcile()
        patterns = config_reader.get_denormalize_patterns()
        assert "env" in patterns
        assert "team.*" in patterns
        assert "mlflow.*" in patterns

    def test_reconcile_preserves_mlflow_star(self, config_reader, monkeypatch):
        monkeypatch.setenv("MLFLOW_DYNAMODB_DENORMALIZE_TAGS", "env")
        config_reader.reconcile()
        patterns = config_reader.get_denormalize_patterns()
        assert "mlflow.*" in patterns  # always re-added


class TestTTLPolicy:
    def test_default_ttl_policy(self, config_reader):
        policy = config_reader.get_ttl_policy()
        assert policy["soft_deleted_retention_days"] == 90
        assert policy["trace_retention_days"] == 30
        assert policy["metric_history_retention_days"] == 365

    def test_set_ttl_policy(self, config_reader):
        config_reader.set_ttl_policy(soft_deleted_retention_days=60)
        policy = config_reader.get_ttl_policy()
        assert policy["soft_deleted_retention_days"] == 60

    def test_ttl_zero_means_disabled(self, config_reader):
        config_reader.set_ttl_policy(trace_retention_days=0)
        assert config_reader.get_trace_ttl_seconds() is None  # disabled

    def test_get_trace_ttl_seconds(self, config_reader):
        ttl = config_reader.get_trace_ttl_seconds()
        assert ttl == 30 * 86400  # 30 days in seconds

    def test_get_soft_deleted_ttl_seconds(self, config_reader):
        ttl = config_reader.get_soft_deleted_ttl_seconds()
        assert ttl == 90 * 86400

    def test_get_metric_history_ttl_seconds(self, config_reader):
        ttl = config_reader.get_metric_history_ttl_seconds()
        assert ttl == 365 * 86400

    def test_reconcile_ttl_from_env(self, config_reader, monkeypatch):
        monkeypatch.setenv("MLFLOW_DYNAMODB_SOFT_DELETED_RETENTION_DAYS", "60")
        config_reader.reconcile()
        policy = config_reader.get_ttl_policy()
        assert policy["soft_deleted_retention_days"] == 60
