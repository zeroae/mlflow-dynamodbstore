"""Tests for scorer CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import json

import pytest
from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType, ScorerVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.online.entities import OnlineScoringConfig

_GATEWAY_SCORER = json.dumps({"instructions_judge_pydantic_data": {"model": "gateway:/ep"}})


def _create_experiment(tracking_store) -> str:
    return tracking_store.create_experiment("scorer-test-exp")


def _ensure_gateway_endpoint(tracking_store, name: str = "ep") -> None:
    """Create a gateway endpoint (with backing secret + model def) if it doesn't exist."""
    try:
        tracking_store.get_gateway_endpoint(name=name)
    except MlflowException:
        secret = tracking_store.create_gateway_secret(
            secret_name=f"{name}-secret", secret_value={"api_key": "value"}
        )
        model_def = tracking_store.create_gateway_model_definition(
            name=f"{name}-model",
            secret_id=secret.secret_id,
            provider="openai",
            model_name="gpt-4",
        )
        tracking_store.create_gateway_endpoint(
            name=name,
            model_configs=[
                GatewayEndpointModelConfig(
                    model_definition_id=model_def.model_definition_id,
                    linkage_type=GatewayModelLinkageType.PRIMARY,
                    weight=1.0,
                ),
            ],
        )


class TestRegisterScorer:
    def test_register_first_version(self, tracking_store):
        """First registration creates META + Version, returns version 1."""
        exp_id = _create_experiment(tracking_store)
        result = tracking_store.register_scorer(exp_id, "accuracy", '{"name": "accuracy"}')
        assert isinstance(result, ScorerVersion)
        assert result.scorer_name == "accuracy"
        assert result.scorer_version == 1
        assert result.experiment_id == exp_id
        assert result.scorer_id is not None
        assert result._serialized_scorer == '{"name": "accuracy"}'

    def test_register_increments_version(self, tracking_store):
        """Second registration increments version to 2."""
        exp_id = _create_experiment(tracking_store)
        v1 = tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        v2 = tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        assert v1.scorer_version == 1
        assert v2.scorer_version == 2
        assert v1.scorer_id == v2.scorer_id  # same scorer_id

    def test_register_different_names_separate_scorers(self, tracking_store):
        """Different names create independent scorers."""
        exp_id = _create_experiment(tracking_store)
        s1 = tracking_store.register_scorer(exp_id, "accuracy", "{}")
        s2 = tracking_store.register_scorer(exp_id, "relevance", "{}")
        assert s1.scorer_id != s2.scorer_id
        assert s1.scorer_version == 1
        assert s2.scorer_version == 1

    def test_register_same_name_different_experiments(self, tracking_store):
        """Same name in different experiments creates separate scorers."""
        exp1 = tracking_store.create_experiment("exp1")
        exp2 = tracking_store.create_experiment("exp2")
        s1 = tracking_store.register_scorer(exp1, "accuracy", "{}")
        s2 = tracking_store.register_scorer(exp2, "accuracy", "{}")
        assert s1.scorer_id != s2.scorer_id

    def test_register_condition_prevents_duplicate_meta(self, tracking_store):
        """ConditionExpression on META put prevents duplicate scorer META items."""
        exp_id = _create_experiment(tracking_store)
        v1 = tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        v2 = tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        assert v1.scorer_id == v2.scorer_id
        assert v2.scorer_version == 2
        # Verify only one scorer in list (list_scorers not yet implemented,
        # so we check via direct query)
        pk = f"EXP#{exp_id}"
        items = tracking_store._table.query(pk=pk, sk_prefix="SCOR#")
        meta_items = [i for i in items if "#" not in i["SK"][len("SCOR#") :]]
        assert len(meta_items) == 1


class TestGetScorer:
    def test_get_latest_version(self, tracking_store):
        """get_scorer without version returns latest."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        result = tracking_store.get_scorer(exp_id, "accuracy")
        assert result.scorer_version == 2
        assert result._serialized_scorer == '{"v": 2}'

    def test_get_specific_version(self, tracking_store):
        """get_scorer with version returns that version."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        result = tracking_store.get_scorer(exp_id, "accuracy", version=1)
        assert result.scorer_version == 1
        assert result._serialized_scorer == '{"v": 1}'

    def test_get_nonexistent_raises(self, tracking_store):
        """get_scorer for missing scorer raises."""
        exp_id = _create_experiment(tracking_store)
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.get_scorer(exp_id, "no-such-scorer")

    def test_get_nonexistent_version_raises(self, tracking_store):
        """get_scorer for missing version raises."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", "{}")
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.get_scorer(exp_id, "accuracy", version=99)


class TestListScorers:
    def test_list_empty(self, tracking_store):
        """list_scorers returns empty list for experiment with no scorers."""
        exp_id = _create_experiment(tracking_store)
        result = tracking_store.list_scorers(exp_id)
        assert result == []

    def test_list_returns_latest_versions(self, tracking_store):
        """list_scorers returns latest version per scorer name."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        tracking_store.register_scorer(exp_id, "relevance", '{"v": 1}')
        result = tracking_store.list_scorers(exp_id)
        assert len(result) == 2
        by_name = {s.scorer_name: s for s in result}
        assert by_name["accuracy"].scorer_version == 2
        assert by_name["relevance"].scorer_version == 1


class TestListScorerVersions:
    def test_list_versions(self, tracking_store):
        """list_scorer_versions returns all versions ascending."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 3}')
        result = tracking_store.list_scorer_versions(exp_id, "accuracy")
        assert len(result) == 3
        assert [s.scorer_version for s in result] == [1, 2, 3]

    def test_list_versions_nonexistent_raises(self, tracking_store):
        """list_scorer_versions for missing scorer raises."""
        exp_id = _create_experiment(tracking_store)
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.list_scorer_versions(exp_id, "no-such")


class TestDeleteScorer:
    def test_delete_all_versions(self, tracking_store):
        """delete_scorer without version removes everything."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        tracking_store.delete_scorer(exp_id, "accuracy")
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.get_scorer(exp_id, "accuracy")

    def test_delete_single_version(self, tracking_store):
        """delete_scorer with version removes only that version."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        tracking_store.delete_scorer(exp_id, "accuracy", version=1)
        versions = tracking_store.list_scorer_versions(exp_id, "accuracy")
        assert len(versions) == 1
        assert versions[0].scorer_version == 2

    def test_delete_last_version_removes_meta(self, tracking_store):
        """Deleting the only version also removes META."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", "{}")
        tracking_store.delete_scorer(exp_id, "accuracy", version=1)
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.get_scorer(exp_id, "accuracy")

    def test_delete_latest_version_updates_cache(self, tracking_store):
        """After deleting latest version, get_scorer returns new latest."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 3}')
        tracking_store.delete_scorer(exp_id, "accuracy", version=3)
        result = tracking_store.get_scorer(exp_id, "accuracy")
        assert result.scorer_version == 2

    def test_delete_middle_version_creates_gap(self, tracking_store):
        """After deleting v2 from v1/v2/v3, new registration creates v4."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 1}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 2}')
        tracking_store.register_scorer(exp_id, "accuracy", '{"v": 3}')
        tracking_store.delete_scorer(exp_id, "accuracy", version=2)
        v4 = tracking_store.register_scorer(exp_id, "accuracy", '{"v": 4}')
        assert v4.scorer_version == 4
        versions = tracking_store.list_scorer_versions(exp_id, "accuracy")
        assert [v.scorer_version for v in versions] == [1, 3, 4]

    def test_delete_nonexistent_scorer_raises(self, tracking_store):
        """delete_scorer for missing scorer raises."""
        exp_id = _create_experiment(tracking_store)
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.delete_scorer(exp_id, "no-such")

    def test_delete_nonexistent_version_raises(self, tracking_store):
        """delete_scorer for missing version raises."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", "{}")
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.delete_scorer(exp_id, "accuracy", version=99)


class TestUpsertOnlineScoringConfig:
    @pytest.fixture(autouse=True)
    def _setup_gateway_endpoint(self, tracking_store):
        _ensure_gateway_endpoint(tracking_store)

    def test_create_config(self, tracking_store):
        """upsert creates a new config."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", _GATEWAY_SCORER)
        config = tracking_store.upsert_online_scoring_config(exp_id, "accuracy", sample_rate=0.5)
        assert isinstance(config, OnlineScoringConfig)
        assert config.sample_rate == 0.5
        assert config.filter_string is None

    def test_upsert_replaces_config(self, tracking_store):
        """upsert overwrites existing config atomically."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", _GATEWAY_SCORER)
        c1 = tracking_store.upsert_online_scoring_config(exp_id, "accuracy", sample_rate=0.5)
        c2 = tracking_store.upsert_online_scoring_config(
            exp_id, "accuracy", sample_rate=0.8, filter_string="status = 'OK'"
        )
        assert c2.sample_rate == 0.8
        assert c2.filter_string == "status = 'OK'"
        assert c1.online_scoring_config_id != c2.online_scoring_config_id

    def test_invalid_sample_rate_raises(self, tracking_store):
        """upsert rejects sample_rate outside [0.0, 1.0]."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", _GATEWAY_SCORER)
        with pytest.raises(MlflowException, match="sample_rate"):
            tracking_store.upsert_online_scoring_config(exp_id, "accuracy", 1.5)
        with pytest.raises(MlflowException, match="sample_rate"):
            tracking_store.upsert_online_scoring_config(exp_id, "accuracy", -0.1)

    def test_upsert_nonexistent_scorer_raises(self, tracking_store):
        """upsert for missing scorer raises."""
        exp_id = _create_experiment(tracking_store)
        with pytest.raises(MlflowException, match="not found"):
            tracking_store.upsert_online_scoring_config(exp_id, "no-such", 0.5)

    def test_config_with_filter_string(self, tracking_store):
        """upsert stores filter_string."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", _GATEWAY_SCORER)
        config = tracking_store.upsert_online_scoring_config(
            exp_id, "accuracy", sample_rate=0.1, filter_string="tag.env = 'prod'"
        )
        assert config.filter_string == "tag.env = 'prod'"


class TestGetOnlineScoringConfigs:
    @pytest.fixture(autouse=True)
    def _setup_gateway_endpoint(self, tracking_store):
        _ensure_gateway_endpoint(tracking_store)

    def test_get_configs(self, tracking_store):
        """get_online_scoring_configs returns configs for given scorer_ids."""
        exp_id = _create_experiment(tracking_store)
        s = tracking_store.register_scorer(exp_id, "accuracy", _GATEWAY_SCORER)
        tracking_store.upsert_online_scoring_config(exp_id, "accuracy", 0.5)
        configs = tracking_store.get_online_scoring_configs([s.scorer_id])
        assert len(configs) == 1
        assert configs[0].scorer_id == s.scorer_id
        assert configs[0].sample_rate == 0.5

    def test_get_configs_no_config(self, tracking_store):
        """get_online_scoring_configs returns empty for scorer without config."""
        exp_id = _create_experiment(tracking_store)
        s = tracking_store.register_scorer(exp_id, "accuracy", "{}")
        configs = tracking_store.get_online_scoring_configs([s.scorer_id])
        assert configs == []


class TestGetActiveOnlineScorers:
    @pytest.fixture(autouse=True)
    def _setup_gateway_endpoint(self, tracking_store):
        _ensure_gateway_endpoint(tracking_store)

    def test_active_scorers(self, tracking_store):
        """get_active_online_scorers returns scorers with sample_rate > 0."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", _GATEWAY_SCORER)
        tracking_store.register_scorer(exp_id, "relevance", _GATEWAY_SCORER)
        tracking_store.upsert_online_scoring_config(exp_id, "accuracy", 0.5)
        tracking_store.upsert_online_scoring_config(exp_id, "relevance", 0.0)
        active = tracking_store.get_active_online_scorers()
        assert len(active) == 1
        assert active[0].name == "accuracy"
        assert active[0].online_config.sample_rate == 0.5

    def test_no_active_scorers(self, tracking_store):
        """get_active_online_scorers returns empty when none active."""
        exp_id = _create_experiment(tracking_store)
        tracking_store.register_scorer(exp_id, "accuracy", "{}")
        active = tracking_store.get_active_online_scorers()
        assert active == []


class TestScorerVersionCompatToProto:
    def test_to_proto_skips_experiment_id(self, tracking_store):
        """to_proto works with ULID experiment_id (no int conversion)."""
        exp_id = _create_experiment(tracking_store)
        sv = tracking_store.register_scorer(exp_id, "accuracy", '{"name": "acc"}')
        proto = sv.to_proto()
        # experiment_id is int32 in proto, should be 0 (unset) since we skip it
        assert proto.experiment_id == 0
        assert proto.scorer_name == "accuracy"
        assert proto.scorer_version == 1
        assert proto.scorer_id == sv.scorer_id

    def test_to_proto_without_scorer_id(self, tracking_store):
        """to_proto handles None scorer_id."""
        from mlflow_dynamodbstore.tracking_store import _ScorerVersionCompat

        sv = _ScorerVersionCompat(
            experiment_id="test",
            scorer_name="acc",
            scorer_version=1,
            serialized_scorer="{}",
            creation_time=123,
            scorer_id=None,
        )
        proto = sv.to_proto()
        assert not proto.HasField("scorer_id")
