"""Tests for scorer CRUD in DynamoDBTrackingStore."""

from __future__ import annotations

import pytest
from mlflow.entities import ScorerVersion
from mlflow.exceptions import MlflowException


def _create_experiment(tracking_store) -> str:
    return tracking_store.create_experiment("scorer-test-exp")


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
