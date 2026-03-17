"""Tests for logged model search parsing, planning, and execution."""

from mlflow_dynamodbstore.dynamodb.search import (
    FilterPredicate,
    parse_logged_model_filter,
    plan_logged_model_query,
)


class TestParseLoggedModelFilter:
    def test_empty_filter(self):
        assert parse_logged_model_filter(None) == []
        assert parse_logged_model_filter("") == []

    def test_attribute_filter(self):
        preds = parse_logged_model_filter("name = 'my-model'")
        assert len(preds) == 1
        assert preds[0].field_type == "attribute"
        assert preds[0].key == "name"
        assert preds[0].op == "="
        assert preds[0].value == "my-model"

    def test_metric_filter(self):
        preds = parse_logged_model_filter("metrics.accuracy > 0.5")
        assert len(preds) == 1
        assert preds[0].field_type == "metric"
        assert preds[0].key == "accuracy"

    def test_tag_filter(self):
        preds = parse_logged_model_filter("tags.env = 'prod'")
        assert len(preds) == 1
        assert preds[0].field_type == "tag"

    def test_param_filter(self):
        preds = parse_logged_model_filter("params.lr = '0.01'")
        assert len(preds) == 1
        assert preds[0].field_type == "param"


class TestPlanLoggedModelQuery:
    def test_default_plan(self):
        plan = plan_logged_model_query([], None, None)
        assert plan.strategy == "index"
        assert plan.index == "lsi1"
        assert plan.sk_prefix == "active#"

    def test_metric_order_by_uses_rank(self):
        plan = plan_logged_model_query(
            [], [{"field_name": "metrics.accuracy", "ascending": False}], None
        )
        assert plan.strategy == "rank"
        assert plan.rank_key == "accuracy"

    def test_status_filter_uses_lsi3(self):
        preds = [FilterPredicate("attribute", "status", "=", "READY")]
        plan = plan_logged_model_query(preds, None, None)
        assert plan.index == "lsi3"
        assert plan.sk_prefix == "READY#"

    def test_name_order_by_uses_lsi4(self):
        plan = plan_logged_model_query([], [{"field_name": "name", "ascending": True}], None)
        assert plan.index == "lsi4"

    def test_metric_filter_becomes_rank_filter(self):
        preds = [FilterPredicate("metric", "accuracy", ">", 0.5)]
        plan = plan_logged_model_query(preds, None, None)
        assert len(plan.rank_filters) == 1
        assert plan.rank_filters[0].key == "accuracy"
