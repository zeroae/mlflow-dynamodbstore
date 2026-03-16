from __future__ import annotations

from mlflow_dynamodbstore.dynamodb.search import (
    FilterPredicate,
    parse_experiment_filter,
    parse_run_filter,
)


class TestRunFilterParser:
    def test_empty_filter(self):
        predicates = parse_run_filter("")
        assert predicates == []

    def test_attribute_equals(self):
        predicates = parse_run_filter("status = 'FINISHED'")
        assert len(predicates) == 1
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "status"
        assert predicates[0].op == "="
        assert predicates[0].value == "FINISHED"

    def test_metric_comparison(self):
        predicates = parse_run_filter("metric.accuracy > 0.9")
        assert predicates[0].field_type == "metric"
        assert predicates[0].key == "accuracy"
        assert predicates[0].op == ">"
        assert predicates[0].value == 0.9

    def test_param_like(self):
        predicates = parse_run_filter("param.model LIKE '%transformer%'")
        assert predicates[0].field_type == "param"
        assert predicates[0].op == "LIKE"

    def test_tag_equals(self):
        predicates = parse_run_filter("tag.env = 'prod'")
        assert predicates[0].field_type == "tag"

    def test_compound_and(self):
        predicates = parse_run_filter("metric.acc > 0.9 AND param.lr = '0.01'")
        assert len(predicates) == 2

    def test_dataset_filter(self):
        predicates = parse_run_filter("dataset.name = 'my_data'")
        assert predicates[0].field_type == "dataset"

    def test_run_name_like(self):
        predicates = parse_run_filter("run_name LIKE '%pipeline%'")
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "run_name"

    def test_metric_value_is_float(self):
        predicates = parse_run_filter("metric.loss < 0.5")
        assert isinstance(predicates[0].value, float)

    def test_none_filter(self):
        predicates = parse_run_filter(None)
        assert predicates == []

    def test_filter_predicate_immutable(self):
        predicates = parse_run_filter("status = 'FINISHED'")
        p = predicates[0]
        assert isinstance(p, FilterPredicate)
        # frozen dataclass - cannot modify
        import pytest

        with pytest.raises((AttributeError, TypeError)):
            p.key = "other"  # type: ignore[misc]


class TestRunQueryPlanner:
    def test_no_filter_uses_base_sk(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(predicates=[], order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi1"  # lifecycle filter
        assert plan.sk_prefix == "active#"

    def test_status_filter_uses_lsi3(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("status = 'FINISHED'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi3"
        assert plan.sk_prefix == "FINISHED#"

    def test_metric_order_uses_rank(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(
            predicates=[], order_by=["metric.accuracy DESC"], view_type=ViewType.ACTIVE_ONLY
        )
        assert plan.strategy == "rank"
        assert plan.rank_key == "accuracy"

    def test_dataset_filter_uses_dlink(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("dataset.name = 'my_data'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "dlink"

    def test_run_name_like_uses_fts(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("run_name LIKE '%pipeline%'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "fts"

    def test_tag_filter_denormalized_uses_filter_expression(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        preds = parse_run_filter("tag.mlflow.user = 'alice'")
        plan = plan_run_query(
            predicates=preds,
            order_by=None,
            view_type=ViewType.ACTIVE_ONLY,
            denormalized_patterns=["mlflow.*"],
        )
        assert any(fe for fe in plan.filter_expressions)

    def test_order_by_end_time_uses_lsi2(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(
            predicates=[], order_by=["end_time ASC"], view_type=ViewType.ACTIVE_ONLY
        )
        assert plan.index == "lsi2"

    def test_order_by_duration_uses_lsi5(self):
        from mlflow.entities import ViewType

        from mlflow_dynamodbstore.dynamodb.search import plan_run_query

        plan = plan_run_query(
            predicates=[], order_by=["duration DESC"], view_type=ViewType.ACTIVE_ONLY
        )
        assert plan.index == "lsi5"


class TestExperimentFilterParser:
    def test_name_like(self):
        predicates = parse_experiment_filter("name LIKE 'prod%'")
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "name"
        assert predicates[0].op == "LIKE"

    def test_tag_filter(self):
        predicates = parse_experiment_filter("tag.team = 'ml'")
        assert predicates[0].field_type == "tag"

    def test_empty_filter(self):
        predicates = parse_experiment_filter("")
        assert predicates == []

    def test_none_filter(self):
        predicates = parse_experiment_filter(None)
        assert predicates == []
