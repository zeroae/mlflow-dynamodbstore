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
