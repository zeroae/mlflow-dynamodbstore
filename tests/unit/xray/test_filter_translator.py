"""Tests for X-Ray filter translator."""

from mlflow_dynamodbstore.dynamodb.search import FilterPredicate
from mlflow_dynamodbstore.xray.annotation_config import DEFAULT_ANNOTATION_CONFIG
from mlflow_dynamodbstore.xray.filter_translator import translate_span_filters


class TestFilterTranslator:
    def test_span_type_equals(self):
        xray_filter, remaining = translate_span_filters(
            [FilterPredicate("span_attribute", "mlflow.spanType", "=", "LLM")],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert 'annotation.mlflow_spanType = "LLM"' in xray_filter
        assert len(remaining) == 0

    def test_span_name_equals(self):
        xray_filter, remaining = translate_span_filters(
            [FilterPredicate("span_attribute", "name", "=", "ChatModel")],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert 'annotation.mlflow_spanName = "ChatModel"' in xray_filter

    def test_span_like_not_translatable(self):
        xray_filter, remaining = translate_span_filters(
            [
                FilterPredicate("span_attribute", "mlflow.spanType", "=", "LLM"),
                FilterPredicate("span_attribute", "name", "LIKE", "%chat%"),
            ],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert len(remaining) == 1  # LIKE stays as post-filter
        assert 'annotation.mlflow_spanType = "LLM"' in xray_filter

    def test_unmapped_attribute_stays_as_post_filter(self):
        xray_filter, remaining = translate_span_filters(
            [FilterPredicate("span_attribute", "unknown.attr", "=", "val")],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert len(remaining) == 1
        assert xray_filter == ""

    def test_multiple_translatable(self):
        xray_filter, remaining = translate_span_filters(
            [
                FilterPredicate("span_attribute", "mlflow.spanType", "=", "LLM"),
                FilterPredicate("span_attribute", "name", "=", "Chat"),
            ],
            DEFAULT_ANNOTATION_CONFIG,
        )
        assert "mlflow_spanType" in xray_filter
        assert "mlflow_spanName" in xray_filter
        assert "AND" in xray_filter

    def test_empty_input(self):
        xray_filter, remaining = translate_span_filters([], DEFAULT_ANNOTATION_CONFIG)
        assert xray_filter == ""
        assert remaining == []
