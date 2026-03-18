"""Unit tests for trace metrics filter module."""

import pytest
from mlflow.entities.trace_metrics import MetricViewType

from mlflow_dynamodbstore.trace_metrics.extractors import (
    build_dimension_key,
    get_timestamp_for_view,
)
from mlflow_dynamodbstore.trace_metrics.filters import (
    apply_trace_metric_filters,
    filter_assessment_items,
    filter_span_items,
    meta_prefilter_spans,
)


class TestTraceMetricFilters:
    def test_empty_filters_passes(self):
        assert apply_trace_metric_filters({}, None, MetricViewType.TRACES) is True

    def test_trace_status_pass(self):
        assert (
            apply_trace_metric_filters(
                {"state": "OK"}, ["trace.status = 'OK'"], MetricViewType.TRACES
            )
            is True
        )

    def test_trace_status_fail(self):
        assert (
            apply_trace_metric_filters(
                {"state": "ERROR"}, ["trace.status = 'OK'"], MetricViewType.TRACES
            )
            is False
        )

    def test_span_filter_rejected_for_traces_view(self):
        with pytest.raises(Exception):
            apply_trace_metric_filters({}, ["span.name = 'X'"], MetricViewType.TRACES)

    def test_assessment_filter_rejected_for_spans_view(self):
        with pytest.raises(Exception):
            apply_trace_metric_filters({}, ["assessment.name = 'X'"], MetricViewType.SPANS)


class TestFilterSpanItems:
    def test_name_filter(self):
        items = [{"name": "A", "type": "LLM"}, {"name": "B", "type": "CHAIN"}]
        result = filter_span_items(items, ["span.name = 'A'"])
        assert len(result) == 1 and result[0]["name"] == "A"

    def test_type_filter(self):
        items = [{"name": "A", "type": "LLM"}, {"name": "B", "type": "CHAIN"}]
        result = filter_span_items(items, ["span.type = 'LLM'"])
        assert len(result) == 1

    def test_no_filters(self):
        items = [{"name": "A"}, {"name": "B"}]
        assert filter_span_items(items, None) == items


class TestFilterAssessmentItems:
    def test_name_filter(self):
        items = [
            {"name": "q", "assessment_type": "feedback"},
            {"name": "a", "assessment_type": "expectation"},
        ]
        result = filter_assessment_items(items, ["assessment.name = 'q'"])
        assert len(result) == 1

    def test_type_filter(self):
        items = [
            {"name": "q", "assessment_type": "feedback"},
            {"name": "a", "assessment_type": "expectation"},
        ]
        result = filter_assessment_items(items, ["assessment.type = 'feedback'"])
        assert len(result) == 1


class TestMetaPrefilterSpans:
    def test_passes_when_type_present(self):
        meta = {"span_types": {"LLM", "CHAIN"}}
        assert meta_prefilter_spans(meta, ["span.type = 'LLM'"]) is True

    def test_fails_when_type_absent(self):
        meta = {"span_types": {"CHAIN"}}
        assert meta_prefilter_spans(meta, ["span.type = 'LLM'"]) is False

    def test_no_filters_passes(self):
        assert meta_prefilter_spans({}, None) is True

    def test_unknown_entity_is_skipped(self):
        """Filter on an entity not in field_map is skipped (line 118 — continue branch)."""
        from unittest.mock import patch

        # Inject a pre-parsed span filter with an entity not in field_map
        fake_filters = [{"entity": "unknown_attr", "value": "foo"}]
        with patch(
            "mlflow_dynamodbstore.trace_metrics.filters._extract_span_filters",
            return_value=fake_filters,
        ):
            meta = {}
            # The unknown entity has no field_map entry => continue, returns True
            assert meta_prefilter_spans(meta, ["span.name = 'anything'"]) is True


class TestGetTimestampForView:
    """Tests for extractors.get_timestamp_for_view (lines 33-34 for SPANS)."""

    def test_traces_view_uses_request_time(self):
        meta = {"request_time": 1700000000000}
        ts = get_timestamp_for_view(MetricViewType.TRACES, {}, meta)
        assert ts == 1700000000000

    def test_spans_view_uses_start_time_ns(self):
        """SPANS view returns start_time_ns converted to ms (lines 33-34)."""
        item = {"start_time_ns": 1_700_000_000_000_000_000}  # 1.7e18 ns
        ts = get_timestamp_for_view(MetricViewType.SPANS, item, None)
        assert ts == 1_700_000_000_000  # 1.7e12 ms

    def test_assessments_view_uses_created_timestamp(self):
        item = {"created_timestamp": 1700000001234}
        ts = get_timestamp_for_view(MetricViewType.ASSESSMENTS, item, None)
        assert ts == 1700000001234

    def test_unknown_view_returns_zero(self):
        ts = get_timestamp_for_view(None, {}, None)  # type: ignore[arg-type]
        assert ts == 0


class TestBuildDimensionKeyAssessments:
    """Tests for extractors.build_dimension_key with ASSESSMENTS view (lines 78-86)."""

    def test_assessment_name_dimension(self):
        """assessment_name dimension returns item['name'] (line 79-80)."""
        item = {"name": "quality", "data": {}}
        key = build_dimension_key(["assessment_name"], MetricViewType.ASSESSMENTS, item)
        assert key == ("quality",)

    def test_assessment_value_dimension_feedback(self):
        """assessment_value returns feedback value (lines 82-85)."""
        item = {"name": "q", "data": {"feedback": {"value": "yes"}}}
        key = build_dimension_key(["assessment_value"], MetricViewType.ASSESSMENTS, item)
        assert key == ("yes",)

    def test_assessment_value_dimension_expectation(self):
        """assessment_value falls back to expectation value (lines 83-84)."""
        item = {"name": "q", "data": {"expectation": {"value": "expected"}}}
        key = build_dimension_key(["assessment_value"], MetricViewType.ASSESSMENTS, item)
        assert key == ("expected",)

    def test_assessment_value_dimension_none_when_missing(self):
        """assessment_value returns None when no feedback or expectation."""
        item = {"name": "q", "data": {}}
        key = build_dimension_key(["assessment_value"], MetricViewType.ASSESSMENTS, item)
        assert key == (None,)


class TestApplyTraceMetricFiltersTagAndMetadata:
    """Tests for apply_trace_metric_filters with tag/metadata filters (lines 28-35)."""

    def test_trace_tag_filter_passes(self):
        """trace.tag filter passes when tag exists in tag_items."""
        tag_items = [{"key": "env", "value": "prod"}]
        assert (
            apply_trace_metric_filters(
                {"state": "OK"},
                ["trace.tag.env = 'prod'"],
                MetricViewType.TRACES,
                tag_items=tag_items,
            )
            is True
        )

    def test_trace_tag_filter_fails_when_tag_absent(self):
        """trace.tag filter fails when tag is not present."""
        assert (
            apply_trace_metric_filters(
                {"state": "OK"},
                ["trace.tag.env = 'prod'"],
                MetricViewType.TRACES,
                tag_items=[],
            )
            is False
        )

    def test_trace_metadata_filter_passes(self):
        """trace.metadata filter passes when metadata key/value exists."""
        metadata_items = [{"key": "model", "value": "gpt-4"}]
        assert (
            apply_trace_metric_filters(
                {"state": "OK"},
                ["trace.metadata.model = 'gpt-4'"],
                MetricViewType.TRACES,
                metadata_items=metadata_items,
            )
            is True
        )

    def test_trace_metadata_filter_fails_when_absent(self):
        """trace.metadata filter fails when metadata is missing."""
        assert (
            apply_trace_metric_filters(
                {"state": "OK"},
                ["trace.metadata.model = 'gpt-4'"],
                MetricViewType.TRACES,
                metadata_items=[],
            )
            is False
        )
