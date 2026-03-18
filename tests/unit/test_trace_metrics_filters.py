"""Unit tests for trace metrics filter module."""

import pytest
from mlflow.entities.trace_metrics import MetricViewType

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
