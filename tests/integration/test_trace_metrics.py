"""Integration tests for query_trace_metrics."""

import time

import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceLocationType, TraceState
from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.tracing.constant import AssessmentMetricKey, SpanMetricKey, TraceMetricKey


def _make_trace_info(experiment_id, trace_id, **overrides):
    defaults = dict(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=int(time.time() * 1000),
        execution_duration=500,
        state=TraceState.OK,
        trace_metadata={},
        tags={},
    )
    defaults.update(overrides)
    return TraceInfo(**defaults)


class _FakeSpan:
    def __init__(
        self,
        trace_id,
        span_id,
        name="span",
        span_type="CHAIN",
        status="OK",
        start_time_ns=1_000_000_000,
        end_time_ns=2_000_000_000,
        parent_id=None,
        attributes=None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.name = name
        self.span_type = span_type
        self.status = status
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.parent_id = parent_id
        self._attributes = attributes or {}

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "span_type": self.span_type,
            "status": self.status,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "parent_id": self.parent_id,
            "attributes": self._attributes,
        }


class TestQueryTraceMetricsIntegration:
    def test_traces_view_count(self, tracking_store):
        """Count traces across an experiment."""
        exp_id = tracking_store.create_experiment("integ-qtm-count")
        for i in range(3):
            tracking_store.start_trace(_make_trace_info(exp_id, f"tr-i-{i}"))
            tracking_store.log_spans(exp_id, [_FakeSpan(f"tr-i-{i}", f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0

    def test_traces_view_avg_latency(self, tracking_store):
        """Average latency across traces."""
        exp_id = tracking_store.create_experiment("integ-qtm-latency")
        for i, dur in enumerate([100, 200, 300]):
            tid = f"tr-lat-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid, execution_duration=dur))
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.LATENCY,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        )
        assert len(result) == 1
        assert result[0].values["AVG"] == pytest.approx(200.0)

    def test_spans_view_with_dimensions(self, tracking_store):
        """Count spans grouped by span_type."""
        exp_id = tracking_store.create_experiment("integ-qtm-spans")
        trace_id = "tr-integ-spans"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(trace_id, "s1", name="ChatModel", span_type="LLM"),
                _FakeSpan(trace_id, "s2", name="Retriever", span_type="RETRIEVER"),
                _FakeSpan(trace_id, "s3", name="Pipeline", span_type="CHAIN"),
            ],
        )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
            dimensions=["span_type"],
        )
        types = {dp.dimensions["span_type"]: dp.values["COUNT"] for dp in result}
        assert types["LLM"] == 1.0
        assert types["RETRIEVER"] == 1.0
        assert types["CHAIN"] == 1.0

    def test_assessments_view_count(self, tracking_store):
        """Count assessments across traces."""
        exp_id = tracking_store.create_experiment("integ-qtm-assess")
        trace_id = "tr-integ-assess"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [_FakeSpan(trace_id, "s1")])

        for name in ["quality", "relevance", "accuracy"]:
            tracking_store.create_assessment(
                Feedback(
                    name=name,
                    source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                    trace_id=trace_id,
                    value="good",
                )
            )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0

    def test_assessments_view_avg(self, tracking_store):
        """Average assessment values (yes/no -> 1.0/0.0)."""
        exp_id = tracking_store.create_experiment("integ-qtm-assess-avg")
        trace_id = "tr-integ-assess-avg"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(exp_id, [_FakeSpan(trace_id, "s1")])

        for val in ["yes", "no", "yes"]:
            tracking_store.create_assessment(
                Feedback(
                    name="approval",
                    source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                    trace_id=trace_id,
                    value=val,
                )
            )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.AVG)],
        )
        assert len(result) == 1
        # yes=1.0, no=0.0, yes=1.0 => avg = 2/3
        assert result[0].values["AVG"] == pytest.approx(2.0 / 3.0)

    def test_multi_experiment(self, tracking_store):
        """Query trace metrics across multiple experiments."""
        exp1 = tracking_store.create_experiment("integ-qtm-multi-1")
        exp2 = tracking_store.create_experiment("integ-qtm-multi-2")

        for exp_id, count in [(exp1, 2), (exp2, 3)]:
            for i in range(count):
                tid = f"tr-multi-{exp_id}-{i}"
                tracking_store.start_trace(_make_trace_info(exp_id, tid))
                tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s-{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp1, exp2],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[MetricAggregation(aggregation_type=AggregationType.COUNT)],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 5.0
