"""Tests for query_trace_metrics orchestrator method."""

import json
import time

import pytest
from mlflow.entities import TraceInfo, TraceLocation, TraceLocationType, TraceState
from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.entities.trace_location import MlflowExperimentLocation
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation, MetricViewType
from mlflow.tracing.constant import AssessmentMetricKey, SpanMetricKey, TraceMetricKey
from moto import mock_aws


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


def _make_trace_info(
    experiment_id, trace_id="tr-test", state=TraceState.OK, duration=500, request_time=None
):
    return TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            mlflow_experiment=MlflowExperimentLocation(experiment_id=experiment_id),
        ),
        request_time=request_time or int(time.time() * 1000),
        execution_duration=duration,
        state=state,
        trace_metadata={},
        tags={},
    )


def _count_agg():
    return MetricAggregation(aggregation_type=AggregationType.COUNT)


def _sum_agg():
    return MetricAggregation(aggregation_type=AggregationType.SUM)


def _avg_agg():
    return MetricAggregation(aggregation_type=AggregationType.AVG)


@mock_aws
class TestTraceMetricsQueryTraces:
    """Tests for TRACES view type."""

    def test_trace_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-trace-count")
        for i in range(3):
            tid = f"tr-count-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid))
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0

    def test_latency_avg(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-latency")
        for i, dur in enumerate([100, 200, 300]):
            tid = f"tr-lat-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid, duration=dur))
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.LATENCY,
            aggregations=[_avg_agg()],
        )
        assert len(result) == 1
        assert result[0].values["AVG"] == pytest.approx(200.0)

    def test_token_metrics_sum(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-tokens")
        for i in range(2):
            tid = f"tr-tok-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid))
            tracking_store.log_spans(
                exp_id,
                [
                    _FakeSpan(
                        tid,
                        f"s{i}",
                        attributes={
                            "mlflow.chat.tokenUsage": json.dumps(
                                {
                                    "input_tokens": 100,
                                    "output_tokens": 50,
                                    "total_tokens": 150,
                                }
                            ),
                        },
                    ),
                ],
            )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="total_tokens",
            aggregations=[_sum_agg()],
        )
        assert len(result) == 1
        assert result[0].values["SUM"] == pytest.approx(300.0)

    def test_status_dimension(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-status-dim")

        tid_ok = "tr-ok"
        tracking_store.start_trace(_make_trace_info(exp_id, tid_ok, state=TraceState.OK))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid_ok, "s1")])

        tid_err = "tr-err"
        tracking_store.start_trace(_make_trace_info(exp_id, tid_err, state=TraceState.ERROR))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid_err, "s2")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            dimensions=["trace_status"],
        )
        assert len(result) == 2
        by_status = {dp.dimensions["trace_status"]: dp for dp in result}
        assert by_status["OK"].values["COUNT"] == 1.0
        assert by_status["ERROR"].values["COUNT"] == 1.0

    def test_time_bucketing(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-time-bucket")
        base_time = 1700000000000  # ms

        for i in range(3):
            tid = f"tr-bucket-{i}"
            # All within same 1-hour bucket
            tracking_store.start_trace(
                _make_trace_info(exp_id, tid, request_time=base_time + i * 60_000)
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            time_interval_seconds=3600,
            start_time_ms=base_time,
            end_time_ms=base_time + 3600_000,
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0
        assert "time_bucket" in result[0].dimensions


@mock_aws
class TestTraceMetricsQuerySpans:
    """Tests for SPANS view type."""

    def test_span_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-span-count")
        tid = "tr-span-cnt"
        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(tid, "s1", name="embed"),
                _FakeSpan(tid, "s2", name="retrieve"),
                _FakeSpan(tid, "s3", name="generate"),
            ],
        )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[_count_agg()],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 3.0

    def test_span_latency(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-span-lat")
        tid = "tr-span-lat"
        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(tid, "s1", start_time_ns=0, end_time_ns=100_000_000),  # 100ms
                _FakeSpan(tid, "s2", start_time_ns=0, end_time_ns=300_000_000),  # 300ms
            ],
        )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.LATENCY,
            aggregations=[_avg_agg()],
        )
        assert len(result) == 1
        assert result[0].values["AVG"] == pytest.approx(200.0)

    def test_span_name_filter(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-span-filter")
        tid = "tr-span-filter"
        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(tid, "s1", name="embed", span_type="LLM"),
                _FakeSpan(tid, "s2", name="retrieve", span_type="RETRIEVER"),
            ],
        )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[_count_agg()],
            filters=["span.name = 'embed'"],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 1.0

    def test_span_type_dimension(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-span-type-dim")
        tid = "tr-span-type"
        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(tid, "s1", name="embed", span_type="LLM"),
                _FakeSpan(tid, "s2", name="retrieve", span_type="RETRIEVER"),
                _FakeSpan(tid, "s3", name="chat", span_type="LLM"),
            ],
        )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[_count_agg()],
            dimensions=["span_type"],
        )
        by_type = {dp.dimensions["span_type"]: dp for dp in result}
        assert by_type["LLM"].values["COUNT"] == 2.0
        assert by_type["RETRIEVER"].values["COUNT"] == 1.0


@mock_aws
class TestTraceMetricsQueryAssessments:
    """Tests for ASSESSMENTS view type."""

    def test_assessment_count(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-assess-count")
        tid = "tr-assess-cnt"
        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid, "s1")])

        for name in ["quality", "relevance"]:
            tracking_store.create_assessment(
                Feedback(
                    name=name,
                    source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                    trace_id=tid,
                    value="good",
                )
            )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
            aggregations=[_count_agg()],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 2.0

    def test_assessment_value_avg(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-assess-val")
        tid = "tr-assess-val"
        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid, "s1")])

        for val in ["yes", "no", "yes"]:
            tracking_store.create_assessment(
                Feedback(
                    name="approval",
                    source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                    trace_id=tid,
                    value=val,
                )
            )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name=AssessmentMetricKey.ASSESSMENT_VALUE,
            aggregations=[_avg_agg()],
        )
        assert len(result) == 1
        # yes=1.0, no=0.0, yes=1.0 => avg = 2/3
        assert result[0].values["AVG"] == pytest.approx(2.0 / 3.0)


@mock_aws
class TestTraceMetricsPagination:
    """Tests for pagination and caching."""

    def test_max_results_limits_output(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-pagination")
        for i in range(5):
            tid = f"tr-page-{i}"
            tracking_store.start_trace(
                _make_trace_info(
                    exp_id, tid, state=TraceState.OK if i % 2 == 0 else TraceState.ERROR
                )
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        # Query with status dimension => 2 groups (OK, ERROR)
        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            dimensions=["trace_status"],
            max_results=1,
        )
        assert len(result) == 1
        assert result.token is not None

    def test_cache_based_pagination(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-cache-page")
        for i in range(5):
            tid = f"tr-cache-{i}"
            tracking_store.start_trace(
                _make_trace_info(
                    exp_id, tid, state=TraceState.OK if i % 2 == 0 else TraceState.ERROR
                )
            )
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        # First page
        result1 = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            dimensions=["trace_status"],
            max_results=1,
        )
        assert len(result1) == 1
        assert result1.token is not None

        # Second page
        result2 = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            dimensions=["trace_status"],
            max_results=1,
            page_token=result1.token,
        )
        assert len(result2) == 1
        assert result2.token is None  # No more pages

        # Combined should have both statuses
        all_statuses = set()
        for dp in list(result1) + list(result2):
            all_statuses.add(dp.dimensions["trace_status"])
        assert len(all_statuses) == 2


@mock_aws
class TestTraceMetricsMultiExperiment:
    """Tests for querying across multiple experiments."""

    def test_merges_across_experiments(self, tracking_store):
        exp1 = tracking_store.create_experiment("qm-multi-1")
        exp2 = tracking_store.create_experiment("qm-multi-2")

        for exp_id, count in [(exp1, 2), (exp2, 3)]:
            for i in range(count):
                tid = f"tr-multi-{exp_id}-{i}"
                tracking_store.start_trace(_make_trace_info(exp_id, tid))
                tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp1, exp2],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 5.0


@mock_aws
class TestTraceMetricsValidation:
    """Tests for parameter validation."""

    def test_time_interval_without_start_end_raises(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-validation")

        with pytest.raises(Exception, match="start_time_ms and end_time_ms are required"):
            tracking_store.query_trace_metrics(
                experiment_ids=[exp_id],
                view_type=MetricViewType.TRACES,
                metric_name=TraceMetricKey.TRACE_COUNT,
                aggregations=[_count_agg()],
                time_interval_seconds=3600,
            )


@mock_aws
class TestTraceMetricsTraceStatusFilter:
    """Tests for trace status filtering."""

    def test_filter_by_status(self, tracking_store):
        exp_id = tracking_store.create_experiment("qm-status-filter")

        tid_ok = "tr-filt-ok"
        tracking_store.start_trace(_make_trace_info(exp_id, tid_ok, state=TraceState.OK))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid_ok, "s1")])

        tid_err = "tr-filt-err"
        tracking_store.start_trace(_make_trace_info(exp_id, tid_err, state=TraceState.ERROR))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid_err, "s2")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            filters=["trace.status = 'OK'"],
        )
        assert len(result) == 1
        assert result[0].values["COUNT"] == 1.0
