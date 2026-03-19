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

    def test_span_cost_sum(self, tracking_store):
        """SPANS view cost metrics from SMETRIC items."""
        import json

        exp_id = tracking_store.create_experiment("qtm-span-cost")
        trace_id = "tr-span-cost"
        tracking_store.start_trace(_make_trace_info(exp_id, trace_id))
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(
                    trace_id,
                    "s1",
                    attributes={
                        "mlflow.llm.cost": json.dumps(
                            {"input_cost": 0.01, "output_cost": 0.02, "total_cost": 0.03}
                        ),
                    },
                ),
                _FakeSpan(
                    trace_id,
                    "s2",
                    attributes={
                        "mlflow.llm.cost": json.dumps(
                            {"input_cost": 0.05, "output_cost": 0.10, "total_cost": 0.15}
                        ),
                    },
                ),
            ],
        )
        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name="total_cost",
            aggregations=[MetricAggregation(AggregationType.SUM)],
        )
        assert len(result) == 1
        assert result[0].values["SUM"] == pytest.approx(0.18, rel=1e-2)

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
class TestTraceMetricsTraceName:
    """Tests for TRACES view with trace_name dimension (line 4968)."""

    def test_trace_name_dimension(self, tracking_store):
        """Query with trace_name dimension triggers needs_tags=True (line 4968)."""
        exp_id = tracking_store.create_experiment("qm-trace-name")

        for i, tname in enumerate(["alpha", "beta"]):
            tid = f"tr-name-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid))
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])
            tracking_store.set_trace_tag(tid, "mlflow.traceName", tname)

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            dimensions=["trace_name"],
        )
        assert len(result) == 2
        by_name = {dp.dimensions["trace_name"]: dp for dp in result}
        assert by_name["alpha"].values["COUNT"] == 1.0
        assert by_name["beta"].values["COUNT"] == 1.0


@mock_aws
class TestTraceMetricsTokenNone:
    """Tests for TRACES view returning None for missing token metrics (line 5007)."""

    def test_input_tokens_without_token_usage_returns_empty(self, tracking_store):
        """Traces without token usage data should yield no data points for input_tokens."""
        exp_id = tracking_store.create_experiment("qm-no-tokens")
        for i in range(2):
            tid = f"tr-notok-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid))
            # Log spans without any token usage attributes
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name="input_tokens",
            aggregations=[_sum_agg()],
        )
        # No traces have token data => value is None => all skipped => empty result
        assert len(result) == 0


@mock_aws
class TestTraceMetricsSpansTimeBucket:
    """Tests for SPANS view with time bucketing (lines 5049-5050, 5073-5074)."""

    def test_spans_time_bucketing(self, tracking_store):
        """SPANS view with time_interval_seconds triggers time bucket path."""
        exp_id = tracking_store.create_experiment("qm-spans-timebucket")
        base_time_ms = 1700000000000
        base_time_ns = base_time_ms * 1_000_000
        tid = "tr-spans-tbucket"

        tracking_store.start_trace(_make_trace_info(exp_id, tid, request_time=base_time_ms))
        # Two spans in the same 1-hour bucket but different start times
        tracking_store.log_spans(
            exp_id,
            [
                _FakeSpan(
                    tid,
                    "s1",
                    start_time_ns=base_time_ns,
                    end_time_ns=base_time_ns + 100_000_000,
                ),
                _FakeSpan(
                    tid,
                    "s2",
                    start_time_ns=base_time_ns + 60_000_000_000,  # +60s
                    end_time_ns=base_time_ns + 160_000_000_000,
                ),
            ],
        )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.SPANS,
            metric_name=SpanMetricKey.SPAN_COUNT,
            aggregations=[_count_agg()],
            time_interval_seconds=3600,
            start_time_ms=base_time_ms,
            end_time_ms=base_time_ms + 3_600_000,
        )
        assert len(result) >= 1
        # All spans in the same bucket
        assert result[0].values["COUNT"] == 2.0
        assert "time_bucket" in result[0].dimensions


@mock_aws
class TestTraceMetricsAssessmentsTimeBucket:
    """Tests for ASSESSMENTS view with time bucketing (lines 5093-5094, 5111-5112, 5114-5115)."""

    def test_assessments_time_bucketing(self, tracking_store):
        """ASSESSMENTS view with time_interval_seconds triggers time bucket path."""
        exp_id = tracking_store.create_experiment("qm-assess-timebucket")
        base_time_ms = 1700000000000
        tid = "tr-assess-tbucket"

        tracking_store.start_trace(_make_trace_info(exp_id, tid, request_time=base_time_ms))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid, "s1")])

        tracking_store.create_assessment(
            Feedback(
                name="quality",
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
            time_interval_seconds=3600,
            start_time_ms=base_time_ms - 3_600_000,
            end_time_ms=base_time_ms + 3_600_000 * 24,
        )
        assert len(result) >= 1
        assert result[0].values["COUNT"] >= 1.0
        assert "time_bucket" in result[0].dimensions


@mock_aws
class TestTraceMetricsAssessmentNameDimension:
    """Tests for ASSESSMENTS view with assessment_name dimension (line 5104)."""

    def test_assessment_name_dimension(self, tracking_store):
        """Query ASSESSMENTS with assessment_name dimension groups by name."""
        exp_id = tracking_store.create_experiment("qm-assess-name-dim")
        tid = "tr-assess-namedim"

        tracking_store.start_trace(_make_trace_info(exp_id, tid))
        tracking_store.log_spans(exp_id, [_FakeSpan(tid, "s1")])

        for name in ["accuracy", "fluency", "accuracy"]:
            tracking_store.create_assessment(
                Feedback(
                    name=name,
                    source=AssessmentSource(source_type="HUMAN", source_id="u1"),
                    trace_id=tid,
                    value="yes",
                )
            )

        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.ASSESSMENTS,
            metric_name=AssessmentMetricKey.ASSESSMENT_COUNT,
            aggregations=[_count_agg()],
            dimensions=["assessment_name"],
        )
        assert len(result) == 2
        by_name = {dp.dimensions["assessment_name"]: dp for dp in result}
        assert by_name["accuracy"].values["COUNT"] == 2.0
        assert by_name["fluency"].values["COUNT"] == 1.0


@mock_aws
class TestTraceMetricsCacheHitPagination:
    """Tests for cache-hit pagination (line 4929)."""

    def test_cache_hit_pagination_generates_next_token(self, tracking_store):
        """When cached results span multiple pages, next_token is generated (line 4929)."""
        exp_id = tracking_store.create_experiment("qm-cache-hit-pag")
        base_time_ms = 1700000000000
        # Create traces in 4 different 1-hour buckets to get 4 data points
        for i in range(4):
            tid = f"tr-cachehit-{i}"
            t = base_time_ms + i * 3_600_000
            tracking_store.start_trace(_make_trace_info(exp_id, tid, request_time=t))
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        # First query: max_results=1 with 4 buckets => 4 data points, only 1 returned
        result1 = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            time_interval_seconds=3600,
            start_time_ms=base_time_ms,
            end_time_ms=base_time_ms + 4 * 3_600_000,
            max_results=1,
        )
        assert len(result1) == 1
        assert result1.token is not None

        # Second query with page_token => hits cache (line 4929 path when >1 page remains)
        result2 = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            time_interval_seconds=3600,
            start_time_ms=base_time_ms,
            end_time_ms=base_time_ms + 4 * 3_600_000,
            max_results=1,
            page_token=result1.token,
        )
        assert len(result2) == 1
        # More pages remain => next token should be set (exercises line 4929)
        assert result2.token is not None


@mock_aws
class TestTraceMetricsCacheMissPagination:
    """Tests for cache-miss pagination (lines 5147-5150)."""

    def test_cache_miss_uses_offset_from_token(self, tracking_store):
        """Fabricated page_token with non-zero offset that misses cache uses offset (5147-5150)."""

        from mlflow_dynamodbstore.trace_metrics.pagination import encode_page_token

        exp_id = tracking_store.create_experiment("qm-cache-miss-pag")
        for i in range(3):
            tid = f"tr-cachemiss-{i}"
            tracking_store.start_trace(_make_trace_info(exp_id, tid))
            tracking_store.log_spans(exp_id, [_FakeSpan(tid, f"s{i}")])

        # Build a fabricated page_token pointing to an unknown (expired) query_hash
        fake_token = encode_page_token("deadbeefdeadbeef", 1)

        # When cache misses, the store re-runs the full query then applies the offset
        result = tracking_store.query_trace_metrics(
            experiment_ids=[exp_id],
            view_type=MetricViewType.TRACES,
            metric_name=TraceMetricKey.TRACE_COUNT,
            aggregations=[_count_agg()],
            page_token=fake_token,
        )
        # offset=1 on 1-element result means empty page (all 3 traces => 1 data point, skip 1)
        assert len(result) == 0


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
