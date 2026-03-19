"""Unit tests for MetricAccumulator."""

import math

import pytest
from mlflow.entities.trace_metrics import AggregationType, MetricAggregation

from mlflow_dynamodbstore.trace_metrics.accumulators import MetricAccumulator


class TestMetricAccumulator:
    def test_count(self):
        acc = MetricAccumulator()
        for v in [1.0, 2.0, 3.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.COUNT)]
        result = acc.finalize(aggs)
        assert result["COUNT"] == 3

    def test_sum(self):
        acc = MetricAccumulator()
        for v in [1.0, 2.0, 3.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.SUM)]
        result = acc.finalize(aggs)
        assert result["SUM"] == 6.0

    def test_avg(self):
        acc = MetricAccumulator()
        for v in [1.0, 2.0, 3.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.AVG)]
        result = acc.finalize(aggs)
        assert result["AVG"] == 2.0

    def test_avg_empty(self):
        acc = MetricAccumulator()
        aggs = [MetricAggregation(AggregationType.AVG)]
        result = acc.finalize(aggs)
        assert result["AVG"] == 0.0

    def test_min_max(self):
        acc = MetricAccumulator()
        for v in [3.0, 1.0, 2.0]:
            acc.add(v)
        aggs = [
            MetricAggregation(AggregationType.MIN),
            MetricAggregation(AggregationType.MAX),
        ]
        result = acc.finalize(aggs)
        assert result["MIN"] == 1.0
        assert result["MAX"] == 3.0

    def test_percentile_median(self):
        acc = MetricAccumulator(collect_values=True)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            acc.add(v)
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0)]
        result = acc.finalize(aggs)
        assert result["P50.0"] == 3.0

    def test_percentile_p90(self):
        acc = MetricAccumulator(collect_values=True)
        for v in range(1, 101):
            acc.add(float(v))
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=90.0)]
        result = acc.finalize(aggs)
        assert result["P90.0"] == pytest.approx(90.1, rel=1e-3)

    def test_percentile_p0_and_p100(self):
        acc = MetricAccumulator(collect_values=True)
        for v in [10.0, 20.0, 30.0]:
            acc.add(v)
        aggs = [
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=0.0),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=100.0),
        ]
        result = acc.finalize(aggs)
        assert result["P0.0"] == 10.0
        assert result["P100.0"] == 30.0

    def test_percentile_single_value(self):
        acc = MetricAccumulator(collect_values=True)
        acc.add(42.0)
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0)]
        result = acc.finalize(aggs)
        assert result["P50.0"] == 42.0

    def test_percentile_without_collect_values(self):
        """Percentile on accumulator without collect_values returns NaN."""
        acc = MetricAccumulator(collect_values=False)
        acc.add(1.0)
        aggs = [MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0)]
        result = acc.finalize(aggs)
        assert math.isnan(result["P50.0"])

    def test_multiple_aggregations(self):
        acc = MetricAccumulator(collect_values=True)
        for v in [10.0, 20.0, 30.0]:
            acc.add(v)
        aggs = [
            MetricAggregation(AggregationType.COUNT),
            MetricAggregation(AggregationType.SUM),
            MetricAggregation(AggregationType.AVG),
            MetricAggregation(AggregationType.PERCENTILE, percentile_value=50.0),
        ]
        result = acc.finalize(aggs)
        assert result["COUNT"] == 3
        assert result["SUM"] == 60.0
        assert result["AVG"] == 20.0
        assert result["P50.0"] == 20.0

    def test_multiple_groups_independent(self):
        """Separate accumulators for different dimension groups stay independent."""
        groups: dict[tuple, MetricAccumulator] = {}
        for dim, val in [("a", 1.0), ("b", 10.0), ("a", 2.0), ("b", 20.0)]:
            key = (dim,)
            if key not in groups:
                groups[key] = MetricAccumulator()
            groups[key].add(val)

        aggs = [MetricAggregation(AggregationType.SUM)]
        assert groups[("a",)].finalize(aggs)["SUM"] == 3.0
        assert groups[("b",)].finalize(aggs)["SUM"] == 30.0
