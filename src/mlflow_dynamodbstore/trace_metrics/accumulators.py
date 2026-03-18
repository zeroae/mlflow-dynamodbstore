"""Streaming metric accumulators with percentile support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mlflow.entities.trace_metrics import AggregationType, MetricAggregation


@dataclass
class MetricAccumulator:
    """O(1) streaming accumulator for metric aggregations.

    Tracks count, sum, min, max incrementally. Optionally collects raw values
    for percentile computation (linear interpolation matching percentile_cont).
    """

    collect_values: bool = False
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    _values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        if self.collect_values:
            self._values.append(value)

    def finalize(self, aggregations: list[MetricAggregation]) -> dict[str, float]:
        result: dict[str, float] = {}
        for agg in aggregations:
            match agg.aggregation_type:
                case AggregationType.COUNT:
                    result[str(agg)] = float(self.count)
                case AggregationType.SUM:
                    result[str(agg)] = self.sum
                case AggregationType.AVG:
                    result[str(agg)] = self.sum / self.count if self.count else 0.0
                case AggregationType.MIN:
                    result[str(agg)] = self.min if self.count else 0.0
                case AggregationType.MAX:
                    result[str(agg)] = self.max if self.count else 0.0
                case AggregationType.PERCENTILE:
                    result[str(agg)] = self._percentile(agg.percentile_value or 0.0)
        return result

    def _percentile(self, p: float) -> float:
        if not self._values:
            return float("nan")
        sorted_vals = sorted(self._values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        rank = (p / 100.0) * (n - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return sorted_vals[lower]
        fraction = rank - lower
        return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])
