"""View-specific data extraction from DynamoDB items."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mlflow.entities.trace_metrics import MetricViewType
from mlflow.tracing.constant import (
    AssessmentMetricKey,
    SpanMetricKey,
    TraceMetricDimensionKey,
    TraceMetricKey,
)

TIME_BUCKET_LABEL = "time_bucket"


def compute_time_bucket(timestamp_ms: int, interval_seconds: int) -> str:
    """Floor timestamp to nearest bucket and return ISO 8601 UTC string."""
    bucket_ms = interval_seconds * 1000
    bucket_start_ms = (timestamp_ms // bucket_ms) * bucket_ms
    dt = datetime.fromtimestamp(bucket_start_ms / 1000.0, tz=UTC)
    return dt.isoformat()


def get_timestamp_for_view(
    view_type: MetricViewType, item: dict[str, Any], meta_item: dict[str, Any] | None = None
) -> int:
    """Get relevant timestamp (ms) for time bucketing based on view type."""
    if view_type == MetricViewType.TRACES:
        return int(meta_item.get("request_time", 0) if meta_item else 0)
    elif view_type == MetricViewType.SPANS:
        return int(item.get("start_time_ns", 0)) // 1_000_000
    elif view_type == MetricViewType.ASSESSMENTS:
        return int(item.get("created_timestamp", 0))
    return 0


def build_dimension_key(
    dimensions: list[str] | None,
    view_type: MetricViewType,
    item: dict[str, Any],
    meta_item: dict[str, Any] | None = None,
    trace_tags: dict[str, str] | None = None,
    time_bucket: str | None = None,
) -> tuple[str | None, ...]:
    """Build dimension key tuple for grouping: (time_bucket, dim1, dim2, ...)."""
    parts: list[str | None] = []
    if time_bucket is not None:
        parts.append(time_bucket)
    for dim in dimensions or []:
        parts.append(_extract_dimension_value(dim, view_type, item, meta_item, trace_tags))
    return tuple(parts)


def _extract_dimension_value(
    dimension: str,
    view_type: MetricViewType,
    item: dict[str, Any],
    meta_item: dict[str, Any] | None = None,
    trace_tags: dict[str, str] | None = None,
) -> str | None:
    if view_type == MetricViewType.TRACES:
        if dimension == TraceMetricDimensionKey.TRACE_NAME:
            return (trace_tags or {}).get("mlflow.traceName")
        elif dimension == TraceMetricDimensionKey.TRACE_STATUS:
            return meta_item.get("state") if meta_item else None
    elif view_type == MetricViewType.SPANS:
        field_map = {
            "span_name": "name",
            "span_type": "type",
            "span_status": "status",
            "span_model_name": "model_name",
            "span_model_provider": "model_provider",
        }
        return item.get(field_map.get(dimension, dimension))
    elif view_type == MetricViewType.ASSESSMENTS:
        if dimension == "assessment_name":
            return item.get("name")
        elif dimension == "assessment_value":
            fb = item.get("data", {}).get("feedback", {})
            ex = item.get("data", {}).get("expectation", {})
            raw = fb.get("value") if fb else ex.get("value")
            return str(raw) if raw is not None else None
    return None


def extract_metric_value(
    metric_name: str,
    view_type: MetricViewType,
    item: dict[str, Any],
    meta_item: dict[str, Any] | None = None,
    trace_metric_items: list[dict[str, Any]] | None = None,
    span_metric_items: list[dict[str, Any]] | None = None,
) -> float | None:
    """Extract the metric value to aggregate from an item."""
    if view_type == MetricViewType.TRACES:
        if metric_name == TraceMetricKey.TRACE_COUNT:
            return 1.0
        elif metric_name == TraceMetricKey.LATENCY:
            return float(meta_item.get("execution_duration", 0)) if meta_item else None
        elif metric_name in TraceMetricKey.token_usage_keys():
            if trace_metric_items:
                for mi in trace_metric_items:
                    if mi.get("key") == metric_name:
                        return float(mi["value"])
            return None
    elif view_type == MetricViewType.SPANS:
        if metric_name == SpanMetricKey.SPAN_COUNT:
            return 1.0
        elif metric_name == SpanMetricKey.LATENCY:
            return float(item.get("duration_ms", 0))
        elif metric_name in SpanMetricKey.cost_keys():
            span_id = item.get("span_id") or item.get("SK", "").split("#SPAN#")[-1]
            if span_metric_items:
                for mi in span_metric_items:
                    if mi.get("span_id") == span_id and mi.get("key") == metric_name:
                        return float(mi["value"])
            return None
    elif view_type == MetricViewType.ASSESSMENTS:
        if metric_name == AssessmentMetricKey.ASSESSMENT_COUNT:
            return 1.0
        elif metric_name == AssessmentMetricKey.ASSESSMENT_VALUE:
            nv = item.get("numeric_value")
            return float(nv) if nv is not None else None
    return None
