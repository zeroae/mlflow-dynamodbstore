"""Filter parsing and application for trace metrics queries."""

from __future__ import annotations

from typing import Any

from mlflow.entities.trace_metrics import MetricViewType
from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchTraceMetricsUtils


def apply_trace_metric_filters(
    meta_item: dict[str, Any],
    filters: list[str] | None,
    view_type: MetricViewType,
    tag_items: list[dict[str, Any]] | None = None,
    metadata_items: list[dict[str, Any]] | None = None,
) -> bool:
    """Apply trace-level filters against a META item. Returns True if passes."""
    if not filters:
        return True
    for filter_string in filters:
        parsed = SearchTraceMetricsUtils.parse_search_filter(filter_string)
        if parsed.view_type == "trace":
            if parsed.entity == "status":
                if meta_item.get("state") != parsed.value:
                    return False
            elif parsed.entity == "tag":
                tags = tag_items or []
                if not any(t["key"] == parsed.key and t["value"] == parsed.value for t in tags):
                    return False
            elif parsed.entity == "metadata":
                metadata = metadata_items or []
                if not any(m["key"] == parsed.key and m["value"] == parsed.value for m in metadata):
                    return False
        elif parsed.view_type == "span":
            if view_type != MetricViewType.SPANS:
                raise MlflowException.invalid_parameter_value(  # type: ignore[no-untyped-call]
                    f"Filtering by span is only supported for {MetricViewType.SPANS} view type, "
                    f"got {view_type}",
                )
        elif parsed.view_type == "assessment":
            if view_type != MetricViewType.ASSESSMENTS:
                raise MlflowException.invalid_parameter_value(  # type: ignore[no-untyped-call]
                    f"Filtering by assessment is only supported for "
                    f"{MetricViewType.ASSESSMENTS} view type, got {view_type}",
                )
    return True


def _extract_span_filters(filters: list[str] | None) -> list[dict[str, Any]]:
    if not filters:
        return []
    result: list[dict[str, Any]] = []
    for f in filters:
        parsed = SearchTraceMetricsUtils.parse_search_filter(f)
        if parsed.view_type == "span":
            result.append({"entity": parsed.entity, "value": parsed.value})
    return result


def filter_span_items(
    span_items: list[dict[str, Any]], filters: list[str] | None
) -> list[dict[str, Any]]:
    """Filter individual span items by span-level filters (exact match)."""
    span_filters = _extract_span_filters(filters)
    if not span_filters:
        return span_items
    entity_to_field = {"name": "name", "status": "status", "type": "type"}
    return [
        item
        for item in span_items
        if all(
            item.get(entity_to_field.get(sf["entity"], sf["entity"])) == sf["value"]
            for sf in span_filters
        )
    ]


def _extract_assessment_filters(filters: list[str] | None) -> list[dict[str, Any]]:
    if not filters:
        return []
    result: list[dict[str, Any]] = []
    for f in filters:
        parsed = SearchTraceMetricsUtils.parse_search_filter(f)
        if parsed.view_type == "assessment":
            result.append({"entity": parsed.entity, "value": parsed.value})
    return result


def filter_assessment_items(
    assessment_items: list[dict[str, Any]], filters: list[str] | None
) -> list[dict[str, Any]]:
    """Filter assessment items by assessment-level filters (exact match)."""
    assess_filters = _extract_assessment_filters(filters)
    if not assess_filters:
        return assessment_items
    entity_to_field = {"name": "name", "type": "assessment_type"}
    return [
        item
        for item in assessment_items
        if all(
            item.get(entity_to_field.get(af["entity"], af["entity"])) == af["value"]
            for af in assess_filters
        )
    ]


def meta_prefilter_spans(meta_item: dict[str, Any], filters: list[str] | None) -> bool:
    """Fast pre-filter: check META denormalized sets before fetching span items."""
    span_filters = _extract_span_filters(filters)
    if not span_filters:
        return True
    field_map = {"name": "span_names", "type": "span_types", "status": "span_statuses"}
    for sf in span_filters:
        meta_field = field_map.get(sf["entity"])
        if not meta_field:
            continue
        values = meta_item.get(meta_field)
        if values is not None and sf["value"] not in values:
            return False
        # If values is None (sets not yet populated), fall through to per-span filtering
    return True
