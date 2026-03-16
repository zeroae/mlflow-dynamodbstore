"""Translate MLflow span filter predicates into X-Ray filter expressions."""

from __future__ import annotations

from mlflow_dynamodbstore.dynamodb.search import FilterPredicate


def translate_span_filters(
    predicates: list[FilterPredicate],
    annotation_config: dict[str, str],
) -> tuple[str, list[FilterPredicate]]:
    """Translate span filter predicates to X-Ray filter expression.

    Only ``=`` comparisons on mapped annotations are translatable to X-Ray
    filter expressions. Everything else is returned as remaining predicates
    for post-filtering.

    Returns:
        (xray_filter_expression, remaining_predicates)
    """
    xray_parts: list[str] = []
    remaining: list[FilterPredicate] = []

    for pred in predicates:
        annotation_name = annotation_config.get(pred.key)
        if annotation_name and pred.op == "=":
            xray_parts.append(f'annotation.{annotation_name} = "{pred.value}"')
        else:
            remaining.append(pred)

    xray_filter = " AND ".join(xray_parts) if xray_parts else ""
    return xray_filter, remaining
