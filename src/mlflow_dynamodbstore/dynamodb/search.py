from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils

# MLflow internally uses 'parameter' for the param field type; we normalize to 'param'.
_TYPE_MAP: dict[str, str] = {
    "parameter": "param",
}


@dataclass(frozen=True)
class FilterPredicate:
    """Normalized representation of a single MLflow filter clause."""

    field_type: str  # "attribute", "metric", "param", "tag", "dataset"
    key: str
    # "=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "NOT IN", "IS NULL", "IS NOT NULL"
    op: str
    value: Any


def _normalize_type(mlflow_type: str) -> str:
    """Map MLflow internal type names to our canonical field_type strings."""
    return _TYPE_MAP.get(mlflow_type, mlflow_type)


def _coerce_value(field_type: str, value: Any) -> Any:
    """Coerce string values to appropriate Python types based on field_type."""
    if field_type == "metric" and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value


def _to_predicates(parsed: list[dict[str, Any]]) -> list[FilterPredicate]:
    """Convert MLflow's list of parsed filter dicts into FilterPredicate objects."""
    result: list[FilterPredicate] = []
    for clause in parsed:
        field_type = _normalize_type(clause["type"])
        value = _coerce_value(field_type, clause.get("value"))
        result.append(
            FilterPredicate(
                field_type=field_type,
                key=clause["key"],
                op=clause["comparator"],
                value=value,
            )
        )
    return result


def parse_run_filter(filter_string: str | None) -> list[FilterPredicate]:
    """Parse an MLflow run search filter string into a list of FilterPredicates.

    Delegates actual parsing to :class:`mlflow.utils.search_utils.SearchUtils` and
    normalises the result into our :class:`FilterPredicate` dataclass.

    Args:
        filter_string: An MLflow filter expression such as
            ``"metric.acc > 0.9 AND tag.env = 'prod'"``
            or ``None`` / empty string for no filter.

    Returns:
        A (possibly empty) list of :class:`FilterPredicate` objects, one per AND clause.
    """
    if not filter_string:
        return []
    parsed = SearchUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
    return _to_predicates(parsed)


def parse_experiment_filter(filter_string: str | None) -> list[FilterPredicate]:
    """Parse an MLflow experiment search filter string into a list of FilterPredicates.

    Delegates actual parsing to :class:`mlflow.utils.search_utils.SearchExperimentsUtils`
    and normalises the result into our :class:`FilterPredicate` dataclass.

    Args:
        filter_string: An MLflow filter expression such as ``"name LIKE 'prod%'"``
            or ``None`` / empty string for no filter.

    Returns:
        A (possibly empty) list of :class:`FilterPredicate` objects, one per AND clause.
    """
    if not filter_string:
        return []
    parsed = SearchExperimentsUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
    return _to_predicates(parsed)
