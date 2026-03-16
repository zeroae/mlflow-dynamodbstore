from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

from mlflow.entities import ViewType
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


@dataclass
class QueryPlan:
    """Describes the optimal DynamoDB execution strategy for a run search query."""

    strategy: str  # "index", "rank", "dlink", "fts"
    index: str | None  # LSI/GSI name
    sk_prefix: str | None  # begins_with condition on the sort key
    scan_forward: bool
    # DynamoDB FilterExpression clauses (evaluated server-side)
    filter_expressions: list[str] = field(default_factory=list)
    # Applied in Python after fetch (BatchGetItem for non-denormalized tags, params, metrics)
    post_filters: list[FilterPredicate] = field(default_factory=list)
    rank_key: str | None = None  # For strategy="rank"
    fts_query: str | None = None  # For strategy="fts"


# LSI mapping for order_by attributes
_ORDER_BY_LSI: dict[str, str] = {
    "end_time": "lsi2",
    "status": "lsi3",
    "run_name": "lsi4",
    "duration": "lsi5",
}

# Regex to detect both-sides wildcard LIKE pattern: '%word%'
_FTS_LIKE_RE = re.compile(r"^%[^%]+%$")


def _parse_order_by_token(token: str) -> tuple[str | None, str | None, bool]:
    """Parse a single order_by token into (field_type_or_none, key, scan_forward)."""
    token = token.strip()
    direction_upper = "ASC"
    # Split off trailing ASC/DESC
    parts = token.rsplit(None, 1)
    if len(parts) == 2 and parts[1].upper() in ("ASC", "DESC"):
        token_body = parts[0]
        direction_upper = parts[1].upper()
    else:
        token_body = token

    scan_forward = direction_upper == "ASC"

    if "." in token_body:
        field_type, key = token_body.split(".", 1)
        return field_type.lower(), key, scan_forward
    else:
        return None, token_body.lower(), scan_forward


def plan_run_query(
    predicates: list[FilterPredicate],
    order_by: list[str] | None,
    view_type: int,
    denormalized_patterns: list[str] | None = None,
) -> QueryPlan:
    """Analyze predicates and order_by to produce an optimal DynamoDB QueryPlan.

    Strategy selection priority:
    1. RANK  -- metric/param order_by
    2. DLINK -- dataset filter predicate
    3. FTS   -- LIKE '%word%' on run_name or other string attributes
    4. LSI   -- order_by maps to a specific LSI, or status= maps to LSI3
    5. Default -- lifecycle filter via LSI1

    Args:
        predicates: Parsed filter predicates from :func:`parse_run_filter`.
        order_by: List of order-by tokens (e.g. ``["metric.accuracy DESC"]``).
        view_type: MLflow ``ViewType`` constant (ACTIVE_ONLY, DELETED_ONLY, ALL).
        denormalized_patterns: Glob patterns for tag keys stored directly on the
            run item. Matching tags are turned into DynamoDB FilterExpressions
            instead of post-filters.

    Returns:
        A :class:`QueryPlan` describing the chosen execution strategy.
    """
    denormalized_patterns = denormalized_patterns or []

    # ------------------------------------------------------------------ #
    # 1. Check for RANK strategy (metric/param in order_by)               #
    # ------------------------------------------------------------------ #
    if order_by:
        for token in order_by:
            field_type, key, scan_forward = _parse_order_by_token(token)
            if field_type in ("metric", "param"):
                return QueryPlan(
                    strategy="rank",
                    index=None,
                    sk_prefix=None,
                    scan_forward=scan_forward,
                    rank_key=key,
                )

    # ------------------------------------------------------------------ #
    # 2. Check for DLINK strategy (dataset filter)                        #
    # ------------------------------------------------------------------ #
    for pred in predicates:
        if pred.field_type == "dataset":
            return QueryPlan(
                strategy="dlink",
                index=None,
                sk_prefix=None,
                scan_forward=True,
            )

    # ------------------------------------------------------------------ #
    # 3. Check for FTS strategy (LIKE '%word%' on attribute fields)       #
    # ------------------------------------------------------------------ #
    for pred in predicates:
        if pred.op in ("LIKE", "ILIKE") and isinstance(pred.value, str):
            if _FTS_LIKE_RE.match(pred.value):
                # Extract the word(s) for the FTS query (strip % characters)
                fts_query = pred.value.strip("%")
                return QueryPlan(
                    strategy="fts",
                    index=None,
                    sk_prefix=None,
                    scan_forward=True,
                    fts_query=fts_query,
                )

    # ------------------------------------------------------------------ #
    # 4. Determine index from order_by or status predicate                #
    # ------------------------------------------------------------------ #
    chosen_index: str | None = None
    scan_forward = False  # default: newest first (DESC)

    # Check order_by for LSI mapping
    if order_by:
        for token in order_by:
            field_type, key, sf = _parse_order_by_token(token)
            if field_type is None and key in _ORDER_BY_LSI:
                chosen_index = _ORDER_BY_LSI[key]
                scan_forward = sf
                break

    # Check for status= predicate (maps to LSI3)
    status_sk_prefix: str | None = None
    for pred in predicates:
        if pred.field_type == "attribute" and pred.key == "status" and pred.op == "=":
            chosen_index = "lsi3"
            status_sk_prefix = f"{pred.value}#"
            break

    # ------------------------------------------------------------------ #
    # 5. Default: lifecycle filter via LSI1                               #
    # ------------------------------------------------------------------ #
    if chosen_index is None:
        chosen_index = "lsi1"

    # Determine SK prefix for LSI1 lifecycle
    if chosen_index == "lsi1":
        if view_type == ViewType.DELETED_ONLY:
            sk_prefix = "deleted#"
        else:
            sk_prefix = "active#"
    elif chosen_index == "lsi3" and status_sk_prefix:
        sk_prefix = status_sk_prefix
    else:
        sk_prefix = None

    # ------------------------------------------------------------------ #
    # 6. Classify remaining predicates into FilterExpressions/post-filters#
    # ------------------------------------------------------------------ #
    filter_expressions: list[str] = []
    post_filters: list[FilterPredicate] = []

    for pred in predicates:
        # Skip the status predicate that already drives the SK prefix
        if (
            pred.field_type == "attribute"
            and pred.key == "status"
            and pred.op == "="
            and chosen_index == "lsi3"
        ):
            continue

        if pred.field_type == "tag":
            # Check if tag key matches any denormalized pattern
            is_denormalized = any(fnmatch.fnmatch(pred.key, pat) for pat in denormalized_patterns)
            if is_denormalized:
                # Generate a placeholder DynamoDB FilterExpression string
                safe_key = pred.key.replace(".", "_").replace("-", "_")
                filter_expressions.append(f"tags.#{safe_key} {pred.op} :{safe_key}v")
            else:
                post_filters.append(pred)
        elif pred.field_type in ("metric", "param"):
            post_filters.append(pred)
        elif pred.field_type == "attribute":
            # Non-status attribute predicates become post-filters
            post_filters.append(pred)
        # dataset predicates would already have triggered DLINK above

    return QueryPlan(
        strategy="index",
        index=chosen_index,
        sk_prefix=sk_prefix,
        scan_forward=scan_forward,
        filter_expressions=filter_expressions,
        post_filters=post_filters,
    )


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
