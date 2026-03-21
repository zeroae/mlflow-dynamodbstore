from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from boto3.dynamodb.conditions import Attr, ConditionBase
from mlflow.entities import ViewType
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchTraceUtils, SearchUtils

if TYPE_CHECKING:
    from mlflow_dynamodbstore.dynamodb.table import DynamoDBTable

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


def parse_trace_filter(filter_string: str | None) -> list[FilterPredicate]:
    """Parse an MLflow trace search filter string into a list of FilterPredicates.

    Delegates actual parsing to :class:`mlflow.utils.search_utils.SearchTraceUtils`
    and normalises the result into our :class:`FilterPredicate` dataclass.

    Args:
        filter_string: An MLflow trace filter expression such as
            ``"attribute.status = 'OK' AND tag.my_tag = 'hello'"``
            or ``None`` / empty string for no filter.

    Returns:
        A (possibly empty) list of :class:`FilterPredicate` objects, one per AND clause.
    """
    if not filter_string:
        return []
    parsed = SearchTraceUtils.parse_search_filter_for_search_traces(filter_string)  # type: ignore[no-untyped-call]
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
    # Boto3 FilterExpression (ConditionBase) for server-side filtering
    filter_expression: Any = None
    # Metric predicates applied after RANK/index fetch (logged model search)
    rank_filters: list[FilterPredicate] = field(default_factory=list)
    # Dataset scope for logged model metric search
    datasets: list[dict[str, Any]] | None = None


# LSI mapping for order_by attributes
_ORDER_BY_LSI: dict[str, str] = {
    "start_time": "lsi1",
    "created": "lsi1",  # MLflow alias for start_time
    "end_time": "lsi2",
    "status": "lsi3",
    "run_name": "lsi4",
    "duration": "lsi5",
}

# Regex to detect both-sides wildcard LIKE pattern: '%word%'
_FTS_LIKE_RE = re.compile(r"^%[^%]+%$")


# Normalize alternate field type identifiers to internal canonical form.
# MLflow clients may send plural forms (e.g. "metrics.score DESC").
# See mlflow.utils.search_utils._METRIC_IDENTIFIER and alternates.
_FIELD_TYPE_ALIASES: dict[str, str] = {
    "metrics": "metric",
    "parameter": "param",
    "parameters": "param",
    "params": "param",
    "tags": "tag",
    "attributes": "attribute",
    "attr": "attribute",
    "run": "attribute",
}


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
        field_type = field_type.lower()
        field_type = _FIELD_TYPE_ALIASES.get(field_type, field_type)
        key = key.lower()
        return field_type, key, scan_forward
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
                    # RANK items use inverted values, so flip scan direction
                    scan_forward=not scan_forward,
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
    # 3. Check for FTS strategy (LIKE '%word%' on FTS-indexed fields)     #
    # ------------------------------------------------------------------ #
    # Only use FTS for attribute fields that are actually trigram-indexed.
    # Non-indexed fields (params, metrics, tags) fall through to post-filter.
    run_fts_fields = {"attribute"}
    for pred in predicates:
        if pred.op in ("LIKE", "ILIKE") and isinstance(pred.value, str):
            if _FTS_LIKE_RE.match(pred.value) and pred.field_type in run_fts_fields:
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
            if field_type in (None, "attribute") and key in _ORDER_BY_LSI:
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

    # Determine SK prefix(es) for LSI1 lifecycle
    if chosen_index == "lsi1":
        if view_type == ViewType.DELETED_ONLY:
            sk_prefix = "deleted#"
        elif view_type == ViewType.ALL:
            sk_prefix = None
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


# Trace LSI mapping for order_by attributes
_TRACE_ORDER_BY_LSI: dict[str, str] = {
    "timestamp_ms": "lsi1",
    "end_time_ms": "lsi2",
    "execution_time_ms": "lsi5",
}


def plan_trace_query(
    predicates: list[FilterPredicate],
    order_by: list[str] | None,
) -> QueryPlan:
    """Analyze predicates and order_by to produce an optimal DynamoDB QueryPlan for traces.

    Strategy selection priority:
    1. FTS   -- LIKE '%word%' on tag/metadata text fields
    2. LSI3  -- status= filter (composite key: status#timestamp_ms)
    3. LSI4  -- name LIKE/ILIKE prefix
    4. LSI5  -- order_by execution_time_ms
    5. LSI2  -- order_by end_time_ms
    6. Default: LSI1 timestamp_ms (DESC)

    Key difference from run search: trace LSI1/2/5 values are NUMERIC, so
    begins_with (sk_prefix) is not used on them. LSI3 is STRING (status#ts),
    and LSI4 is STRING (lowercased name), so sk_prefix works for those.

    Args:
        predicates: Parsed filter predicates from :func:`parse_trace_filter`.
        order_by: List of order-by tokens (e.g. ``["execution_time_ms ASC"]``).

    Returns:
        A :class:`QueryPlan` describing the chosen execution strategy.
    """
    # ------------------------------------------------------------------ #
    # 1. Check for FTS strategy (LIKE '%word%' on FTS-indexed fields)     #
    # ------------------------------------------------------------------ #
    # Only use FTS for fields that are actually trigram-indexed.
    # Non-indexed fields (tags, metadata, name) fall through to post-filter.
    trace_fts_fields = {"run_name"}  # fields with FTS trigram index
    for pred in predicates:
        if pred.op in ("LIKE", "ILIKE") and isinstance(pred.value, str):
            if _FTS_LIKE_RE.match(pred.value) and pred.key in trace_fts_fields:
                fts_query = pred.value.strip("%")
                return QueryPlan(
                    strategy="fts",
                    index=None,
                    sk_prefix=None,
                    scan_forward=True,
                    fts_query=fts_query,
                )

    # ------------------------------------------------------------------ #
    # 2. Check for status= filter -> LSI3                                 #
    # ------------------------------------------------------------------ #
    status_sk_prefix: str | None = None
    for pred in predicates:
        if pred.field_type == "attribute" and pred.key == "status" and pred.op == "=":
            status_sk_prefix = f"{pred.value}#"
            remaining_preds = [
                p
                for p in predicates
                if not (p.field_type == "attribute" and p.key == "status" and p.op == "=")
            ]
            return QueryPlan(
                strategy="index",
                index="lsi3",
                sk_prefix=status_sk_prefix,
                scan_forward=False,  # newest first
                post_filters=remaining_preds,
            )

    # ------------------------------------------------------------------ #
    # 3. Check for name LIKE/ILIKE prefix -> LSI4                         #
    # ------------------------------------------------------------------ #
    for pred in predicates:
        if pred.field_type == "tag" and pred.key == "mlflow.traceName":
            if pred.op in ("LIKE", "ILIKE") and isinstance(pred.value, str):
                # Only handle prefix patterns: 'word%'
                # LSI4 stores lowercase name → narrows by ILIKE prefix
                # For case-sensitive LIKE, add FilterExpression on trace_name
                val = pred.value
                if val.endswith("%") and not val.startswith("%"):
                    prefix = val.rstrip("%").lower()
                    remaining_preds = [p for p in predicates if p is not pred]
                    # For LIKE (case-sensitive), use FilterExpression on trace_name
                    fe = None
                    if pred.op == "LIKE":
                        fe = Attr("trace_name").begins_with(val.rstrip("%"))
                    return QueryPlan(
                        strategy="index",
                        index="lsi4",
                        sk_prefix=prefix,
                        scan_forward=True,
                        post_filters=remaining_preds,
                        filter_expression=fe,
                    )

    # ------------------------------------------------------------------ #
    # 4. Determine index from order_by                                    #
    # ------------------------------------------------------------------ #
    chosen_index: str | None = None
    scan_forward = False  # default: newest first (DESC)

    if order_by:
        for token in order_by:
            field_type, key, sf = _parse_order_by_token(token)
            if key in _TRACE_ORDER_BY_LSI:
                chosen_index = _TRACE_ORDER_BY_LSI[key]
                scan_forward = sf
                break

    # ------------------------------------------------------------------ #
    # 5. Default: LSI1 (timestamp_ms DESC)                                #
    # ------------------------------------------------------------------ #
    if chosen_index is None:
        chosen_index = "lsi1"

    # Build FilterExpression for prompt predicates (server-side)
    post_filters: list[FilterPredicate] = []
    prompt_conditions: list[ConditionBase] = []
    for pred in predicates:
        if pred.key == "mlflow.linkedPrompts" and pred.op == "=":
            prompt_conditions.append(Attr(f"prompts.{pred.value}").exists())
        else:
            post_filters.append(pred)

    filter_expression: ConditionBase | None = None
    for cond in prompt_conditions:
        filter_expression = cond if filter_expression is None else filter_expression & cond

    return QueryPlan(
        strategy="index",
        index=chosen_index,
        sk_prefix=None,
        scan_forward=scan_forward,
        post_filters=post_filters,
        filter_expression=filter_expression,
    )


def _compare(actual: Any, op: str, expected: Any) -> bool:
    """Evaluate a single comparison predicate."""
    if actual is None:
        return op == "IS NULL"

    if op == "=":
        return bool(actual == expected)
    if op == "!=":
        return bool(actual != expected)
    if op == ">":
        return bool(actual > expected)
    if op == ">=":
        return bool(actual >= expected)
    if op == "<":
        return bool(actual < expected)
    if op == "<=":
        return bool(actual <= expected)
    if op == "LIKE":
        # Convert SQL LIKE to fnmatch: % -> * , _ -> ?
        pattern = str(expected).replace("%", "*").replace("_", "?")
        return fnmatch.fnmatch(str(actual), pattern)
    if op == "ILIKE":
        pattern = str(expected).lower().replace("%", "*").replace("_", "?")
        return fnmatch.fnmatch(str(actual).lower(), pattern)
    if op == "IN":
        return actual in expected
    if op == "NOT IN":
        return actual not in expected
    if op == "IS NULL":
        return actual is None
    if op == "IS NOT NULL":
        return actual is not None
    return False


_ATTRIBUTE_KEY_ALIASES: dict[str, str] = {
    "created": "start_time",
}


_NUMERIC_ATTRIBUTES = {"start_time", "end_time"}


def _apply_attribute_filter(item: dict[str, Any], pred: FilterPredicate) -> bool:
    """Check an attribute-level predicate against a META item."""
    key = _ATTRIBUTE_KEY_ALIASES.get(pred.key, pred.key)
    actual = item.get(key)
    value = pred.value
    # Coerce to comparable types for numeric attributes
    if key in _NUMERIC_ATTRIBUTES:
        if actual is not None:
            actual = int(actual)
        if isinstance(value, str):
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
    return _compare(actual, pred.op, value)


def _apply_denormalized_tag_filters(
    item: dict[str, Any],
    filter_expressions: list[str],
    predicates: list[FilterPredicate],
) -> bool:
    """Apply denormalized tag filter expressions as Python post-filters.

    Matches filter_expression entries against tag predicates to determine
    which tag key/value to check on the item's embedded ``tags`` map.
    """
    if not filter_expressions:
        return True

    tags = item.get("tags", {})
    # Build a lookup of denormalized tag predicates for matching
    tag_preds = [p for p in predicates if p.field_type == "tag"]

    for fe in filter_expressions:
        # filter_expressions look like: "tags.#mlflow_user = :mlflow_userv"
        # Find the matching tag predicate by checking if the safe key appears in the FE
        matched = False
        for pred in tag_preds:
            safe_key = pred.key.replace(".", "_").replace("-", "_")
            if f"#{safe_key}" in fe:
                actual_value = tags.get(pred.key)
                if not _compare(actual_value, pred.op, pred.value):
                    return False
                matched = True
                break
        if not matched:
            # If we can't find a matching predicate, skip this FE
            pass
    return True


def _apply_post_filter(
    table: DynamoDBTable,
    pk: str,
    run_id: str,
    item: dict[str, Any],
    pred: FilterPredicate,
) -> bool:
    """Apply a single post-filter predicate. May require sub-item lookups."""
    from mlflow_dynamodbstore.dynamodb.schema import (
        SK_METRIC_PREFIX,
        SK_PARAM_PREFIX,
        SK_RUN_PREFIX,
        SK_TAG_PREFIX,
    )

    if pred.field_type == "attribute":
        return _apply_attribute_filter(item, pred)

    # For tag, param, metric: look up the sub-item
    if pred.field_type == "tag":
        sk = f"{SK_RUN_PREFIX}{run_id}{SK_TAG_PREFIX}{pred.key}"
    elif pred.field_type == "param":
        sk = f"{SK_RUN_PREFIX}{run_id}{SK_PARAM_PREFIX}{pred.key}"
    elif pred.field_type == "metric":
        sk = f"{SK_RUN_PREFIX}{run_id}{SK_METRIC_PREFIX}{pred.key}"
    else:
        return True  # Unknown type, don't filter

    sub_item = table.get_item(pk, sk)
    if sub_item is None:
        # No sub-item means null value
        return _compare(None, pred.op, pred.value)

    actual = sub_item.get("value")
    # For metrics, coerce to float for comparison
    if pred.field_type == "metric" and actual is not None:
        try:
            actual = float(actual)
        except (ValueError, TypeError):
            pass
    return _compare(actual, pred.op, pred.value)


def _fetch_meta_items(
    table: DynamoDBTable,
    pk: str,
    run_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch run META items by run_id, preserving order."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_RUN_PREFIX

    items: list[dict[str, Any]] = []
    for run_id in run_ids:
        item = table.get_item(pk, f"{SK_RUN_PREFIX}{run_id}")
        if item is not None:
            items.append(item)
    return items


def execute_query(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
    max_results: int,
    page_token: str | None = None,
    predicates: list[FilterPredicate] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute a DynamoDB query based on a QueryPlan.

    Args:
        table: DynamoDBTable instance.
        plan: The QueryPlan describing the execution strategy.
        pk: The partition key (e.g. ``"EXP#01JQ"``).
        max_results: Maximum number of run META items to return.
        page_token: Opaque pagination token from a previous call.
        predicates: Full list of filter predicates (used for denormalized
            tag matching against filter_expressions).

    Returns:
        A tuple ``(items, next_page_token)`` where *items* are run META
        item dicts and *next_page_token* is ``None`` when there are no
        more results.
    """
    from mlflow_dynamodbstore.dynamodb.pagination import (
        decode_page_token,
        encode_page_token,
    )

    predicates = predicates or []
    token_data = decode_page_token(page_token)
    offset = token_data.get("offset", 0) if token_data else 0

    if plan.strategy == "index":
        items = _execute_index(table, plan, pk, offset, max_results, predicates)
    elif plan.strategy == "rank":
        items = _execute_rank(table, plan, pk)
    elif plan.strategy == "dlink":
        items = _execute_dlink(table, plan, pk, predicates)
    elif plan.strategy == "fts":
        items = _execute_fts(table, plan, pk)
    else:
        items = []

    # Apply post-filters
    filtered: list[dict[str, Any]] = []
    for item in items:
        run_id = item.get("run_id", "")
        if all(_apply_post_filter(table, pk, run_id, item, pred) for pred in plan.post_filters):
            filtered.append(item)

    # Pagination: simple offset-based
    page = filtered[offset : offset + max_results]
    has_more = len(filtered) > offset + max_results
    next_token: str | None = None
    if has_more:
        next_token = encode_page_token({"offset": offset + max_results})

    return page, next_token


def _execute_index(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
    offset: int,
    max_results: int,
    predicates: list[FilterPredicate],
) -> list[dict[str, Any]]:
    """Execute an index-based query strategy."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_RUN_PREFIX

    # Fetch more than needed to account for filtering
    has_filters = bool(plan.filter_expressions) or plan.sk_prefix is None
    fetch_limit = offset + max_results * 2 if has_filters else None

    # When sk_prefix is None (ViewType.ALL), add FilterExpression to limit
    # results to run META items only (avoid scanning traces, tags, etc.)
    filter_expr = None
    if plan.sk_prefix is None:
        filter_expr = Attr("SK").begins_with(SK_RUN_PREFIX)

    items = table.query(
        pk=pk,
        sk_prefix=plan.sk_prefix,
        index_name=plan.index,
        scan_forward=plan.scan_forward,
        limit=fetch_limit,
        filter_expression=filter_expr,
    )

    # Apply denormalized tag filters
    if plan.filter_expressions:
        items = [
            item
            for item in items
            if _apply_denormalized_tag_filters(item, plan.filter_expressions, predicates)
        ]

    # For DESC on LSI2 (end_time), move null-sentinel items to the end
    # so that nulls sort last regardless of direction.
    if plan.index == "lsi2" and not plan.scan_forward:
        from mlflow_dynamodbstore.dynamodb.schema import LSI2_NULL_SENTINEL, LSI2_SK

        real = [it for it in items if it.get(LSI2_SK) != LSI2_NULL_SENTINEL]
        nulls = [it for it in items if it.get(LSI2_SK) == LSI2_NULL_SENTINEL]
        items = real + nulls

    return items


def _execute_rank(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute a rank-based query strategy."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_RANK_PREFIX

    # Query RANK items: RANK#m#<key># for metrics, RANK#p#<key># for params
    # Try metrics first, then params
    for type_prefix in ("m", "p"):
        sk_prefix = f"{SK_RANK_PREFIX}{type_prefix}#{plan.rank_key}#"
        rank_items = table.query(
            pk=pk,
            sk_prefix=sk_prefix,
            scan_forward=plan.scan_forward,
        )
        if rank_items:
            break
    else:
        return []

    # Extract run_ids from RANK items (preserving order)
    run_ids = [item["run_id"] for item in rank_items if "run_id" in item]
    return _fetch_meta_items(table, pk, run_ids)


def _execute_dlink(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
    predicates: list[FilterPredicate],
) -> list[dict[str, Any]]:
    """Execute a dataset-link query strategy."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_DLINK_PREFIX

    # Find the dataset predicate to build the SK prefix
    ds_name = None
    ds_digest = None
    for pred in predicates:
        if pred.field_type == "dataset":
            if pred.key == "name":
                ds_name = pred.value
            elif pred.key == "digest":
                ds_digest = pred.value

    # Build SK prefix from available dataset info
    if ds_name and ds_digest:
        sk_prefix = f"{SK_DLINK_PREFIX}{ds_name}#{ds_digest}#"
    elif ds_name:
        sk_prefix = f"{SK_DLINK_PREFIX}{ds_name}#"
    else:
        sk_prefix = SK_DLINK_PREFIX

    dlink_items = table.query(pk=pk, sk_prefix=sk_prefix, scan_forward=plan.scan_forward)

    # Extract run_ids from DLINK items
    run_ids = [item.get("run_id", "") for item in dlink_items if item.get("run_id")]
    return _fetch_meta_items(table, pk, run_ids)


def _execute_fts(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute a full-text search query strategy."""
    from mlflow_dynamodbstore.dynamodb.fts import tokenize_trigrams, tokenize_words
    from mlflow_dynamodbstore.dynamodb.schema import SK_FTS_PREFIX

    if not plan.fts_query:
        return []

    # Tokenize with word stemmer
    word_tokens = tokenize_words(plan.fts_query)

    if word_tokens:
        # For each word token, query FTS items and collect entity IDs
        entity_id_sets: list[set[str]] = []
        for token in word_tokens:
            sk_prefix = f"{SK_FTS_PREFIX}W#R#{token}#"
            fts_items = table.query(pk=pk, sk_prefix=sk_prefix)
            # Extract entity_id (run_id) from the SK pattern:
            # FTS#W#R#<token>#<run_id>
            ids = set()
            for item in fts_items:
                sk = item.get("SK", "")
                # Parse run_id from SK: FTS#W#R#<token>#<run_id>[#<field>]
                parts = sk.split("#")
                # parts[0]=FTS, parts[1]=W, parts[2]=R, parts[3]=token, parts[4]=run_id
                if len(parts) >= 5:
                    ids.add(parts[4])
            entity_id_sets.append(ids)

        if not entity_id_sets:
            return []

        # AND semantics: intersect all sets
        result_ids = entity_id_sets[0]
        for s in entity_id_sets[1:]:
            result_ids &= s

        if result_ids:
            return _fetch_meta_items(table, pk, list(result_ids))

    # Fallback to trigrams if no word matches
    trigram_tokens = tokenize_trigrams(plan.fts_query)
    if trigram_tokens:
        entity_id_sets = []
        for token in trigram_tokens:
            sk_prefix = f"{SK_FTS_PREFIX}3#R#{token}#"
            fts_items = table.query(pk=pk, sk_prefix=sk_prefix)
            ids = set()
            for item in fts_items:
                sk = item.get("SK", "")
                # Parse run_id from SK: FTS#3#R#<token>#<run_id>[#<field>]
                parts = sk.split("#")
                # parts[0]=FTS, parts[1]=3, parts[2]=R, parts[3]=token, parts[4]=run_id
                if len(parts) >= 5:
                    ids.add(parts[4])
            entity_id_sets.append(ids)

        if not entity_id_sets:
            return []

        result_ids = entity_id_sets[0]
        for s in entity_id_sets[1:]:
            result_ids &= s

        if result_ids:
            return _fetch_meta_items(table, pk, list(result_ids))

    return []


def _is_trace_meta_item(item: dict[str, Any]) -> bool:
    """Check if a DynamoDB item is a trace META item (not a sub-item or run item)."""
    return "trace_id" in item


def _apply_trace_post_filter(
    table: DynamoDBTable,
    pk: str,
    trace_id: str,
    item: dict[str, Any],
    pred: FilterPredicate,
) -> bool:
    """Apply a single post-filter predicate for traces. May require sub-item lookups."""
    from mlflow_dynamodbstore.dynamodb.schema import SK_TRACE_PREFIX

    if pred.field_type == "attribute":
        # Map attribute keys to item fields
        key_map = {
            "timestamp_ms": "request_time",
            "end_time_ms": "lsi2sk",
            "execution_time_ms": "execution_duration",
            "status": "state",
            "name": "trace_name",
            "client_request_id": "client_request_id",
        }
        item_key = key_map.get(pred.key, pred.key)
        actual = item.get(item_key)
        # Coerce numeric comparisons
        if pred.key in ("timestamp_ms", "execution_time_ms", "end_time_ms"):
            if actual is not None:
                actual = int(actual)
        # For != on optional attributes, None means "not present" → exclude
        if actual is None and pred.op == "!=":
            return False
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "tag":
        # For trace name, use the trace_name attribute on META
        if pred.key == "mlflow.traceName":
            actual = item.get("trace_name")
            return _compare(actual, pred.op, pred.value)
        # Prompt filter: handled by FilterExpression via denormalized prompts map
        if pred.key == "mlflow.linkedPrompts":
            prompts = item.get("prompts", {})
            return pred.value in prompts
        # Check denormalized tags first
        tags = item.get("tags", {})
        if pred.key in tags:
            return _compare(tags[pred.key], pred.op, pred.value)
        # Fall back to sub-item lookup
        sk = f"{SK_TRACE_PREFIX}{trace_id}#TAG#{pred.key}"
        sub_item = table.get_item(pk, sk)
        actual = sub_item["value"] if sub_item else None
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "request_metadata":
        sk = f"{SK_TRACE_PREFIX}{trace_id}#RMETA#{pred.key}"
        sub_item = table.get_item(pk, sk)
        actual = sub_item["value"] if sub_item else None
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "feedback":
        feedbacks = item.get("feedbacks", {})
        actual = feedbacks.get(pred.key)
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "expectation":
        expectations = item.get("expectations", {})
        actual = expectations.get(pred.key)
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "span":
        # Span predicates are handled by the hybrid search layer in tracking_store.
        # Skip here so they don't accidentally pass through.
        return True

    return True  # Unknown type, don't filter


def _execute_trace_fts(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute a full-text search query strategy for traces."""
    from mlflow_dynamodbstore.dynamodb.fts import tokenize_trigrams, tokenize_words
    from mlflow_dynamodbstore.dynamodb.schema import SK_FTS_PREFIX, SK_TRACE_PREFIX

    if not plan.fts_query:
        return []

    # Tokenize with word stemmer
    word_tokens = tokenize_words(plan.fts_query)

    def _extract_trace_ids(fts_items: list[dict[str, Any]]) -> set[str]:
        """Extract trace_ids from FTS items with T# entity prefix."""
        ids = set()
        for item in fts_items:
            sk = item.get("SK", "")
            # FTS#<level>#T#<token>#<trace_id>[#<field>]
            # parts[0]=FTS, parts[1]=level, parts[2]=T, parts[3]=token, parts[4]=trace_id
            parts = sk.split("#")
            if len(parts) >= 5 and parts[2] == "T":
                ids.add(parts[4])
        return ids

    for level, tokens in [
        ("W", word_tokens),
        ("3", tokenize_trigrams(plan.fts_query) if not word_tokens else []),
    ]:
        if not tokens:
            continue
        entity_id_sets: list[set[str]] = []
        for token in tokens:
            sk_prefix = f"{SK_FTS_PREFIX}{level}#T#{token}#"
            fts_items = table.query(pk=pk, sk_prefix=sk_prefix)
            entity_id_sets.append(_extract_trace_ids(fts_items))

        if not entity_id_sets:
            continue

        result_ids = entity_id_sets[0]
        for s in entity_id_sets[1:]:
            result_ids &= s

        if result_ids:
            # Fetch trace META items
            items = []
            for tid in result_ids:
                meta = table.get_item(pk, f"{SK_TRACE_PREFIX}{tid}")
                if meta and _is_trace_meta_item(meta):
                    items.append(meta)
            return items

    return []


def _execute_trace_index(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute an index-based query strategy for traces.

    For LSIs with STRING sort keys (lsi3, lsi4), we can query the index
    directly with sk_prefix. For numeric LSIs (lsi1, lsi2, lsi5), we
    query the main table by SK prefix "T#" to get only trace items,
    then sort by the LSI attribute in Python.
    """
    from mlflow_dynamodbstore.dynamodb.schema import (
        LSI1_SK,
        LSI2_SK,
        LSI4_SK,
        LSI5_SK,
        SK_TRACE_PREFIX,
    )

    # LSIs with string sort keys can use begins_with
    string_lsis = {"lsi3", "lsi4"}

    if plan.index in string_lsis and plan.sk_prefix is not None:
        # Query the LSI directly with sk_prefix, with optional FilterExpression
        items = table.query(
            pk=pk,
            sk_prefix=plan.sk_prefix,
            index_name=plan.index,
            scan_forward=plan.scan_forward,
            filter_expression=plan.filter_expression,
        )
        # Filter to trace META items only
        return [item for item in items if _is_trace_meta_item(item)]

    # For numeric LSIs (lsi1, lsi2, lsi5): query main table by SK prefix T#
    # to get all trace items, then sort in Python by the LSI attribute.
    items = table.query(pk=pk, sk_prefix=SK_TRACE_PREFIX, filter_expression=plan.filter_expression)
    # Filter to META items only (exclude sub-items like T#<id>#TAG#...)
    meta_items = [item for item in items if _is_trace_meta_item(item)]

    # Determine sort key attribute
    lsi_attr_map = {
        "lsi1": LSI1_SK,
        "lsi2": LSI2_SK,
        "lsi4": LSI4_SK,
        "lsi5": LSI5_SK,
    }
    sort_attr = lsi_attr_map.get(plan.index or "lsi1", LSI1_SK)

    # Sort by the LSI attribute
    meta_items.sort(
        key=lambda item: item.get(sort_attr, 0),
        reverse=not plan.scan_forward,
    )
    return meta_items


def execute_trace_query(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
    max_results: int,
    page_token: str | None = None,
    predicates: list[FilterPredicate] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Execute a DynamoDB query for traces based on a QueryPlan.

    Similar to :func:`execute_query` but filters results to only include
    trace META items and applies trace-specific post-filters.

    Args:
        table: DynamoDBTable instance.
        plan: The QueryPlan describing the execution strategy.
        pk: The partition key (e.g. ``"EXP#01JQ"``).
        max_results: Maximum number of trace META items to return.
        page_token: Opaque pagination token from a previous call.
        predicates: Full list of filter predicates for trace search.

    Returns:
        A tuple ``(items, next_page_token)`` where *items* are trace META
        item dicts and *next_page_token* is ``None`` when there are no
        more results.
    """
    from mlflow_dynamodbstore.dynamodb.pagination import (
        decode_page_token,
        encode_page_token,
    )

    predicates = predicates or []
    token_data = decode_page_token(page_token)
    offset = token_data.get("offset", 0) if token_data else 0

    if plan.strategy == "fts":
        items = _execute_trace_fts(table, plan, pk)
    elif plan.strategy == "index":
        items = _execute_trace_index(table, plan, pk)
    else:
        items = []

    # Apply post-filters
    filtered: list[dict[str, Any]] = []
    for item in items:
        trace_id = item.get("trace_id", "")
        if all(
            _apply_trace_post_filter(table, pk, trace_id, item, pred) for pred in plan.post_filters
        ):
            filtered.append(item)

    # Pagination: simple offset-based
    page = filtered[offset : offset + max_results]
    has_more = len(filtered) > offset + max_results
    next_token: str | None = None
    if has_more:
        next_token = encode_page_token({"offset": offset + max_results})

    return page, next_token


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


def build_experiment_filter_expression(
    predicates: list[FilterPredicate],
) -> ConditionBase | None:
    """Build a DynamoDB FilterExpression from experiment search predicates.

    Uses ``name_lower``/``name_lower_rev`` and ``tags_lower``/``tags_lower_rev``
    for case-insensitive (ILIKE) comparisons. Suffix patterns use reversed
    attributes with ``begins_with`` to avoid post-filtering.

    Returns None if no predicates are provided.
    """
    if not predicates:
        return None

    conditions: list[ConditionBase] = []
    for pred in predicates:
        if pred.field_type == "attribute" and pred.key == "name":
            _build_name_condition(pred, conditions)
        elif pred.field_type == "attribute" and pred.key in (
            "creation_time",
            "last_update_time",
        ):
            _build_time_condition(pred, conditions)
        elif pred.field_type == "tag":
            _build_tag_condition(pred, conditions)

    if not conditions:
        return None
    result = conditions[0]
    for c in conditions[1:]:
        result = result & c
    return result


def _rev(s: str) -> str:
    return s[::-1]


def _build_name_condition(pred: FilterPredicate, conditions: list[ConditionBase]) -> None:
    """Build FilterExpression condition for experiment name predicates.

    Suffix patterns use reversed attributes (``name_rev``, ``name_lower_rev``)
    with ``begins_with`` — fully server-side, no post-filtering needed.
    """
    if pred.op == "=":
        conditions.append(Attr("name").eq(pred.value))
    elif pred.op == "!=":
        conditions.append(Attr("name").ne(pred.value))
    elif pred.op == "LIKE":
        pattern = pred.value
        if pattern.startswith("%") and pattern.endswith("%"):
            conditions.append(Attr("name").contains(pattern.strip("%")))
        elif pattern.endswith("%"):
            conditions.append(Attr("name").begins_with(pattern.rstrip("%")))
        elif pattern.startswith("%"):
            suffix = pattern.lstrip("%")
            conditions.append(Attr("name_rev").begins_with(_rev(suffix)))
        else:
            conditions.append(Attr("name").eq(pattern))
    elif pred.op == "ILIKE":
        pattern = pred.value.lower()
        if pattern.startswith("%") and pattern.endswith("%"):
            conditions.append(Attr("name_lower").contains(pattern.strip("%")))
        elif pattern.endswith("%"):
            conditions.append(Attr("name_lower").begins_with(pattern.rstrip("%")))
        elif pattern.startswith("%"):
            suffix = pattern.lstrip("%")
            conditions.append(Attr("name_lower_rev").begins_with(_rev(suffix)))
        else:
            conditions.append(Attr("name_lower").eq(pattern))


def _build_time_condition(pred: FilterPredicate, conditions: list[ConditionBase]) -> None:
    """Build FilterExpression condition for time attribute predicates."""
    # Map filter key to DynamoDB attribute name
    attr_name = pred.key
    val = int(pred.value) if pred.value is not None else 0
    match pred.op:
        case "=":
            conditions.append(Attr(attr_name).eq(val))
        case "!=":
            conditions.append(Attr(attr_name).ne(val))
        case ">":
            conditions.append(Attr(attr_name).gt(val))
        case ">=":
            conditions.append(Attr(attr_name).gte(val))
        case "<":
            conditions.append(Attr(attr_name).lt(val))
        case "<=":
            conditions.append(Attr(attr_name).lte(val))


def _build_tag_condition(pred: FilterPredicate, conditions: list[ConditionBase]) -> None:
    """Build FilterExpression condition for tag predicates.

    Suffix patterns use reversed tag maps (``tags_lower_rev``) with ``begins_with``.
    """
    tag_attr = f"tags.{pred.key}"
    tag_lower_attr = f"tags_lower.{pred.key}"
    tag_lower_rev_attr = f"tags_lower_rev.{pred.key}"
    match pred.op:
        case "=":
            conditions.append(Attr(tag_attr).eq(pred.value))
        case "!=":
            # != means tag exists AND has a different value
            conditions.append(Attr(tag_attr).exists() & Attr(tag_attr).ne(pred.value))
        case "LIKE":
            pattern = pred.value
            if pattern.startswith("%") and pattern.endswith("%"):
                conditions.append(Attr(tag_attr).contains(pattern.strip("%")))
            elif pattern.endswith("%"):
                conditions.append(Attr(tag_attr).begins_with(pattern.rstrip("%")))
            elif pattern.startswith("%"):
                suffix = pattern.lstrip("%")
                # Case-sensitive suffix: no reversed original-case map stored.
                # contains() may produce false positives (matches substring, not
                # just suffix). Acceptable for experiments (small result sets).
                # TODO: add tags_rev map for exact suffix matching if needed.
                conditions.append(Attr(tag_attr).contains(suffix))
            else:
                conditions.append(Attr(tag_attr).eq(pattern))
        case "ILIKE":
            pattern = pred.value.lower()
            if pattern.startswith("%") and pattern.endswith("%"):
                conditions.append(Attr(tag_lower_attr).contains(pattern.strip("%")))
            elif pattern.endswith("%"):
                conditions.append(Attr(tag_lower_attr).begins_with(pattern.rstrip("%")))
            elif pattern.startswith("%"):
                suffix = pattern.lstrip("%")
                conditions.append(Attr(tag_lower_rev_attr).begins_with(_rev(suffix)))
            else:
                conditions.append(Attr(tag_lower_attr).eq(pattern))
        case "IS NULL":
            conditions.append(Attr(tag_attr).not_exists())
        case "IS NOT NULL":
            conditions.append(Attr(tag_attr).exists())


def parse_logged_model_filter(filter_string: str | None) -> list[FilterPredicate]:
    """Parse an MLflow logged model search filter string into a list of FilterPredicates.

    Delegates actual parsing to :class:`mlflow.utils.search_utils.SearchLoggedModelsUtils`
    and normalises the result into our :class:`FilterPredicate` dataclass.

    Args:
        filter_string: An MLflow filter expression such as
            ``"metrics.accuracy > 0.9 AND tags.env = 'prod'"``
            or ``None`` / empty string for no filter.

    Returns:
        A (possibly empty) list of :class:`FilterPredicate` objects, one per AND clause.
    """
    if not filter_string:
        return []
    from mlflow.utils.search_utils import SearchLoggedModelsUtils

    parsed = SearchLoggedModelsUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
    return _to_predicates(parsed)


# LSI mapping for logged model order_by attributes
_LM_ORDER_BY_LSI: dict[str, str] = {
    "creation_timestamp": "lsi2",
    "creation_time": "lsi2",
    "last_updated_timestamp": "lsi2",
    "name": "lsi4",
}


def plan_logged_model_query(
    predicates: list[FilterPredicate],
    order_by: list[dict[str, Any]] | None,
    datasets: list[dict[str, Any]] | None,
) -> QueryPlan:
    """Analyze predicates and order_by to produce an optimal DynamoDB QueryPlan for logged models.

    Strategy selection priority:
    1. RANK  -- metric field_name in order_by
    2. LSI3  -- status= filter (composite key: status#model_id)
    3. LSI4  -- name order_by
    4. LSI2  -- creation_timestamp/last_updated_timestamp order_by
    5. Default -- lifecycle filter via LSI1 (active#)

    Metric predicates are always separated into ``rank_filters`` and applied
    post-fetch via RANK item range queries, regardless of the chosen strategy.

    Args:
        predicates: Parsed filter predicates from :func:`parse_logged_model_filter`.
        order_by: List of order-by dicts with ``field_name`` and ``ascending`` keys,
            as produced by ``SearchLoggedModelsUtils``.
        datasets: Optional list of dataset dicts for dataset-scoped metric RANK queries.

    Returns:
        A :class:`QueryPlan` describing the chosen execution strategy.
    """
    # ------------------------------------------------------------------ #
    # 1. Check for RANK strategy (metric field_name in order_by)          #
    # ------------------------------------------------------------------ #
    if order_by:
        for ob in order_by:
            field_name = ob.get("field_name", "")
            if "." in field_name:
                entity, key = field_name.split(".", 1)
                if entity == "metrics":
                    ascending = ob.get("ascending", True)
                    return QueryPlan(
                        strategy="rank",
                        index=None,
                        sk_prefix=None,
                        # RANK items use inverted values, so ascending query needs DESC scan
                        scan_forward=ascending,
                        rank_key=key,
                        rank_filters=[p for p in predicates if p.field_type == "metric"],
                        post_filters=[p for p in predicates if p.field_type != "metric"],
                        datasets=datasets,
                    )

    # Separate metric predicates as rank_filters for all non-RANK strategies
    rank_filters = [p for p in predicates if p.field_type == "metric"]
    non_metric_preds = [p for p in predicates if p.field_type != "metric"]

    # ------------------------------------------------------------------ #
    # 2. Check for status= filter -> LSI3                                 #
    # ------------------------------------------------------------------ #
    for pred in non_metric_preds:
        if pred.field_type == "attribute" and pred.key == "status" and pred.op == "=":
            remaining = [p for p in non_metric_preds if p is not pred]
            return QueryPlan(
                strategy="index",
                index="lsi3",
                sk_prefix=f"{pred.value}#",
                scan_forward=False,
                post_filters=remaining,
                rank_filters=rank_filters,
                datasets=datasets,
            )

    # ------------------------------------------------------------------ #
    # 3. Determine index from order_by attribute                          #
    # ------------------------------------------------------------------ #
    chosen_index = "lsi1"
    scan_forward = False
    sk_prefix: str | None = "active#"

    if order_by:
        for ob in order_by:
            field_name = ob.get("field_name", "")
            if field_name in _LM_ORDER_BY_LSI:
                chosen_index = _LM_ORDER_BY_LSI[field_name]
                scan_forward = ob.get("ascending", True)
                sk_prefix = None
                break

    return QueryPlan(
        strategy="index",
        index=chosen_index,
        sk_prefix=sk_prefix,
        scan_forward=scan_forward,
        post_filters=non_metric_preds,
        rank_filters=rank_filters,
        datasets=datasets,
    )


def _is_logged_model_meta(item: dict[str, Any]) -> bool:
    """Check if a DynamoDB item is a logged model META item (not a sub-item)."""
    return "model_id" in item and "lifecycle_stage" in item


def _execute_lm_rank(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute rank-based strategy for logged models."""
    from mlflow_dynamodbstore.dynamodb.schema import (
        SK_LM_PREFIX,
        SK_RANK_LM_PREFIX,
        SK_RANK_LMD_PREFIX,
    )

    sk_prefix = f"{SK_RANK_LM_PREFIX}{plan.rank_key}#"
    # If datasets specified, use dataset-scoped RANK
    if plan.datasets:
        ds = plan.datasets[0]
        ds_name = ds.get("name", "")
        ds_digest = ds.get("digest", "")
        if ds_name:
            sk_prefix = f"{SK_RANK_LMD_PREFIX}{plan.rank_key}#{ds_name}#"
            if ds_digest:
                sk_prefix = f"{SK_RANK_LMD_PREFIX}{plan.rank_key}#{ds_name}#{ds_digest}#"

    rank_items = table.query(
        pk=pk,
        sk_prefix=sk_prefix,
        # RANK items use inverted values; ascending needs forward scan
        scan_forward=not plan.scan_forward,
    )

    model_ids = [item["model_id"] for item in rank_items if "model_id" in item]
    # Batch get META items, preserving rank order and excluding deleted models
    items: list[dict[str, Any]] = []
    for mid in model_ids:
        meta = table.get_item(pk, f"{SK_LM_PREFIX}{mid}")
        if meta and _is_logged_model_meta(meta) and meta.get("lifecycle_stage") != "deleted":
            items.append(meta)
    return items


def _execute_lm_index(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
) -> list[dict[str, Any]]:
    """Execute index-based strategy for logged models."""
    items = table.query(
        pk=pk,
        sk_prefix=plan.sk_prefix,
        index_name=plan.index,
        scan_forward=plan.scan_forward,
    )
    # Filter to logged model META items only (excludes tag/param/metric sub-items)
    return [item for item in items if _is_logged_model_meta(item)]


def _apply_lm_post_filter(
    table: DynamoDBTable,
    pk: str,
    model_id: str,
    item: dict[str, Any],
    pred: FilterPredicate,
) -> bool:
    """Apply a single post-filter predicate for logged model search.

    For logged models, tags and params are denormalized onto the META item,
    so no sub-item lookups are required for those field types.
    """
    if pred.field_type == "attribute":
        # Map alternate attribute key names to actual item field names
        key_map = {
            "creation_timestamp": "creation_timestamp_ms",
            "creation_time": "creation_timestamp_ms",
            "last_updated_timestamp": "last_updated_timestamp_ms",
        }
        item_key = key_map.get(pred.key, pred.key)
        actual = item.get(item_key)
        # Coerce timestamp comparisons to int
        if pred.key in ("creation_timestamp", "creation_time", "last_updated_timestamp"):
            if actual is not None:
                actual = int(actual)
            expected = int(pred.value) if pred.value is not None else pred.value
            return _compare(actual, pred.op, expected)
        return _compare(actual, pred.op, pred.value)

    if pred.field_type == "tag":
        tags = item.get("tags", {})
        return _compare(tags.get(pred.key), pred.op, pred.value)

    if pred.field_type == "param":
        params = item.get("params", {})
        return _compare(params.get(pred.key), pred.op, pred.value)

    return True  # Unknown type, don't filter


def execute_logged_model_query(
    table: DynamoDBTable,
    plan: QueryPlan,
    pk: str,
    predicates: list[FilterPredicate] | None = None,
) -> list[dict[str, Any]]:
    """Execute a DynamoDB query for logged models based on a QueryPlan.

    Unlike :func:`execute_query` and :func:`execute_trace_query`, this function
    returns a flat list without pagination. The caller (``search_logged_models``)
    is responsible for merging results across multiple experiments and applying
    offset-based pagination.

    Args:
        table: DynamoDBTable instance.
        plan: The QueryPlan describing the execution strategy.
        pk: The partition key (e.g. ``"EXP#01JQ"``).
        predicates: Full list of filter predicates (unused here; kept for API
            symmetry with other execute functions).

    Returns:
        A list of logged model META item dicts matching the query.
    """
    from mlflow_dynamodbstore.dynamodb.schema import SK_RANK_LM_PREFIX

    predicates = predicates or []

    if plan.strategy == "rank":
        items = _execute_lm_rank(table, plan, pk)
    else:
        items = _execute_lm_index(table, plan, pk)

    # Apply rank_filters (metric predicates) via RANK item range queries
    if plan.rank_filters and plan.strategy != "rank":
        from mlflow_dynamodbstore.dynamodb.schema import SK_RANK_LMD_PREFIX

        for rf in plan.rank_filters:
            # Use dataset-scoped RANK items when datasets are specified
            if plan.datasets:
                ds = plan.datasets[0]
                ds_name = ds.get("dataset_name", ds.get("name", ""))
                ds_digest = ds.get("dataset_digest", ds.get("digest", ""))
                rank_sk = f"{SK_RANK_LMD_PREFIX}{rf.key}#{ds_name}#"
                if ds_digest:
                    rank_sk = f"{SK_RANK_LMD_PREFIX}{rf.key}#{ds_name}#{ds_digest}#"
            else:
                rank_sk = f"{SK_RANK_LM_PREFIX}{rf.key}#"
            rank_items = table.query(pk=pk, sk_prefix=rank_sk)
            matching_ids: set[str] = set()
            for ri in rank_items:
                val = float(ri.get("metric_value", 0))
                if _compare(val, rf.op, rf.value):
                    matching_ids.add(ri["model_id"])
            items = [i for i in items if i.get("model_id") in matching_ids]

    # Apply dataset-only filter when datasets specified but no metric predicates
    if plan.datasets and not plan.rank_filters and plan.strategy != "rank":
        from mlflow_dynamodbstore.dynamodb.schema import SK_RANK_LMD_PREFIX

        ds = plan.datasets[0]
        ds_name = ds.get("dataset_name", ds.get("name", ""))
        ds_digest = ds.get("dataset_digest", ds.get("digest", ""))
        rank_sk = f"{SK_RANK_LMD_PREFIX}"
        # Query all dataset-scoped RANK items and collect model_ids matching the dataset
        all_ds_rank = table.query(pk=pk, sk_prefix=rank_sk)
        ds_model_ids: set[str] = set()
        for ri in all_ds_rank:
            ri_name = ri.get("dataset_name", "")
            ri_digest = ri.get("dataset_digest", "")
            if ri_name == ds_name and (not ds_digest or ri_digest == ds_digest):
                ds_model_ids.add(ri["model_id"])
        items = [i for i in items if i.get("model_id") in ds_model_ids]

    # Apply post-filters
    filtered: list[dict[str, Any]] = []
    for item in items:
        model_id = item.get("model_id", "")
        if all(
            _apply_lm_post_filter(table, pk, model_id, item, pred) for pred in plan.post_filters
        ):
            filtered.append(item)

    return filtered
