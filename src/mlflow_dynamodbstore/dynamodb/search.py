from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlflow.entities import ViewType
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils

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


def _compare(actual: Any, op: str, expected: Any) -> bool:
    """Evaluate a single comparison predicate."""
    if actual is None:
        return op in ("IS NULL", "!=", "NOT IN")

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


def _apply_attribute_filter(item: dict[str, Any], pred: FilterPredicate) -> bool:
    """Check an attribute-level predicate against a META item."""
    return _compare(item.get(pred.key), pred.op, pred.value)


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
    # Fetch more than needed to account for filtering
    fetch_limit = offset + max_results * 2 if plan.filter_expressions else None
    items = table.query(
        pk=pk,
        sk_prefix=plan.sk_prefix,
        index_name=plan.index,
        scan_forward=plan.scan_forward,
        limit=fetch_limit,
    )

    # Apply denormalized tag filters
    if plan.filter_expressions:
        items = [
            item
            for item in items
            if _apply_denormalized_tag_filters(item, plan.filter_expressions, predicates)
        ]

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
            sk_prefix = f"{SK_FTS_PREFIX}W#{token}#R#"
            fts_items = table.query(pk=pk, sk_prefix=sk_prefix)
            # Extract entity_id (run_id) from the SK pattern:
            # FTS#W#<token>#R#<run_id>
            ids = set()
            for item in fts_items:
                sk = item.get("SK", "")
                # Parse run_id from SK: FTS#W#<token>#R#<run_id>[#<field>]
                parts = sk.split("#")
                # Find the R marker and get the next part
                for i, part in enumerate(parts):
                    if part == "R" and i + 1 < len(parts):
                        ids.add(parts[i + 1])
                        break
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
            sk_prefix = f"{SK_FTS_PREFIX}3#{token}#R#"
            fts_items = table.query(pk=pk, sk_prefix=sk_prefix)
            ids = set()
            for item in fts_items:
                sk = item.get("SK", "")
                parts = sk.split("#")
                for i, part in enumerate(parts):
                    if part == "R" and i + 1 < len(parts):
                        ids.add(parts[i + 1])
                        break
            entity_id_sets.append(ids)

        if not entity_id_sets:
            return []

        result_ids = entity_id_sets[0]
        for s in entity_id_sets[1:]:
            result_ids &= s

        if result_ids:
            return _fetch_meta_items(table, pk, list(result_ids))

    return []


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
