"""Overflow cache for pagination tiebreak handling.

When paginating search results ordered by timestamp, tie groups (items sharing
the same timestamp) may cause overflow beyond ``max_results``.  The excess
models are cached in DynamoDB pages so that subsequent page-token requests can
retrieve them without re-querying.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from base64 import b64encode
from typing import Any

logger = logging.getLogger(__name__)

OVERFLOW_TTL_SECONDS = 900  # 15 minutes

MAX_ITEM_BYTES = 390_000  # safety margin under DynamoDB's 400 KB limit


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------


def compute_cache_hash(
    pk: str,
    index_name: str,
    order_by: list[str] | None,
    lek: dict[str, Any] | None,
) -> str:
    """Return a 16-char hex digest uniquely identifying the query context."""
    key_parts = {
        "pk": pk,
        "index_name": index_name,
        "order_by": order_by,
        "lek": lek,
    }
    canonical = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache write
# ---------------------------------------------------------------------------


def cache_put_overflow(
    table: Any,
    cache_hash: str,
    models_data: list[dict[str, Any]],
    ddb_lek: dict[str, Any] | None,
    max_per_page: int,
) -> None:
    """Split *models_data* into pages and write them to the overflow cache."""
    from mlflow_dynamodbstore.dynamodb.schema import (
        PK_OVERFLOW_PREFIX,
        SK_OVERFLOW_PAGE_PREFIX,
    )

    ttl = int(time.time()) + OVERFLOW_TTL_SECONDS
    pk = f"{PK_OVERFLOW_PREFIX}{cache_hash}"

    # Split models into pages of max_per_page
    pages: list[list[dict[str, Any]]] = []
    for i in range(0, len(models_data), max_per_page):
        pages.append(models_data[i : i + max_per_page])

    if not pages:
        pages = [[]]  # write at least one (empty) page so the token resolves

    for page_idx, page_models in enumerate(pages):
        sk = f"{SK_OVERFLOW_PAGE_PREFIX}{page_idx:08d}"
        serialized_models = json.dumps(page_models)

        is_last = page_idx == len(pages) - 1

        item: dict[str, Any] = {
            "PK": pk,
            "SK": sk,
            "models": serialized_models,
            "ttl": ttl,
        }

        if is_last:
            item["ddb_lek"] = json.dumps(ddb_lek) if ddb_lek is not None else ""

        # Size guard
        estimated_size = len(json.dumps(item).encode("utf-8"))
        if estimated_size > MAX_ITEM_BYTES:
            logger.warning(
                "Overflow cache page %d for hash %s is %d bytes, exceeding "
                "the %d byte safety limit. Consider reducing max_results.",
                page_idx,
                cache_hash,
                estimated_size,
                MAX_ITEM_BYTES,
            )
            raise ValueError(
                f"Overflow cache item too large ({estimated_size} bytes). "
                f"Reduce max_results or model payload size."
            )

        table.put_item(item)


# ---------------------------------------------------------------------------
# Cache read
# ---------------------------------------------------------------------------


def cache_get_overflow_page(
    table: Any,
    cache_hash: str,
    page_idx: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, bool]:
    """Read a single overflow page from the cache.

    Returns
    -------
    (models_data, ddb_lek, is_last)
        *models_data* is ``None`` on cache miss or TTL expiry.
        *is_last* is ``True`` when this page carries the ``ddb_lek`` field
        (indicating no further overflow pages exist).
        *ddb_lek* is the DynamoDB ``LastEvaluatedKey`` to resume scanning
        after the overflow cache is consumed, or ``None`` when the original
        query was fully exhausted.
    """
    from mlflow_dynamodbstore.dynamodb.schema import (
        PK_OVERFLOW_PREFIX,
        SK_OVERFLOW_PAGE_PREFIX,
    )

    pk = f"{PK_OVERFLOW_PREFIX}{cache_hash}"
    sk = f"{SK_OVERFLOW_PAGE_PREFIX}{page_idx:08d}"

    item = table.get_item(pk=pk, sk=sk)
    if item is None:
        return None, None, False

    # Manual TTL check (DynamoDB TTL deletion is eventually consistent)
    if item.get("ttl") and int(item["ttl"]) < int(time.time()):
        return None, None, False

    models_data: list[dict[str, Any]] = json.loads(item["models"])

    is_last = "ddb_lek" in item
    ddb_lek: dict[str, Any] | None = None
    if is_last:
        raw_lek = item["ddb_lek"]
        if raw_lek:  # non-empty string → parse
            ddb_lek = json.loads(raw_lek)
        # empty string → original query exhausted, ddb_lek stays None

    return models_data, ddb_lek, is_last


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def encode_overflow_token(cache_hash: str, page_idx: int) -> str:
    """Encode an overflow page reference as a base64 page token."""
    return b64encode(json.dumps({"overflow": cache_hash, "page": page_idx}).encode()).decode()


def is_overflow_token(decoded_dict: dict[str, Any]) -> bool:
    """Return ``True`` if *decoded_dict* represents an overflow page token."""
    return "overflow" in decoded_dict


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cache_cleanup(table: Any, cache_hash: str) -> None:
    """Delete all overflow pages for *cache_hash*."""
    from mlflow_dynamodbstore.dynamodb.schema import (
        PK_OVERFLOW_PREFIX,
        SK_OVERFLOW_PAGE_PREFIX,
    )

    pk = f"{PK_OVERFLOW_PREFIX}{cache_hash}"
    items = table.query(pk=pk, sk_prefix=SK_OVERFLOW_PAGE_PREFIX)
    for item in items:
        table.delete_item(pk=pk, sk=item["SK"])
