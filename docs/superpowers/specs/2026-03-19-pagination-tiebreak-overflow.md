# Pagination Tiebreak with Overflow Cache

## Problem

`search_registered_models` with `order_by=["timestamp DESC"]` returns items with tied timestamps in arbitrary order. MLflow expects a deterministic secondary sort by `name ASC`. DynamoDB sorts by a single key per index — the GSI2 SK `{ts:020d}#{name}` naturally tiebreaks by name in the same direction as the primary sort, but the test expects the **opposite** direction for ties (timestamp DESC, name ASC).

Additionally, pagination must not split a tie group across pages. If items with the same timestamp span a page boundary, the re-sorted order would be inconsistent between pages.

## Design

### Tie Group Detection

A **tie group** is a set of consecutive items sharing the same primary sort value (timestamp) when ordered by DynamoDB.

**Rule:** Always fetch `max_results + 1` items. If item at index `max_results` ties with item at index `max_results - 1`, the tie group is split — keep fetching until the timestamp changes or DDB is exhausted.

### Example

```
DDB order (timestamp ASC via GSI2):
  A(t=1), B(t=1), C(t=2), D(t=2), E(t=2), F(t=3)

max_results = 3
order_by = ["timestamp ASC", "name DESC"]

Step 1: Fetch 4 items: A(1), B(1), C(2), D(2)
Step 2: items[2].ts == items[3].ts (both t=2) → tie group split
Step 3: Keep fetching: E(2), F(3) → F has different ts → group complete
Step 4: Full buffer: A(1), B(1), C(2), D(2), E(2), F(3)
Step 5: Sort within tie groups by secondary key (name DESC):
         B(1), A(1), E(2), D(2), C(2), F(3)
Step 6: Return first 3: [B, A, E]
Step 7: Cache overflow [D, C, F] + DDB LEK (after F)
Step 8: Token → cache hash, page 0
```

### Page 2 (token points to cache)

```
Step 1: Read cache page 0: [D, C, F]
Step 2: Return [D, C, F]
Step 3: Cache exhausted, read stored DDB LEK
Step 4: Token → DDB LEK (if more data) or None
```

### Overflow Cache

Follows the trace metrics cache pattern (`TMCACHE#`).

**Key format:**
```
PK: OVERFLOW#<cache_hash>
SK: PAGE#00000000     → first overflow page (up to max_results serialized models)
SK: PAGE#00000001     → second overflow page
SK: PAGE#<last>       → last page, also stores "ddb_lek" for DDB continuation
```

**Cache hash:** SHA256 of `(pk, index_name, order_by, lek_or_none)`[:16] — uniquely identifies the query + position. The `pk` already contains the workspace, so the hash is workspace-unique.

**TTL:** 900 seconds (15 minutes), matching trace metrics cache.

**Page content:**
```json
{
  "PK": "OVERFLOW#<hash>",
  "SK": "PAGE#00000000",
  "models": "[{name, creation_ts, last_updated_ts, description, tags, aliases}, ...]",
  "ttl": <epoch + 900>
}
```

Last page additionally has:
```json
{
  "ddb_lek": "{...}"  // or null if DDB exhausted
}
```

**Sharding:** Each page item stores up to `max_results` models as JSON. If serialized size exceeds 390KB (leaving margin for key/metadata), split into sub-pages. In practice, model metadata is small (~200 bytes), so `max_results=100` ≈ 20KB — well under the limit. Sharding is a safety net.

**Size check:**
```python
serialized = json.dumps(models_data)
if len(serialized.encode()) > 390_000:
    # Split models_data into smaller chunks and write multiple PAGE# items
```

### Page Token Format

Current token: `base64(json({"PK": ..., "SK": ..., ...}))` — a DDB LEK.

New token discriminator:
```python
# DDB cursor (backward compatible — no "overflow" key):
{"PK": "...", "SK": "...", "gsi2pk": "...", "gsi2sk": "..."}

# Overflow cursor:
{"overflow": "<cache_hash>", "page": 0}
```

Decoding: check for `"overflow"` key. If present → read from cache. If absent → treat as DDB LEK.

### Sort Implementation

`_sort_models` supports multi-column ordering:
- Parse all `order_by` clauses
- When primary is timestamp and no secondary specified, implicit tiebreaker is `name ASC`
- Use a `_Negate` wrapper for DESC columns in a single `sorted()` call
- Only sorts within the fetched buffer — not the entire dataset

### When Tiebreak Is Needed

```python
def _needs_secondary_sort(order_by) -> bool:
    if not order_by or len(order_by) > 1:
        return len(order_by) > 1 if order_by else False
    key = order_by[0].split()[0].lower()
    return key in ("last_updated_timestamp", "timestamp")
```

When `_needs_secondary_sort` is False, the existing cursor-based pagination works unchanged.

### Edge Cases

1. **Tie group smaller than page:** Normal case. No overflow cache needed.

2. **Tie group exactly at page boundary:** Item at `max_results` has different timestamp. Normal pagination, no cache.

3. **Tie group spans page boundary:** Over-fetch to complete group, cache overflow.

4. **Tie group larger than `max_results`:** The entire page is one tie group. Sort it, return first `max_results`, cache rest in multiple pages.

5. **Tie group larger than 400KB per page:** Shard into sub-pages. Each sub-page < 390KB.

6. **All items have same timestamp:** One giant tie group. Sort by secondary key. Paginate through cache pages, last page has DDB LEK (or null if exhausted).

7. **DDB exhausted mid-group:** No more data. Sort what we have, return up to max_results, cache rest.

8. **Cache expired (15min TTL):** Client sends stale token. Return error — caller must restart from page 1.

## Changes

### `src/mlflow_dynamodbstore/dynamodb/schema.py`
- Add `PK_OVERFLOW_PREFIX = "OVERFLOW#"`
- Add `SK_OVERFLOW_PAGE_PREFIX = "PAGE#"`

### `src/mlflow_dynamodbstore/dynamodb/overflow_cache.py` (new)
- `compute_cache_hash(workspace, pk, index_name, order_by, lek)` → 16-char hex
- `cache_put_overflow(table, hash, pages, ddb_lek, ttl=900)` — write pages + LEK
- `cache_get_overflow_page(table, hash, page_idx)` → `(models_data, ddb_lek | None, is_last)`
- Sharding logic for >390KB pages

### `src/mlflow_dynamodbstore/registry_store.py`
- Add `_Negate` class and `_negate()` helper for multi-column sort
- Update `_sort_models` for multi-column ordering with implicit tiebreaker
- Add `_needs_secondary_sort` static method
- Update `_search_models_paginated` with tie group detection, overflow cache, token format
- Update `_decode_page_token` to handle overflow tokens

### `tests/unit/test_registry_search.py`
- `TestSearchRegisteredModelOrderBy` class with edge case tests:
  - Tiebreak with explicit secondary sort
  - Implicit name ASC tiebreak on timestamp DESC
  - Pagination with tie groups at boundaries
  - Tie group larger than page size
  - Both directions reversed (DESC, DESC)

### `tests/compatibility/test_registry_compat.py`
- Remove Cat 13 xfail for `test_search_registered_model_order_by`

## Verification

```bash
uv run pytest tests/unit/test_registry_search.py::TestSearchRegisteredModelOrderBy -x -v
uv run pytest tests/compatibility/test_registry_compat.py::test_search_registered_model_order_by -x -v --runxfail
uv run pytest tests/compatibility/test_registry_compat.py -v
uv run pytest tests/unit/ -x -q
```
