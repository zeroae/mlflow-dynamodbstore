# Pagination Tiebreak with Overflow Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable deterministic tiebreaking in `search_registered_models` order_by, with pagination-safe overflow cache for tie groups that span page boundaries.

**Architecture:** Over-fetch to complete tie groups at page boundaries, sort within groups in Python, cache overflow in DynamoDB with 15min TTL, shard pages at 390KB.

**Tech Stack:** DynamoDB, boto3, MLflow SearchUtils

**Spec:** `docs/superpowers/specs/2026-03-19-pagination-tiebreak-overflow.md`

---

## File Structure

- **Create:** `src/mlflow_dynamodbstore/dynamodb/overflow_cache.py` — cache hash, put/get, sharding
- **Modify:** `src/mlflow_dynamodbstore/dynamodb/schema.py` — overflow key constants (done)
- **Modify:** `src/mlflow_dynamodbstore/registry_store.py` — `_Negate`, `_sort_models`, `_needs_secondary_sort`, `_search_models_paginated` tie group logic
- **Modify:** `tests/unit/test_registry_search.py` — edge case tests
- **Modify:** `tests/compatibility/test_registry_compat.py` — remove Cat 13 xfail for order_by

---

### Task 1: Create overflow cache module

**Files:**
- Create: `src/mlflow_dynamodbstore/dynamodb/overflow_cache.py`

- [ ] **Step 1: Write `compute_cache_hash`**

```python
def compute_cache_hash(pk: str, index_name: str, order_by: list[str] | None, lek: dict | None) -> str:
    """16-char hex hash uniquely identifying this query position."""
    key_parts = {"pk": pk, "index_name": index_name, "order_by": order_by or [], "lek": lek}
    canonical = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

- [ ] **Step 2: Write `cache_put_overflow`**

Stores overflow models in PAGE# items, each up to max_results models. Last page includes `ddb_lek`. Shards if serialized size > 390KB.

```python
OVERFLOW_TTL_SECONDS = 900
MAX_ITEM_BYTES = 390_000

def cache_put_overflow(
    table, cache_hash: str, models_data: list[dict], ddb_lek: dict | None, max_per_page: int
) -> None:
    """Write overflow pages. Each page has up to max_per_page serialized models."""
    pages = [models_data[i:i+max_per_page] for i in range(0, len(models_data), max_per_page)]
    if not pages:
        return
    ttl = int(time.time()) + OVERFLOW_TTL_SECONDS
    pk = f"{PK_OVERFLOW_PREFIX}{cache_hash}"
    for page_idx, page in enumerate(pages):
        is_last = page_idx == len(pages) - 1
        item = {
            "PK": pk,
            "SK": f"{SK_OVERFLOW_PAGE_PREFIX}{page_idx:08d}",
            "models": json.dumps(page),
            "ttl": ttl,
        }
        if is_last:
            item["ddb_lek"] = json.dumps(ddb_lek) if ddb_lek else ""
        # Shard if over 390KB
        _write_with_sharding(table, item)
    # NOTE: sharding splits a single page into sub-items if needed
```

- [ ] **Step 3: Write `cache_get_overflow_page`**

```python
def cache_get_overflow_page(
    table, cache_hash: str, page_idx: int
) -> tuple[list[dict] | None, dict | None, bool]:
    """Read one overflow page. Returns (models_data, ddb_lek_or_None, is_last).
    Returns (None, None, False) if cache miss or expired.
    """
```

- [ ] **Step 4: Write sharding helpers**

```python
def _write_with_sharding(table, item: dict) -> None:
    """Write item, sharding into sub-items if serialized > MAX_ITEM_BYTES."""
    # Check size of "models" field
    # If over limit, split models list and write SHARD#0, SHARD#1 sub-items
    # Otherwise write as-is

def _read_with_sharding(table, pk: str, sk: str) -> dict | None:
    """Read item, reassembling shards if necessary."""
```

- [ ] **Step 5: Write `encode_overflow_token` / `decode_overflow_token`**

```python
def encode_overflow_token(cache_hash: str, page_idx: int) -> str:
    return base64.b64encode(json.dumps({"overflow": cache_hash, "page": page_idx}).encode()).decode()

def is_overflow_token(decoded: dict) -> bool:
    return "overflow" in decoded
```

- [ ] **Step 6: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`

- [ ] **Step 7: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/overflow_cache.py
git commit -m "feat: add overflow cache module for pagination tiebreak"
```

---

### Task 2: Update `_sort_models` for multi-column ordering

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add `_Negate` class and `_negate` helper** (already in file)

Verify the `_Negate` class is present near the top of the file. It should support `__lt__` and `__eq__` for reverse-order comparisons.

- [ ] **Step 2: Update `_sort_models`** (already updated)

Verify `_sort_models` handles multi-column ordering with implicit name ASC tiebreaker for timestamp columns.

- [ ] **Step 3: Add `_needs_secondary_sort` static method**

```python
@staticmethod
def _needs_secondary_sort(order_by: list[str] | None) -> bool:
    if not order_by:
        return False
    if len(order_by) > 1:
        return True
    key = order_by[0].split()[0].lower()
    return key in ("last_updated_timestamp", "timestamp")
```

- [ ] **Step 4: Add `_serialize_model` / `_deserialize_model` helpers**

For caching RegisteredModel objects to/from JSON dicts.

- [ ] **Step 5: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: multi-column sort with tiebreaking for registered models"
```

---

### Task 3: Update `_search_models_paginated` with tie group logic

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add `order_by` parameter to `_search_models_paginated`**

- [ ] **Step 2: Add overflow token detection in `_decode_page_token`**

When token contains `"overflow"` key, return overflow cache data instead of DDB LEK.

- [ ] **Step 3: Implement tie group detection and over-fetch**

After fetching max_results+1 items:
1. If `_needs_secondary_sort` and items[max_results-1].ts == items[max_results].ts:
   - Keep fetching from DDB until timestamp changes or exhausted
2. Sort within tie groups using `_sort_models`
3. Return first max_results

- [ ] **Step 4: Implement overflow caching**

When sorted buffer exceeds max_results:
1. Serialize overflow models
2. Call `cache_put_overflow` with pages of max_results
3. Return overflow token pointing to page 0

- [ ] **Step 5: Implement overflow token consumption**

When token is an overflow token:
1. Read cache page
2. If last page and has ddb_lek → next token resumes DDB pagination
3. If last page and no ddb_lek → no more data
4. If not last page → next token points to next cache page
5. If cache miss (expired) → raise MlflowException

- [ ] **Step 6: Pass `order_by` from `search_registered_models` to `_search_models_paginated`**

- [ ] **Step 7: Run unit tests**

Run: `uv run pytest tests/unit/test_registry_search.py::TestSearchRegisteredModelOrderBy -x -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: pagination-safe tiebreaking with overflow cache"
```

---

### Task 4: Write edge case unit tests

**Files:**
- Modify: `tests/unit/test_registry_search.py`

- [ ] **Step 1: Tests are already written** (from earlier in session)

Verify `TestSearchRegisteredModelOrderBy` exists with:
- `test_order_by_timestamp_asc_with_name_desc_tiebreak`
- `test_order_by_timestamp_desc_implicit_name_asc_tiebreak`
- `test_empty_order_by_defaults_to_name_asc`
- `test_pagination_with_tiebreaking_no_duplicates`
- `test_pagination_tiebreak_at_page_boundary`
- `test_pagination_timestamp_desc_name_desc`

- [ ] **Step 2: Add test for tie group larger than page size**

```python
def test_pagination_tie_group_exceeds_page_size(self, registry_store):
    """When a tie group is larger than max_results, overflow cache is used."""
    # 5 models all with same timestamp, max_results=2
    self._create_models_with_timestamps(
        registry_store,
        [("E", 1), ("D", 1), ("C", 1), ("B", 1), ("A", 1)],
    )
    all_names: list[str] = []
    token = None
    for _ in range(10):
        result = registry_store.search_registered_models(
            order_by=["last_updated_timestamp ASC", "name ASC"],
            max_results=2,
            page_token=token,
        )
        all_names.extend(m.name for m in result)
        token = result.token
        if not token:
            break
    assert all_names == ["A", "B", "C", "D", "E"]
```

- [ ] **Step 3: Add test for cache expiry handling**

```python
def test_overflow_cache_expiry_raises_error(self, registry_store):
    """Expired overflow cache token raises meaningful error."""
    # Create scenario that triggers overflow, get token, expire it, use it
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/unit/test_registry_search.py::TestSearchRegisteredModelOrderBy -x -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_registry_search.py
git commit -m "test: add edge case tests for tiebreaking pagination"
```

---

### Task 5: Remove Cat 13 xfail for order_by and verify

**Files:**
- Modify: `tests/compatibility/test_registry_compat.py`

- [ ] **Step 1: Remove xfail for `test_search_registered_model_order_by`**

Keep xfails for the other two Cat 13 tests (model versions).

- [ ] **Step 2: Run target test**

Run: `uv run pytest tests/compatibility/test_registry_compat.py::test_search_registered_model_order_by -x -v --runxfail`
Expected: PASS

- [ ] **Step 3: Run full compat suite**

Run: `uv run pytest tests/compatibility/test_registry_compat.py -v`
Expected: 34 passed, 2 xfailed.

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`

- [ ] **Step 5: Commit**

```bash
git add tests/compatibility/test_registry_compat.py
git commit -m "test: remove Cat 13 xfail for search_registered_model_order_by"
```
