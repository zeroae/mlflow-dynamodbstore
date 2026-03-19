# Native DynamoDB Sorting & Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement cursor-based pagination and native DynamoDB sorting for `search_registered_models`, fixing 2 compatibility tests (Cat 13 partial).

**Architecture:** Use DynamoDB's `query_page()` with `LastEvaluatedKey` as opaque page tokens (base64-encoded). Select GSI5 for name ordering, GSI2 for timestamp ordering. Post-filter results in Python after DynamoDB returns natively sorted items. Construct `ExclusiveStartKey` from the last returned item when post-filtering trims results.

**Tech Stack:** DynamoDB GSI2/GSI5, `query_page()`, base64/json for token encoding, MLflow `PagedList`.

**Dependency:** Model version pagination/ordering tests depend on PR #28 (prompt filtering + tag filtering in `search_model_versions`). This plan targets only registered model tests.

---

## File Structure

- **Modify:** `src/mlflow_dynamodbstore/registry_store.py` — `search_registered_models`, `_search_models_by_gsi2`, new `_search_models_by_gsi5`, page token helpers, max_results validation
- **Modify:** `tests/compatibility/test_registry_compat.py` — remove xfails for `test_search_registered_model_pagination` and `test_search_registered_model_order_by`
- **Test:** `tests/compatibility/test_registry_compat.py` (vendored tests, no new tests needed)

---

### Task 1: Add page token encode/decode helpers

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add `_encode_page_token` and `_decode_page_token` static methods to `DynamoDBRegistryStore`**

```python
import base64
import json

@staticmethod
def _encode_page_token(last_evaluated_key: dict[str, Any]) -> str:
    """Encode a DynamoDB LastEvaluatedKey as an opaque page token."""
    return base64.b64encode(json.dumps(last_evaluated_key).encode("utf-8")).decode("ascii")

@staticmethod
def _decode_page_token(page_token: str | None) -> dict[str, Any] | None:
    """Decode an opaque page token back to a DynamoDB ExclusiveStartKey."""
    if not page_token:
        return None
    try:
        decoded = base64.b64decode(page_token)
    except Exception:
        raise MlflowException(
            "Invalid page token, could not base64-decode",
            error_code=INVALID_PARAMETER_VALUE,
        )
    try:
        return json.loads(decoded)
    except Exception:
        raise MlflowException(
            f"Invalid page token, decoded value={decoded}",
            error_code=INVALID_PARAMETER_VALUE,
        )
```

Add `import base64` and `import json` to the top-level imports.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/mlflow_dynamodbstore/registry_store.py --fix`

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: add page token encode/decode helpers for DynamoDB cursors"
```

---

### Task 2: Add max_results validation

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add max_results validation to `search_registered_models`**

At the top of `search_registered_models`, before any filter parsing, add:

```python
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
)

if max_results is None:
    max_results = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT
if not isinstance(max_results, int) or max_results < 1:
    raise MlflowException(
        f"Invalid value for request parameter max_results. It must be at most "
        f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
        INVALID_PARAMETER_VALUE,
    )
if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
    raise MlflowException(
        f"Invalid value for request parameter max_results. It must be at most "
        f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
        INVALID_PARAMETER_VALUE,
    )
```

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/mlflow_dynamodbstore/registry_store.py --fix`

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: add max_results validation for search_registered_models"
```

---

### Task 3: Add `_search_models_paginated` using `query_page`

This is the core method that queries a DynamoDB index page-by-page with native sorting, applies Python post-filters, and returns `(results, next_page_token)`.

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add `_build_exclusive_start_key` helper**

This constructs a valid `ExclusiveStartKey` from a DynamoDB item for a given index. Needed when post-filtering trims results and the DynamoDB page boundary cursor is wrong.

```python
@staticmethod
def _build_exclusive_start_key(
    item: dict[str, Any], index_name: str | None
) -> dict[str, Any]:
    """Build an ExclusiveStartKey from an item for the given index."""
    # Main table always needs PK + SK
    key: dict[str, Any] = {"PK": item["PK"], "SK": item["SK"]}
    if index_name is None:
        return key
    # GSI needs table PK+SK plus GSI PK+SK
    # LSI needs table PK+SK plus LSI SK
    from mlflow_dynamodbstore.dynamodb.table import _INDEX_KEY_ATTRS
    idx_pk, idx_sk = _INDEX_KEY_ATTRS[index_name]
    if idx_pk != "PK":
        key[idx_pk] = item[idx_pk]
    key[idx_sk] = item[idx_sk]
    return key
```

- [ ] **Step 2: Add `_search_models_paginated` method**

```python
def _search_models_paginated(
    self,
    pk: str,
    index_name: str,
    sk_prefix: str | None = None,
    scan_forward: bool = True,
    max_results: int = 100,
    page_token: str | None = None,
    filter_fn: Any | None = None,
) -> tuple[list[RegisteredModel], str | None]:
    """Query an index page-by-page, apply filter_fn, return (models, next_token).

    Uses DynamoDB native sorting. filter_fn, if provided, takes a
    RegisteredModel and returns True to keep it.
    """
    exclusive_start_key = self._decode_page_token(page_token)
    results: list[RegisteredModel] = []
    last_item: dict[str, Any] | None = None

    while len(results) < max_results + 1:
        batch_size = (max_results + 1 - len(results)) * 2 if filter_fn else max_results + 1 - len(results)
        items, lek = self._table.query_page(
            pk=pk,
            sk_prefix=sk_prefix,
            index_name=index_name,
            limit=batch_size,
            scan_forward=scan_forward,
            exclusive_start_key=exclusive_start_key,
        )
        for item in items:
            model_name = item.get("name")
            if not model_name:
                continue
            model_ulid = item["PK"].replace(PK_MODEL_PREFIX, "")
            tags = self._get_model_tags(model_ulid)
            rm = _item_to_registered_model(item, tags)
            if filter_fn and not filter_fn(rm):
                continue
            results.append(rm)
            last_item = item
            if len(results) > max_results:
                break

        if lek is None or len(results) > max_results:
            break
        exclusive_start_key = lek

    if len(results) > max_results:
        results = results[:max_results]
        # Build cursor from last kept item
        next_token = self._encode_page_token(
            self._build_exclusive_start_key(last_item, index_name)
        ) if last_item else None
    else:
        next_token = None

    return results, next_token
```

Note: We fetch `max_results + 1` to detect if there's a next page (same pattern as SqlAlchemy store). When post-filtering, we over-fetch by 2x to compensate for filtered-out items.

- [ ] **Step 3: Run linter**

Run: `uv run ruff check src/mlflow_dynamodbstore/registry_store.py --fix`

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: add _search_models_paginated with cursor-based DynamoDB pagination"
```

---

### Task 4: Parse order_by and select index

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Add `_resolve_registered_model_order` method**

This parses the order_by list and returns the index name and scan direction.

```python
@staticmethod
def _resolve_registered_model_order(
    order_by: list[str] | None,
) -> tuple[str, bool]:
    """Determine (index_name, scan_forward) from order_by clauses.

    Returns the index and direction for the primary sort column.
    Default: GSI5 (name ASC).
    """
    from mlflow.utils.search_utils import SearchUtils

    if not order_by:
        return "gsi5", True  # default: name ASC

    # Parse first order_by clause (primary sort)
    attribute, ascending = SearchUtils.parse_order_by_for_search_registered_models(
        order_by[0]
    )

    if attribute == "name":
        return "gsi5", ascending
    if attribute in SearchUtils.VALID_TIMESTAMP_ORDER_BY_KEYS:
        return "gsi2", ascending

    # Fallback to name ASC for unknown attributes
    return "gsi5", True
```

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/mlflow_dynamodbstore/registry_store.py --fix`

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: add _resolve_registered_model_order for index selection"
```

---

### Task 5: Rewrite `search_registered_models` to use paginated queries

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py`

- [ ] **Step 1: Rewrite `search_registered_models`**

Replace the current implementation. The method should:
1. Validate max_results (Task 2)
2. Parse filters into predicates
3. Determine index + direction from order_by (Task 4)
4. Choose query path based on filter (exact name → GSI3, LIKE prefix → GSI5 with sk_prefix, default → selected index)
5. Call `_search_models_paginated` with appropriate params and filter_fn
6. Return `PagedList(results, token)`

Key changes to the query path selection:
- **No filter or LIKE filter:** Use the order_by-selected index (GSI5 or GSI2). For LIKE prefix filters on GSI5, pass `sk_prefix`. For LIKE on GSI2 or general filters, use GSI2 and add a Python `filter_fn` for name matching.
- **Exact name:** Keep using `_search_models_by_name_exact` (returns 0 or 1 result, no pagination needed).
- **Tag predicates and prompt filtering:** Pass as `filter_fn` to `_search_models_paginated`.

```python
def search_registered_models(
    self,
    filter_string: str | None = None,
    max_results: int | None = None,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> list[RegisteredModel]:
    """Search registered models with filter, ordering, and pagination."""
    from mlflow.store.model_registry import (
        SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
    )
    from mlflow.utils.search_utils import SearchModelUtils, SearchUtils

    from mlflow_dynamodbstore.dynamodb.search import (
        FilterPredicate,
        _compare,
    )

    if max_results is None:
        max_results = SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT
    if not isinstance(max_results, int) or max_results < 1:
        raise MlflowException(
            "Invalid value for request parameter max_results. It must be at most "
            f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
            INVALID_PARAMETER_VALUE,
        )
    if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
        raise MlflowException(
            "Invalid value for request parameter max_results. It must be at most "
            f"{SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
            INVALID_PARAMETER_VALUE,
        )

    # Validate order_by clauses
    for clause in order_by or []:
        SearchUtils.parse_order_by_for_search_registered_models(clause)

    # Parse filters
    if filter_string:
        parsed = SearchModelUtils.parse_search_filter(filter_string)
        predicates = [
            FilterPredicate(
                field_type=p["type"], key=p["key"], op=p["comparator"], value=p.get("value"),
            )
            for p in parsed
        ]
    else:
        predicates = []

    name_pred = next(
        (p for p in predicates if p.field_type == "attribute" and p.key == "name"), None,
    )
    tag_preds = [p for p in predicates if p.field_type == "tag" and p.key != IS_PROMPT_TAG_KEY]

    # Exact name lookup — no pagination needed
    if name_pred and name_pred.op == "=":
        models = self._search_models_by_name_exact(name_pred.value)
        if tag_preds:
            models = self._filter_models_by_tags(models, tag_preds, _compare)
        if self._is_querying_prompt(predicates):
            models = [m for m in models if m._is_prompt()]
        else:
            models = [m for m in models if not m._is_prompt()]
        from mlflow.store.entities import PagedList
        return PagedList(models[:max_results], token=None)

    # Determine index and direction from order_by
    index_name, scan_forward = self._resolve_registered_model_order(order_by)

    # Build pk and sk_prefix for the chosen index
    if index_name == "gsi5":
        pk = f"{GSI5_MODEL_NAMES_PREFIX}{self._workspace}"
        # For LIKE prefix filters, use sk_prefix on GSI5
        if name_pred and name_pred.op in ("LIKE", "ILIKE") and name_pred.value.endswith("%") and "%" not in name_pred.value[:-1]:
            sk_prefix = name_pred.value[:-1]
            name_pred = None  # consumed by sk_prefix
        else:
            sk_prefix = None
    else:  # gsi2
        pk = f"{GSI2_MODELS_PREFIX}{self._workspace}"
        sk_prefix = None

    # Build filter_fn combining name filter, tag filter, prompt filter
    import fnmatch as _fnmatch

    def filter_fn(model: RegisteredModel) -> bool:
        # Skip NAME_REV items that appear in GSI5
        if hasattr(model, 'name') and model.name is None:
            return False
        # Name LIKE/ILIKE filter (if not consumed by sk_prefix)
        if name_pred:
            fn_pattern = name_pred.value.replace("%", "*").replace("_", "?")
            if name_pred.op == "ILIKE":
                if not _fnmatch.fnmatch(model.name.lower(), fn_pattern.lower()):
                    return False
            else:
                if not _fnmatch.fnmatch(model.name, fn_pattern):
                    return False
        # Tag filters
        if tag_preds:
            tag_dict: dict[str, str] = {}
            if hasattr(model, "_tags") and isinstance(model._tags, dict):
                tag_dict = model._tags
            elif isinstance(model.tags, dict):
                tag_dict = model.tags
            else:
                for t in model.tags:
                    tag_dict[t.key] = t.value
            for pred in tag_preds:
                actual = tag_dict.get(pred.key)
                if not _compare(actual, pred.op, pred.value):
                    return False
        # Prompt filter
        if self._is_querying_prompt(predicates):
            if not model._is_prompt():
                return False
        else:
            if model._is_prompt():
                return False
        return True

    models, next_token = self._search_models_paginated(
        pk=pk,
        index_name=index_name,
        sk_prefix=sk_prefix,
        scan_forward=scan_forward,
        max_results=max_results,
        page_token=page_token,
        filter_fn=filter_fn,
    )

    from mlflow.store.entities import PagedList
    return PagedList(models, next_token)
```

Important: GSI5 contains both model name items (SK=`{name}#{ulid}`) and NAME_REV items (SK=`REV#{reversed_name}#{ulid}`). The `filter_fn` must skip NAME_REV items. Check the item's `name` field — NAME_REV items have the same `name` field, but their SK starts with "REV#". To distinguish, check that `item.get(GSI5_SK, "").startswith("REV#")` is False. This filtering should happen in `_search_models_paginated` before constructing the RegisteredModel, or use `sk_prefix` to exclude REV items.

Actually, a simpler approach: when querying GSI5 with no prefix filter, skip items whose GSI5_SK starts with "REV#". Add a check in `_search_models_paginated` or pass an `sk_prefix` that excludes REV items. Since model names can't start with "REV#" in practice, we could just check `not item.get(GSI5_SK, "").startswith("REV#")` in the item loop.

- [ ] **Step 2: Run the two target tests**

Run: `uv run pytest tests/compatibility/test_registry_compat.py::test_search_registered_model_pagination tests/compatibility/test_registry_compat.py::test_search_registered_model_order_by -x -v --runxfail`

Expected: Both PASS.

- [ ] **Step 3: Run full compatibility suite**

Run: `uv run pytest tests/compatibility/test_registry_compat.py -v`

Expected: No regressions — all previously passing tests still pass.

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`

Expected: All 845 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: rewrite search_registered_models with native DynamoDB sorting and cursor-based pagination"
```

---

### Task 6: Update xfails in test_registry_compat.py

**Files:**
- Modify: `tests/compatibility/test_registry_compat.py`

- [ ] **Step 1: Remove xfails for the two now-passing tests**

Remove `test_search_registered_model_pagination` and `test_search_registered_model_order_by` from the `_xfail_search_order` block. Keep the 4 model version xfails.

```python
# --- Category 13: search ordering and pagination broken ---
_xfail_search_order = pytest.mark.xfail(
    reason="DynamoDB store search ordering and pagination incomplete"
)
test_search_model_versions = _xfail_search_order(test_search_model_versions)
test_search_model_versions_by_tag = _xfail_search_order(test_search_model_versions_by_tag)
test_search_model_versions_order_by_simple = _xfail_search_order(
    test_search_model_versions_order_by_simple
)
test_search_model_versions_pagination = _xfail_search_order(test_search_model_versions_pagination)
```

- [ ] **Step 2: Run full compatibility suite to confirm**

Run: `uv run pytest tests/compatibility/test_registry_compat.py tests/compatibility/test_registry_workspace_compat.py tests/compatibility/test_registry_contract.py -v`

Expected: 2 more passing tests, no xpassed, no failures.

- [ ] **Step 3: Commit**

```bash
git add tests/compatibility/test_registry_compat.py
git commit -m "test: remove xfails for search_registered_model pagination and order_by"
```
