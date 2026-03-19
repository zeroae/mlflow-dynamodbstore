# FTS Bigram Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable 2-character substring search in `search_registered_models` by adding tail bigram indexing and trigram-prefix queries, with GSI2 SK reorder for entity-type-scoped prefix queries.

**Architecture:** Reorder GSI2 SK from `<level>#<token>#<entity_type>#<entity_id>` to `<level>#<entity_type>#<token>#<entity_id>`. Add `tokenize_tail_bigrams` for indexing (1 entry per word) and `tokenize_bigrams` for querying. Rename `_search_models_by_fts` to `_search_models_by_nss` using only n-gram levels (trigrams + bigrams), not word tokens.

**Tech Stack:** DynamoDB GSI2, boto3, MLflow SearchModelUtils

**Spec:** `docs/superpowers/specs/2026-03-19-fts-bigram-support.md`

---

## File Structure

- **Modify:** `src/mlflow_dynamodbstore/dynamodb/fts.py` — add `tokenize_tail_bigrams`, `tokenize_bigrams`, register level `"2"`, reorder Forward SK and GSI2 SK
- **Modify:** `src/mlflow_dynamodbstore/registry_store.py` — rename `_search_models_by_fts` → `_search_models_by_nss`, rewrite to use trigram + bigram queries, update ID extraction, handle `%%`, default sort
- **Modify:** `src/mlflow_dynamodbstore/tracking_store.py` — update inline FTS key construction to new GSI2 SK format
- **Modify:** `tests/compatibility/test_registry_compat.py` — remove Cat 14 xfail
- **Test:** vendored `test_search_registered_models` (Cat 14)

---

### Task 1: Add bigram tokenizers to fts.py

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/fts.py:197-221`

- [ ] **Step 1: Add `tokenize_tail_bigrams` and `tokenize_bigrams`**

After `tokenize_trigrams` (line 204), add:

```python
def tokenize_tail_bigrams(text: str) -> set[str]:
    """Last 2 characters of each word — covers end-of-word bigram positions."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word[-2:] for word in words if len(word) >= 2}


def tokenize_bigrams(text: str) -> set[str]:
    """All character bigrams of the search term (query-side only)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams: set[str] = set()
    for word in words:
        for i in range(len(word) - 1):
            grams.add(word[i : i + 2])
    return grams
```

- [ ] **Step 2: Register level `"2"` in `_tokens_for_level`**

At `fts.py:215-221`, add the `"2"` case:

```python
def _tokens_for_level(level: str, text: str) -> set[str]:
    if level == "W":
        return tokenize_words(text)
    if level == "3":
        return tokenize_trigrams(text)
    if level == "2":
        return tokenize_tail_bigrams(text)
    raise ValueError(f"Unknown FTS level: {level!r}")
```

- [ ] **Step 3: Update default `levels` tuple**

Change the default in `fts_items_for_text` (line 235) and `fts_diff` (line 282):

```python
levels: tuple[str, ...] = ("W", "3", "2"),
```

- [ ] **Step 4: Run existing unit tests**

Run: `uv run pytest tests/unit/ -x -q -k fts`
Expected: All pass (new functions not yet called by tests).

- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/fts.py
git commit -m "feat(fts): add tail bigram tokenizer and level '2' support"
```

---

### Task 2: Reorder Forward SK and GSI2 SK in fts.py

**Files:**
- Modify: `src/mlflow_dynamodbstore/dynamodb/fts.py:258-271`

- [ ] **Step 1: Reorder entity_type before token in Forward SK and GSI2 SK**

At line 259, change:
```python
# Before:
forward_sk = f"{SK_FTS_PREFIX}{level}#{token}#{entity_type}#{entity_id}{field_suffix}"
# After:
forward_sk = f"{SK_FTS_PREFIX}{level}#{entity_type}#{token}#{entity_id}{field_suffix}"
```

At line 269, change:
```python
# Before:
gsi2sk_val = f"{level}#{token}#{entity_type}#{entity_id}{field_suffix}"
# After:
gsi2sk_val = f"{level}#{entity_type}#{token}#{entity_id}{field_suffix}"
```

Reverse SK (line 261) stays unchanged:
```python
reverse_sk = f"{SK_FTS_REV_PREFIX}{entity_type}#{entity_id}{field_suffix}#{level}#{token}"
```

- [ ] **Step 2: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass (moto tables are recreated per test).

- [ ] **Step 3: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/fts.py
git commit -m "feat(fts): reorder GSI2 SK to put entity_type before token"
```

---

### Task 3: Update tracking_store.py inline FTS key construction

**Files:**
- Modify: `src/mlflow_dynamodbstore/tracking_store.py:564,572,575,800,819`

- [ ] **Step 1: Update `rename_experiment` FTS key construction (lines 564, 572, 575)**

```python
# Line 564 — delete forward item (old format → new format):
forward_sk = f"{SK_FTS_PREFIX}{lvl}#E#{tok}#{experiment_id}"

# Line 572 — write forward item:
forward_sk = f"{SK_FTS_PREFIX}{lvl}#E#{tok}#{experiment_id}"

# Line 575 — GSI2 SK:
gsi2sk_val = f"{lvl}#E#{tok}#{experiment_id}"
```

- [ ] **Step 2: Update experiment search FTS queries (lines 800, 819)**

```python
# Line 800 — word query:
sk_prefix=f"W#E#{token}#",

# Line 819 — trigram query:
sk_prefix=f"3#E#{token}#",
```

- [ ] **Step 3: Update ID extraction (lines 807-811, 825-829)**

```python
# New pattern: <level>#<entity_type>#<token>#<entity_id>
parts = gsi2sk.split("#")
if len(parts) >= 4 and parts[1] == "E":
    ids.add(parts[3])
```

- [ ] **Step 4: Update any other inline FTS key construction**

Check line 1972-1977 (`_update_fts_items_for_entity`):
```python
# Line 1972:
forward_sk = f"{SK_FTS_PREFIX}{lvl}#{entity_prefix}#{tok}"
# Line 1977:
forward[GSI2_SK] = f"{lvl}#{entity_prefix}#{tok}"
```

Wait — `entity_prefix` here is like `E#<experiment_id>` or `M#<model_ulid>`. The current format is `{lvl}#{tok}#{entity_prefix}`. The new format is `{lvl}#{entity_prefix}#{tok}`. So:

```python
# Before:
forward_sk = f"{SK_FTS_PREFIX}{lvl}#{tok}#{entity_prefix}"
forward[GSI2_SK] = f"{lvl}#{tok}#{entity_prefix}"
# After:
forward_sk = f"{SK_FTS_PREFIX}{lvl}#{entity_prefix}#{tok}"
forward[GSI2_SK] = f"{lvl}#{entity_prefix}#{tok}"
```

- [ ] **Step 5: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/mlflow_dynamodbstore/tracking_store.py
git commit -m "feat(tracking): update FTS key format to new entity_type#token order"
```

---

### Task 4: Rename and rewrite `_search_models_by_fts` → `_search_models_by_nss`

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py:599,609-669`

- [ ] **Step 1: Update caller in `_search_models_by_name_like` (line 599)**

```python
# Before:
return self._search_models_by_fts(search_term)
# After:
return self._search_models_by_nss(search_term)
```

- [ ] **Step 2: Rewrite the method**

Replace `_search_models_by_fts` (lines 609-669) with:

```python
def _search_models_by_nss(self, search_term: str) -> list[RegisteredModel]:
    """N-gram Substring Search — find models by character n-gram intersection via GSI2."""
    from mlflow_dynamodbstore.dynamodb.fts import tokenize_bigrams, tokenize_trigrams

    gsi2pk = f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}"

    def _query_ids(sk_prefix: str) -> set[str]:
        """Query GSI2 and extract model ULIDs from results."""
        fts_items = self._table.query(
            pk=gsi2pk, sk_prefix=sk_prefix, index_name="gsi2",
        )
        ids: set[str] = set()
        for item in fts_items:
            gsi2sk = item.get(GSI2_SK, "")
            parts = gsi2sk.split("#")
            # Pattern: <level>#M#<token>#<model_ulid>
            if len(parts) >= 4 and parts[1] == "M":
                ids.add(parts[3])
        return ids

    # Trigrams: exact match for 3+ char tokens
    trigram_tokens = tokenize_trigrams(search_term)
    if trigram_tokens:
        model_ulid_sets: list[set[str]] = []
        for token in trigram_tokens:
            model_ulid_sets.append(_query_ids(f"3#M#{token}#"))
        if model_ulid_sets:
            result_ids = model_ulid_sets[0]
            for s in model_ulid_sets[1:]:
                result_ids &= s
            if result_ids:
                return self._fetch_models_by_ulids(list(result_ids))

    # Bigrams: trigram-prefix + tail-bigram union, for 2-char tokens
    bigram_tokens = tokenize_bigrams(search_term)
    if bigram_tokens:
        model_ulid_sets = []
        for token in bigram_tokens:
            # Union: trigram-prefix (covers non-tail positions) + tail bigram (covers end-of-word)
            ids = _query_ids(f"3#M#{token}") | _query_ids(f"2#M#{token}#")
            model_ulid_sets.append(ids)
        if model_ulid_sets:
            result_ids = model_ulid_sets[0]
            for s in model_ulid_sets[1:]:
                result_ids &= s
            if result_ids:
                return self._fetch_models_by_ulids(list(result_ids))

    return []
```

Key differences from old `_search_models_by_fts`:
- No word-level queries (WWS is for whole-word search, not substring)
- New GSI2 SK format: `<level>#M#<token>#<entity_id>` (entity_type before token)
- ID extraction: `parts[3]` instead of scanning for `"M"` marker
- Bigram fallback with trigram-prefix (`3#M#{token}` no trailing `#`) + tail-bigram union

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: rename _search_models_by_fts to _search_models_by_nss with bigram support"
```

---

### Task 5: Handle `%%` and default sort in search_registered_models

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py:593-600,507`

- [ ] **Step 1: Handle empty search term in `_search_models_by_name_like`**

At line 596 (the `%word%` branch), add a guard for empty search term:

```python
# Contains pattern (%word%): use NSS via GSI2
if pattern.startswith("%") and pattern.endswith("%"):
    search_term = pattern.strip("%")
    if not search_term:
        return self._search_models_by_gsi2()
    return self._search_models_by_nss(search_term)
```

- [ ] **Step 2: Apply default name ASC sort for non-prefix LIKE path**

In `search_registered_models`, the non-prefix LIKE path (around line 507):

```python
# Before:
models = self._sort_models(models, order_by)
# After:
models = self._sort_models(models, order_by or ["name ASC"])
```

- [ ] **Step 3: Run the target test**

Run: `uv run pytest tests/compatibility/test_registry_compat.py::test_search_registered_models -x -v --runxfail`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/mlflow_dynamodbstore/registry_store.py
git commit -m "feat: handle empty search term and default name ASC sort for LIKE path"
```

---

### Task 6: Update registry_store.py FTS ID extraction for existing callers

**Files:**
- Modify: `src/mlflow_dynamodbstore/registry_store.py:620-662`

The old `_search_models_by_fts` is now `_search_models_by_nss` (Task 4), but there may be other places in registry_store.py that construct or parse FTS keys with the old format. Check and update:

- [ ] **Step 1: Verify no other FTS key construction in registry_store.py**

The `create_registered_model`, `rename_registered_model`, and `delete_registered_model` methods call `fts_items_for_text` from `fts.py` — those are updated by Task 2. No inline FTS key construction in registry_store.py.

Verify the `rename_registered_model` FTS cleanup code (around lines 320-335) which queries `SK_FTS_REV_PREFIX` items — the reverse SK format is unchanged, so this should work.

- [ ] **Step 2: Run full compat suite**

Run: `uv run pytest tests/compatibility/test_registry_compat.py tests/compatibility/test_registry_workspace_compat.py tests/compatibility/test_registry_contract.py -v`
Expected: No regressions. `test_search_registered_models` should now pass (still xfailed though).

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All 845 tests pass.

- [ ] **Step 4: Commit** (only if changes needed)

---

### Task 7: Remove Cat 14 xfail

**Files:**
- Modify: `tests/compatibility/test_registry_compat.py`

- [ ] **Step 1: Remove the xfail**

Remove the Cat 14 block:
```python
# --- Category 14: infix LIKE patterns not supported ---
_xfail_like = pytest.mark.xfail(
    reason="DynamoDB store only supports prefix LIKE patterns, not infix '%X%'"
)
test_search_registered_models = _xfail_like(test_search_registered_models)
```

- [ ] **Step 2: Run full compat suite**

Run: `uv run pytest tests/compatibility/test_registry_compat.py tests/compatibility/test_registry_workspace_compat.py tests/compatibility/test_registry_contract.py -v`
Expected: 44 passed, 6 xfailed, no failures.

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest tests/unit/ -x -q`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/compatibility/test_registry_compat.py
git commit -m "test: remove Cat 14 xfail — infix LIKE now supported via bigram search"
```
