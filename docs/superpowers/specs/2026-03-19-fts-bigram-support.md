# FTS 2-Character Substring Search via N-gram Substring Search

## Problem

`search_registered_models` with `name LIKE '%RM%'` returns no results because the FTS substring search can't match 2-character substrings. Trigrams require 3+ characters, and the current code uses word-level tokens (which do exact whole-word matching) as its first fallback.

## Design Decisions

**Separate concerns:**
- **WWS (Whole Word Search)** — level `W`, stemmed tokens, for semantic word matching. Not used by `%X%` patterns.
- **NSS (N-gram Substring Search)** — levels `3` (trigrams) and `2` (tail bigrams), character-level, for `%X%` LIKE substring matching.

**Reorder GSI2 SK** to put entity_type before token, enabling prefix queries scoped to entity type.

**Tail bigrams only** — 1 extra index entry per word (last 2 chars). Non-tail bigrams are found by prefix-querying existing trigram entries.

## Key Format Change

### Forward SK and GSI2 SK (entity_type moved before token)

| | Current | Proposed |
|---|---|---|
| Forward SK | `FTS#<level>#<token>#<entity_type>#<entity_id>` | `FTS#<level>#<entity_type>#<token>#<entity_id>` |
| GSI2 SK | `<level>#<token>#<entity_type>#<entity_id>` | `<level>#<entity_type>#<token>#<entity_id>` |

### Reverse SK (unchanged)

`FTS_REV#<entity_type>#<entity_id>#<level>#<token>` — only used for cleanup.

### Example for model `test_for_search_RM4ab` (ulid `01abc`)

```
Forward SK:  FTS#3#M#rm4#01abc       (trigram, reordered)
GSI2 SK:     3#M#rm4#01abc           (trigram, reordered)
Forward SK:  FTS#2#M#ab#01abc        (tail bigram, NEW)
GSI2 SK:     2#M#ab#01abc            (tail bigram, NEW)
Reverse SK:  FTS_REV#M#01abc#3#rm4   (unchanged)
Reverse SK:  FTS_REV#M#01abc#2#ab    (new level, same pattern)
```

## How Queries Work

### `%RM%` (2-char term)

1. Trigram-prefix: `sk_prefix="3#M#rm"` → matches `3#M#rm4#01abc`, `3#M#rm1#...`, etc.
2. Tail bigram: `sk_prefix="2#M#rm#"` → matches if `rm` is the last 2 chars of any word
3. Union results per bigram, intersect across bigrams

### `%ab%` (2-char, end-of-word)

1. Trigram-prefix: `sk_prefix="3#M#ab"` → no match (no trigram starts with `ab`)
2. Tail bigram: `sk_prefix="2#M#ab#"` → matches `2#M#ab#01abc`

### `%search%` (6-char term)

1. Trigrams: `sea` ∩ `ear` ∩ `arc` ∩ `rch` — each queried with `sk_prefix="3#M#<trigram>#"`
2. Bigrams not needed (trigrams cover 3+ char terms)

### `%%` (empty search term)

Special case: fall back to list-all-models. Do not enter NSS path.

## ID Extraction

GSI2 SK format: `<level>#<entity_type>#<token>#<entity_id>`

```python
parts = gsi2sk.split("#")
# parts[0] = level, parts[1] = entity_type, parts[2] = token, parts[3] = entity_id
if len(parts) >= 4 and parts[1] == "M":
    ids.add(parts[3])
```

For trigram-prefix queries (no trailing `#`), the token part may differ from the search term but entity_id is always at index 3.

## Changes

### `src/mlflow_dynamodbstore/dynamodb/fts.py`

1. Add `tokenize_tail_bigrams(text)` — last 2 chars of each word:
   ```python
   def tokenize_tail_bigrams(text: str) -> set[str]:
       words = re.findall(r"[a-z0-9]+", text.lower())
       return {word[-2:] for word in words if len(word) >= 2}
   ```

2. Add `tokenize_bigrams(text)` — all bigrams, query-side only:
   ```python
   def tokenize_bigrams(text: str) -> set[str]:
       words = re.findall(r"[a-z0-9]+", text.lower())
       grams: set[str] = set()
       for word in words:
           for i in range(len(word) - 1):
               grams.add(word[i : i + 2])
       return grams
   ```

3. Register level `"2"` in `_tokens_for_level` using `tokenize_tail_bigrams`.

4. Update default `levels` in `fts_items_for_text` and `fts_diff`:
   ```python
   levels: tuple[str, ...] = ("W", "3", "2")
   ```

5. Reorder Forward SK and GSI2 SK — entity_type before token:
   ```python
   forward_sk = f"{SK_FTS_PREFIX}{level}#{entity_type}#{token}#{entity_id}{field_suffix}"
   gsi2sk_val = f"{level}#{entity_type}#{token}#{entity_id}{field_suffix}"
   ```

### `src/mlflow_dynamodbstore/registry_store.py`

6. Rename `_search_models_by_fts` → `_search_models_by_nss`.

7. Rewrite `_search_models_by_nss` to use only n-gram levels (no word tokens):
   ```python
   def _search_models_by_nss(self, search_term: str) -> list[RegisteredModel]:
       """N-gram Substring Search via GSI2."""
       from mlflow_dynamodbstore.dynamodb.fts import tokenize_bigrams, tokenize_trigrams

       gsi2pk = f"{GSI2_FTS_NAMES_PREFIX}{self._workspace}"

       # Trigrams (3+ char terms)
       trigram_tokens = tokenize_trigrams(search_term)
       if trigram_tokens:
           # query sk_prefix="3#M#<token>#" per token, intersect
           ...

       # Bigrams (2-char terms or fallback)
       bigram_tokens = tokenize_bigrams(search_term)
       if bigram_tokens:
           for token in bigram_tokens:
               # Trigram-prefix: sk_prefix="3#M#<token>" (no trailing #)
               # Tail bigram:   sk_prefix="2#M#<token>#"
               # Union both sets per bigram, intersect across bigrams
               ...
   ```

8. Update ID extraction for new GSI2 SK format:
   ```python
   parts = gsi2sk.split("#")
   if len(parts) >= 4 and parts[1] == "M":
       ids.add(parts[3])
   ```

9. Update `_search_models_by_name_like` — call `_search_models_by_nss` instead of `_search_models_by_fts`.

10. Apply default name ASC sort to NSS results. NSS is candidate generation — the small result set gets sorted. In the non-prefix LIKE path of `search_registered_models`, change:
    ```python
    models = self._sort_models(models, order_by or ["name ASC"])
    ```

### `src/mlflow_dynamodbstore/tracking_store.py`

10. Update any inline GSI2 SK construction for experiment FTS to use new format:
    `<level>#E#<token>#<experiment_id>` instead of `<level>#<token>#E#<experiment_id>`.

### `tests/compatibility/test_registry_compat.py`

11. Remove xfail for `test_search_registered_models` (Cat 14).

### `search_registered_models` — handle `%%`

12. In `_search_models_by_name_like`, add special case: if pattern is `%%` (empty search term after stripping `%`), fall back to `_search_models_by_gsi2()`.

## Write Amplification

Per word of length N (N >= 3):
- Current: (N-2 trigrams + 1 word) × 2 items = 2(N-1) items
- Proposed: (N-2 trigrams + 1 word + 1 tail bigram) × 2 items = 2N items
- Delta: +2 items per word

For a model with 4 words averaging 5 chars: +8 items total.

## Not Backward Compatible

The GSI2 SK reorder changes the key format. Existing FTS entries become invisible to queries. This is acceptable — the database can be dropped and recreated.

## Verification

```bash
uv run pytest tests/compatibility/test_registry_compat.py::test_search_registered_models -x -v --runxfail
uv run pytest tests/compatibility/test_registry_compat.py -v
uv run pytest tests/unit/ -x -q
```
