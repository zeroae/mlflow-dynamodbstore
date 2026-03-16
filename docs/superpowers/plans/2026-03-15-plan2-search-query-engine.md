# Plan 2: Search & Query Engine — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full search/filter/order_by support to the tracking store and model registry, including the MLflow filter parser adapter, FTS (word + trigram), RANK/DLINK query integration, pagination, and tag denormalization configuration.

**Architecture:** MLflow's filter grammar is SQL-like (`AND`-only, no `OR`). We parse it and translate to a DynamoDB query plan: select the best index (LSI/GSI/RANK/DLINK/FTS), apply key conditions for pushable predicates, use FilterExpressions for denormalized tags, and post-filter in Python for remaining predicates. Pagination uses opaque base64 tokens encoding DynamoDB's `LastEvaluatedKey`.

**Tech Stack:** Same as Plan 1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-15-mlflow-dynamodbstore-design.md`

**Depends on:** Plan 1 (foundation, CRUD, key builders, table client, provisioner)

---

## File Structure (new/modified files only)

```
src/mlflow_dynamodbstore/
├── dynamodb/
│   ├── search.py                   # NEW: Filter parser adapter, query planner, pagination
│   └── fts.py                      # NEW: Word tokenizer, trigram tokenizer, FTS query builder
├── tracking_store.py               # MODIFY: Wire search.py into _search_runs, search_experiments
├── registry_store.py               # MODIFY: Wire search.py into search_registered_models, search_model_versions

tests/
├── unit/
│   ├── dynamodb/
│   │   ├── test_search.py          # NEW: Filter parser, query planner tests
│   │   └── test_fts.py             # NEW: Tokenizer tests (word + trigram)
│   ├── test_tracking_search.py     # NEW: search_runs, search_experiments with filters
│   └── test_registry_search.py     # NEW: search_registered_models, search_model_versions
├── integration/
│   ├── test_search_runs.py         # NEW: End-to-end search with moto server
│   └── test_search_models.py       # NEW: End-to-end model search with moto server
```

---

## Chunk 1: FTS Tokenizers

### Task 1: Word tokenizer (stemmed)

**Files:**
- Create: `src/mlflow_dynamodbstore/dynamodb/fts.py`
- Create: `tests/unit/dynamodb/test_fts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/dynamodb/test_fts.py
from mlflow_dynamodbstore.dynamodb.fts import tokenize_words, tokenize_trigrams

class TestWordTokenizer:
    def test_basic_tokenization(self):
        tokens = tokenize_words("The ChatModel returned an error")
        assert "chatmodel" not in tokens  # lowercased, but "chatmodel" may or may not stem
        assert "error" in tokens or "error" == list(tokens)[0]  # stemmed

    def test_stop_words_removed(self):
        tokens = tokenize_words("the a an is in on at to for")
        assert len(tokens) == 0

    def test_short_words_removed(self):
        tokens = tokenize_words("I a x go")
        # "I", "a", "x" are <= 1 char, removed. "go" stays.
        assert len(tokens) == 1

    def test_stemming(self):
        t1 = tokenize_words("errors")
        t2 = tokenize_words("error")
        t3 = tokenize_words("errored")
        assert t1 == t2 == t3  # all stem to same token

    def test_alphanumeric_only(self):
        tokens = tokenize_words("gpt-4-turbo v2.0")
        assert "gpt" in tokens
        assert "turbo" in tokens
        assert "v2" in tokens or "0" not in tokens  # splits on non-alnum

    def test_case_insensitive(self):
        t1 = tokenize_words("Pipeline")
        t2 = tokenize_words("pipeline")
        assert t1 == t2
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement `tokenize_words`**

```python
# src/mlflow_dynamodbstore/dynamodb/fts.py
"""Full-text search tokenizers: word-level (stemmed) and trigram-level."""
from __future__ import annotations

import re

import snowballstemmer

STOP_WORDS = frozenset({
    "the", "a", "an", "is", "in", "on", "at", "to", "for",
    "of", "and", "or", "not", "it", "this", "that", "with",
    "be", "has", "have", "had", "do", "does", "did", "but",
    "if", "no", "so", "as", "by", "from", "are", "was", "were",
})
_stemmer = snowballstemmer.stemmer("english")


def tokenize_words(text: str) -> set[str]:
    """Stemmed whole-word tokens for LIKE '%complete_word%' matches."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return set(_stemmer.stemWords(words))
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add src/mlflow_dynamodbstore/dynamodb/fts.py tests/unit/dynamodb/test_fts.py
git commit -m "feat: add word-level FTS tokenizer with stemming"
```

### Task 2: Trigram tokenizer

- [ ] **Step 1: Write failing tests**

```python
class TestTrigramTokenizer:
    def test_basic_trigrams(self):
        trigrams = tokenize_trigrams("pipeline")
        assert "pip" in trigrams
        assert "ipe" in trigrams
        assert "pel" in trigrams
        assert "eli" in trigrams
        assert "lin" in trigrams
        assert "ine" in trigrams
        assert len(trigrams) == 6

    def test_short_word_no_trigrams(self):
        trigrams = tokenize_trigrams("ab")
        assert len(trigrams) == 0

    def test_three_char_word_one_trigram(self):
        trigrams = tokenize_trigrams("foo")
        assert trigrams == {"foo"}

    def test_multi_word(self):
        trigrams = tokenize_trigrams("foo bar")
        assert "foo" in trigrams
        assert "bar" in trigrams

    def test_case_insensitive(self):
        t1 = tokenize_trigrams("Pipeline")
        t2 = tokenize_trigrams("pipeline")
        assert t1 == t2
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement `tokenize_trigrams`**

```python
def tokenize_trigrams(text: str) -> set[str]:
    """Character trigrams for LIKE '%partial%' matches."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams: set[str] = set()
    for word in words:
        for i in range(len(word) - 2):
            grams.add(word[i : i + 3])
    return grams
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add trigram FTS tokenizer"
```

### Task 3: FTS key builders and query helpers

- [ ] **Step 1: Write failing tests**

Test FTS forward/reverse key generation for all entity types (experiment name, run name, run param, run tag, trace tag, trace metadata, assessment). Test the `fts_keys_for_write` helper that returns both forward + reverse items for a given text + entity. Test `fts_diff` that computes tokens to add/remove given old and new text.

- [ ] **Step 2: Implement FTS key builders**

Functions: `fts_keys_for_write(pk, entity_type, entity_id, field, text, levels)` returns list of items (forward + reverse, word + trigram based on config). `fts_diff(old_text, new_text, levels)` returns `(tokens_to_add, tokens_to_delete)`.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add FTS key builders and diff helpers"
```

### Task 4: Wire FTS into tracking store writes

Modify `tracking_store.py` to write FTS items during:
- `create_experiment` / `rename_experiment` (experiment name tokens)
- `create_run` / `update_run_info` (run name tokens)
- `log_batch` (param + tag tokens)
- `set_tag` / `delete_tag` (tag tokens with diff)

- [ ] **Step 1: Write failing tests**

Test that after creating an experiment, FTS items exist in DynamoDB. Test that after renaming, old tokens are deleted and new tokens are written.

- [ ] **Step 2: Modify tracking_store.py to call FTS helpers on writes**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: wire FTS writes into tracking store CRUD operations"
```

### Task 5: Wire FTS into registry store writes

Modify `registry_store.py` to write FTS items for model names on `create_registered_model` / `rename_registered_model`.

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Modify registry_store.py**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: wire FTS writes into registry store CRUD operations"
```

---

## Chunk 2: Filter Parser & Query Planner

### Task 6: MLflow filter parser adapter

**Files:**
- Create: `src/mlflow_dynamodbstore/dynamodb/search.py`
- Create: `tests/unit/dynamodb/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/dynamodb/test_search.py
from mlflow_dynamodbstore.dynamodb.search import parse_filter, FilterPredicate

class TestFilterParser:
    def test_empty_filter(self):
        predicates = parse_filter("")
        assert predicates == []

    def test_attribute_equals(self):
        predicates = parse_filter("status = 'FINISHED'")
        assert len(predicates) == 1
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "status"
        assert predicates[0].op == "="
        assert predicates[0].value == "FINISHED"

    def test_metric_comparison(self):
        predicates = parse_filter("metric.accuracy > 0.9")
        assert predicates[0].field_type == "metric"
        assert predicates[0].key == "accuracy"
        assert predicates[0].op == ">"
        assert predicates[0].value == 0.9

    def test_param_like(self):
        predicates = parse_filter("param.model LIKE '%transformer%'")
        assert predicates[0].field_type == "param"
        assert predicates[0].op == "LIKE"
        assert predicates[0].value == "%transformer%"

    def test_tag_equals(self):
        predicates = parse_filter("tag.env = 'prod'")
        assert predicates[0].field_type == "tag"

    def test_compound_and(self):
        predicates = parse_filter("metric.acc > 0.9 AND param.lr = '0.01'")
        assert len(predicates) == 2

    def test_dataset_filter(self):
        predicates = parse_filter("dataset.name = 'my_data'")
        assert predicates[0].field_type == "dataset"
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement filter parser**

Use MLflow's existing `SearchUtils` to parse the filter string into a structured list of predicates. Our `parse_filter` is an adapter that normalizes MLflow's internal representation into our `FilterPredicate` dataclass for the query planner.

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add MLflow filter parser adapter"
```

### Task 7: Query planner

- [ ] **Step 1: Write failing tests**

```python
from mlflow_dynamodbstore.dynamodb.search import QueryPlan, plan_query

class TestQueryPlanner:
    def test_no_filter_uses_base_sk(self):
        plan = plan_query(entity="run", predicates=[], order_by=None)
        assert plan.index is None  # base table SK
        assert plan.scan_forward is False  # default DESC

    def test_status_filter_uses_lsi3(self):
        preds = parse_filter("status = 'FINISHED'")
        plan = plan_query(entity="run", predicates=preds, order_by=None)
        assert plan.index == "lsi3"
        assert plan.key_condition_prefix == "FINISHED#"

    def test_metric_order_uses_rank(self):
        plan = plan_query(entity="run", predicates=[], order_by=["metric.accuracy DESC"])
        assert plan.strategy == "rank"
        assert plan.rank_key == "accuracy"

    def test_dataset_filter_uses_dlink(self):
        preds = parse_filter("dataset.name = 'my_data'")
        plan = plan_query(entity="run", predicates=preds, order_by=None)
        assert plan.strategy == "dlink"

    def test_like_uses_fts(self):
        preds = parse_filter("run_name LIKE '%pipeline%'")
        plan = plan_query(entity="run", predicates=preds, order_by=None)
        assert plan.strategy == "fts"

    def test_tag_filter_denormalized_uses_filter_expression(self):
        preds = parse_filter("tag.mlflow.user = 'alice'")
        plan = plan_query(
            entity="run", predicates=preds, order_by=None,
            denormalized_patterns=["mlflow.*"],
        )
        assert plan.filter_expressions  # uses FilterExpression, not BatchGetItem
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement query planner**

`plan_query()` analyzes predicates and order_by to select the optimal execution strategy:
1. Check if any predicate can use a specialized index (RANK for metric/param, DLINK for dataset, FTS for LIKE)
2. Check if order_by maps to an LSI
3. Remaining predicates become FilterExpressions (for denormalized tags) or post-filters (for BatchGetItem patterns)
4. Return a `QueryPlan` dataclass with index, key conditions, filter expressions, post-filters, and pagination config

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add DynamoDB query planner for MLflow search"
```

### Task 8: Pagination engine

- [ ] **Step 1: Write failing tests**

```python
from mlflow_dynamodbstore.dynamodb.search import encode_page_token, decode_page_token

class TestPagination:
    def test_encode_decode_round_trip(self):
        token_data = {
            "lek": {"PK": {"S": "EXP#01JQ"}, "SK": {"S": "R#01JR"}},
            "exp_idx": 0,
            "accumulated": 25,
        }
        token = encode_page_token(token_data)
        assert isinstance(token, str)
        decoded = decode_page_token(token)
        assert decoded == token_data

    def test_none_token(self):
        assert decode_page_token(None) is None
        assert decode_page_token("") is None
```

- [ ] **Step 2: Implement pagination**

Base64-encoded JSON. Include `ExclusiveStartKey` (DynamoDB's `LastEvaluatedKey`), experiment index for multi-experiment queries, and accumulated result count.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add pagination token encoding/decoding"
```

---

## Chunk 3: Wire Search into Stores

### Task 9: search_runs implementation

Modify `tracking_store.py`: replace the basic `_search_runs` stub with the full implementation using the query planner.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tracking_search.py
class TestSearchRuns:
    def test_search_by_status(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.update_run_info(run1.info.run_id, "FINISHED", end_time=2000, run_name="r1")
        run2 = tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        runs, _ = tracking_store._search_runs([exp_id], "status = 'FINISHED'", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 1
        assert runs[0].info.run_id == run1.info.run_id

    def test_search_order_by_metric(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        run2 = tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="r2")
        tracking_store.log_batch(run1.info.run_id, metrics=[Metric("acc", 0.8, 0, 0)], params=[], tags=[])
        tracking_store.log_batch(run2.info.run_id, metrics=[Metric("acc", 0.95, 0, 0)], params=[], tags=[])
        runs, _ = tracking_store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 100, ["metric.acc DESC"], None)
        assert runs[0].info.run_id == run2.info.run_id  # 0.95 first

    def test_search_by_tag_denormalized(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.set_tag(run1.info.run_id, RunTag("mlflow.user", "alice"))
        runs, _ = tracking_store._search_runs([exp_id], "tag.mlflow.user = 'alice'", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 1

    def test_search_by_dataset(self, tracking_store):
        # Create run, log_inputs, then search by dataset.name
        ...

    def test_search_by_run_name_like(self, tracking_store):
        # Create runs with names, search with LIKE '%pipeline%'
        ...

    def test_search_pagination(self, tracking_store):
        # Create 30 runs, search with max_results=10, verify page_token works
        ...

    def test_multi_experiment_search(self, tracking_store):
        # Create runs in 2 experiments, search both, verify merge
        ...
```

- [ ] **Step 2: Implement `_search_runs`**

1. Parse filter_string via `parse_filter`
2. Plan query via `plan_query`
3. Execute plan: select index, apply key conditions, filter expressions
4. For each experiment_id, run the query (parallel if multiple)
5. Post-filter non-pushable predicates in Python
6. Apply order_by (if not handled by index)
7. Handle pagination tokens
8. Return `(runs, next_page_token)`

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: implement full _search_runs with filter parser and query planner"
```

### Task 10: search_experiments full implementation

Upgrade `search_experiments` from the Plan 1 basic version to support `filter_string` and `order_by`.

- [ ] **Step 1: Write failing tests**

Test filter by name (LIKE prefix/suffix via GSI5, LIKE '%word%' via FTS on GSI2), filter by tag, order by name/creation_time/last_update_time.

- [ ] **Step 2: Implement using query planner**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: upgrade search_experiments with full filter/order_by support"
```

### Task 11: search_registered_models and search_model_versions

- [ ] **Step 1: Write failing tests**

Test filter by name (LIKE prefix/suffix, LIKE '%word%'), filter by tag, order by name/last_update_time. For model versions: filter by run_id (GSI1 or LSI5), order by version/creation_time/last_update_time.

- [ ] **Step 2: Implement using query planner**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add full search to registry store"
```

---

## Chunk 4: Tag Denormalization Config + NAME_REV + Integration

### Task 12: Tag denormalization config (CONFIG item + pattern matching)

- [ ] **Step 1: Write failing tests**

Test reading denormalize patterns from CONFIG item, merging global + per-experiment patterns, `fnmatch` glob matching, `mlflow.*` always present.

- [ ] **Step 2: Implement pattern reader**

Read `CONFIG#DENORMALIZE_TAGS` and `EXP#<id>#E#DENORMALIZE_TAGS` items, merge, cache. Provide `should_denormalize(experiment_id, tag_key) -> bool`.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add tag denormalization pattern config and matching"
```

### Task 13: NAME_REV materialized items for GSI5 suffix ILIKE

- [ ] **Step 1: Write failing tests**

Test that `create_experiment` writes NAME_REV item with GSI5 reversed name. Test that `rename_experiment` updates NAME_REV. Test GSI5 suffix query `begins_with("REV#...")`.

- [ ] **Step 2: Add NAME_REV writes to create/rename in tracking_store.py and registry_store.py**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add NAME_REV materialized items for suffix ILIKE"
```

### Task 14: FTS trigram config (CONFIG item)

- [ ] **Step 1: Write failing tests**

Test reading `CONFIG#FTS_TRIGRAM_FIELDS`, default fields list, entity names always have trigrams.

- [ ] **Step 2: Implement trigram config reader**

`should_trigram(field_type) -> bool`. Entity names always return True. Other fields check the CONFIG item.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add FTS trigram field configuration"
```

### Task 15: Cross-partition FTS via GSI2

- [ ] **Step 1: Write failing tests**

Test that experiment name FTS items have `gsi2pk`/`gsi2sk` attributes. Test GSI2 query `FTS_NAMES#<ws>` returns matching experiments by word and trigram.

- [ ] **Step 2: Add GSI2 attributes to experiment/model name FTS items**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add cross-partition FTS via GSI2 for experiment/model names"
```

### Task 16: Search integration tests

**Files:**
- Create: `tests/integration/test_search_runs.py`
- Create: `tests/integration/test_search_models.py`

- [ ] **Step 1: Write integration tests**

End-to-end search scenarios via moto server:
- Create experiment + 50 runs with varied metrics/params/tags → search with filters → verify results
- Pagination: search with max_results=10, follow page tokens
- Order by metric DESC → verify RANK item ordering
- LIKE '%word%' → verify FTS query
- Dataset filter → verify DLINK query
- Tag filter (denormalized) → verify FilterExpression
- Multi-experiment search → verify merge

- [ ] **Step 2: Run integration tests**

```bash
uv run pytest tests/integration/test_search_runs.py tests/integration/test_search_models.py -v
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test: add search integration tests"
```

### Task 17: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --cov=mlflow_dynamodbstore --cov-report=term-missing
```

- [ ] **Step 2: Run linters**

```bash
uv run ruff check src/ tests/
uv run mypy src/mlflow_dynamodbstore/
```

- [ ] **Step 3: Commit any fixes**

```bash
git commit -m "chore: fix lint and type errors from Plan 2 verification"
```
