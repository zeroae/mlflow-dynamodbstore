# Plan 2: Search & Query Engine — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full search/filter/order_by/pagination support to the tracking store and model registry, including the MLflow filter parser adapter, FTS (word + trigram), RANK/DLINK query integration, tag denormalization configuration, and NAME_REV items for suffix ILIKE.

**Architecture:** MLflow's filter grammar is SQL-like (`AND`-only, no `OR`). We parse it and translate to a DynamoDB query plan: select the best index (LSI/GSI/RANK/DLINK/FTS), apply key conditions for pushable predicates, use FilterExpressions for denormalized tags, and post-filter in Python for remaining predicates. Pagination uses opaque base64 tokens encoding DynamoDB's `LastEvaluatedKey`.

**Tech Stack:** Same as Plan 1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-15-mlflow-dynamodbstore-design.md`

**Depends on:** Plan 1 (foundation, CRUD, key builders, table client, provisioner)

**What Plan 1 already implemented:**
- `search_experiments`: Basic GSI2 query by lifecycle (ViewType), no filter/order_by (raises `INVALID_PARAMETER_VALUE`)
- `_search_runs`: Basic LSI1 query by lifecycle, no filter/order_by (raises `INVALID_PARAMETER_VALUE`)
- `search_registered_models`: Basic GSI2 query, no filter/order_by
- `search_model_versions`: Basic listing, no filter/order_by
- RANK items written on `log_batch` but never queried for search
- DLINK items written on `log_inputs` but never queried for search
- No FTS items written on any CRUD operation
- No tag denormalization (tags written as separate items only, not on META `tags` map)
- No NAME_REV items written
- No pagination tokens

---

## File Structure (new/modified files only)

```
src/mlflow_dynamodbstore/
├── dynamodb/
│   ├── search.py                   # NEW: Filter parser adapter, query planner, query executor, pagination
│   ├── fts.py                      # NEW: Word tokenizer, trigram tokenizer, FTS key builders, diff helpers
│   └── config.py                   # NEW: Config readers (denormalize patterns, FTS trigram fields)
├── tracking_store.py               # MODIFY: Wire search.py into _search_runs, search_experiments; add FTS writes, tag denormalization, NAME_REV
├── registry_store.py               # MODIFY: Wire search.py into search_registered_models, search_model_versions; add FTS writes, NAME_REV

tests/
├── unit/
│   ├── dynamodb/
│   │   ├── test_search.py          # NEW: Filter parser, query planner, pagination tests
│   │   ├── test_fts.py             # NEW: Tokenizer tests (word + trigram), key builder tests
│   │   └── test_config.py          # NEW: Config reader tests
│   ├── test_tracking_search.py     # NEW: search_runs, search_experiments with filters
│   └── test_registry_search.py     # NEW: search_registered_models, search_model_versions with filters
├── integration/
│   ├── test_search_runs.py         # NEW: End-to-end search with moto server
│   └── test_search_models.py       # NEW: End-to-end model search with moto server
```

---

## Chunk 1: FTS Tokenizers + Key Builders

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
        assert "error" in tokens or any("error" in t for t in tokens)

    def test_stop_words_removed(self):
        tokens = tokenize_words("the a an is in on at to for")
        assert len(tokens) == 0

    def test_short_words_removed(self):
        tokens = tokenize_words("I a x go")
        assert len(tokens) == 1

    def test_stemming(self):
        t1 = tokenize_words("errors")
        t2 = tokenize_words("error")
        t3 = tokenize_words("errored")
        assert t1 == t2 == t3

    def test_alphanumeric_only(self):
        tokens = tokenize_words("gpt-4-turbo v2.0")
        assert "gpt" in tokens
        assert "turbo" in tokens

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
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "whose",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "out", "off",
    "down", "against", "until", "while",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet", "both", "either",
    "neither", "not", "only", "than", "when", "if", "because",
    "as", "while", "although", "though",
    # Be verbs
    "be", "am", "is", "are", "was", "were", "been", "being",
    # Have verbs
    "has", "have", "had", "having",
    # Do verbs
    "do", "does", "did", "doing",
    # Modal verbs
    "will", "would", "shall", "should", "may", "might",
    "can", "could", "must",
    # Common adverbs
    "no", "not", "very", "too", "also", "just", "more", "most",
    "now", "then", "here", "there", "where", "how", "all", "each",
    "every", "any", "few", "some", "such", "own", "same", "other",
    "much", "many", "well", "back", "even", "still", "already",
    # Common verbs (low semantic value)
    "get", "got", "gets", "make", "made", "let",
    # Misc
    "no", "yes", "one", "two",
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

### Task 3: FTS key builders and diff helpers

- [ ] **Step 1: Write failing tests**

Test FTS forward/reverse key generation for entity types used in Plan 2 (experiment name, run name, run param, run tag, model name). Test `fts_items_for_text(pk, entity_type, entity_id, field, text, levels)` returning list of DynamoDB items (forward + reverse, word + trigram). Test `fts_diff(old_text, new_text, levels)` returning `(tokens_to_add, tokens_to_remove)`. Test that experiment/model name FTS items include `gsi2pk`/`gsi2sk` for cross-partition search.

```python
from mlflow_dynamodbstore.dynamodb.fts import fts_items_for_text, fts_diff

class TestFTSKeyBuilders:
    def test_experiment_name_fts_items(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ", entity_type="E", entity_id="01JQXYZ",
            field=None, text="my data pipeline", levels=("W", "3"),
            workspace="default",
        )
        # Should have forward + reverse for each word token and each trigram
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        reverse_items = [i for i in items if i["SK"].startswith("FTS_REV#")]
        assert len(forward_items) == len(reverse_items)
        # Experiment name FTS items should have GSI2 attributes
        for item in forward_items:
            assert "gsi2pk" in item
            assert item["gsi2pk"].startswith("FTS_NAMES#")

    def test_run_name_fts_items_no_gsi2(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ", entity_type="R", entity_id="01JRABC",
            field=None, text="training run", levels=("W", "3"),
        )
        # Run name FTS items should NOT have GSI2 (not cross-partition)
        for item in items:
            assert "gsi2pk" not in item

    def test_fts_diff(self):
        to_add, to_remove = fts_diff("old pipeline", "new pipeline", levels=("W",))
        # "pipeline" is common, "old"→remove, "new"→add (both stemmed)
        assert len(to_add) > 0
        assert len(to_remove) > 0
        # "pipeline" stem should not be in either set
```

- [ ] **Step 2: Implement FTS key builders**

`fts_items_for_text()`: Returns list of DynamoDB item dicts (forward + reverse). For entity names (experiment, model), adds `gsi2pk`/`gsi2sk` for cross-partition FTS. For other fields (run name, params, tags), no GSI attributes.

`fts_diff()`: Computes (old_tokens - new_tokens, new_tokens - old_tokens) for both word and trigram levels.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add FTS key builders, diff helpers, and cross-partition GSI2 support"
```

---

## Chunk 2: Config Readers + Tag Denormalization + NAME_REV

### Task 4: Config readers (denormalize patterns, FTS trigram fields)

**Files:**
- Create: `src/mlflow_dynamodbstore/dynamodb/config.py`
- Create: `tests/unit/dynamodb/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/dynamodb/test_config.py
from mlflow_dynamodbstore.dynamodb.config import ConfigReader

class TestConfigReader:
    def test_denormalize_patterns_default(self, config_reader):
        patterns = config_reader.get_denormalize_patterns()
        assert "mlflow.*" in patterns

    def test_denormalize_patterns_per_experiment(self, config_reader):
        # Write per-experiment config, verify merge
        config_reader.set_experiment_denormalize_patterns("01JQXYZ", ["team.*"])
        patterns = config_reader.get_effective_denormalize_patterns("01JQXYZ")
        assert "mlflow.*" in patterns  # global
        assert "team.*" in patterns    # experiment-specific

    def test_should_denormalize(self, config_reader):
        assert config_reader.should_denormalize(None, "mlflow.user") is True
        assert config_reader.should_denormalize(None, "custom_tag") is False

    def test_fts_trigram_fields_default(self, config_reader):
        fields = config_reader.get_fts_trigram_fields()
        assert isinstance(fields, list)

    def test_should_trigram_entity_names_always_true(self, config_reader):
        assert config_reader.should_trigram("experiment_name") is True
        assert config_reader.should_trigram("run_name") is True
        assert config_reader.should_trigram("model_name") is True

    def test_should_trigram_other_fields_configurable(self, config_reader):
        # Default: no extra trigram fields
        assert config_reader.should_trigram("run_param_value") is False

    def test_reconcile_from_env(self, config_reader, monkeypatch):
        monkeypatch.setenv("MLFLOW_DYNAMODB_DENORMALIZE_TAGS", "mlflow.*,env,team.*")
        config_reader.reconcile()
        patterns = config_reader.get_denormalize_patterns()
        assert "env" in patterns
        assert "team.*" in patterns
        assert "mlflow.*" in patterns

    def test_reconcile_preserves_mlflow_star(self, config_reader, monkeypatch):
        monkeypatch.setenv("MLFLOW_DYNAMODB_DENORMALIZE_TAGS", "env")
        config_reader.reconcile()
        patterns = config_reader.get_denormalize_patterns()
        assert "mlflow.*" in patterns  # always re-added
```

- [ ] **Step 2: Implement ConfigReader**

Reads CONFIG items from DynamoDB, caches in memory, provides `should_denormalize(experiment_id, tag_key)` and `should_trigram(field_type)`. Implements `reconcile()` for env var override on startup.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add config readers for denormalize patterns and FTS trigram fields"
```

### Task 5: Wire tag denormalization into tracking store

Modify `tracking_store.py`:
- Initialize `ConfigReader` in `__init__`, call `reconcile()`
- `set_tag` / `log_batch` tags: if `should_denormalize()`, also UpdateItem META `tags.<key> = value`
- `delete_tag`: if denormalized, also UpdateItem META `REMOVE tags.<key>`
- `set_experiment_tag`: same pattern for experiment META

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tracking_search.py (or extend test_tracking_store.py)
class TestTagDenormalization:
    def test_mlflow_tag_denormalized_on_run_meta(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.set_tag(run.info.run_id, RunTag("mlflow.user", "alice"))
        # Verify the run META item has tags.mlflow.user
        meta = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert meta.get("tags", {}).get("mlflow.user") == "alice"

    def test_custom_tag_not_denormalized_by_default(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.set_tag(run.info.run_id, RunTag("custom_tag", "value"))
        meta = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert "custom_tag" not in meta.get("tags", {})

    def test_delete_tag_removes_from_denormalized(self, tracking_store, table):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r")
        tracking_store.set_tag(run.info.run_id, RunTag("mlflow.user", "alice"))
        tracking_store.delete_tag(run.info.run_id, "mlflow.user")
        meta = table.get_item(f"EXP#{exp_id}", f"R#{run.info.run_id}")
        assert "mlflow.user" not in meta.get("tags", {})
```

- [ ] **Step 2: Implement denormalization in tracking store writes**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Run existing Plan 1 tests to verify no regressions**

```bash
uv run pytest tests/unit/test_tracking_store.py -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: wire tag denormalization into tracking store writes"
```

### Task 6: Wire tag denormalization into registry store

Same pattern for `set_registered_model_tag` / `delete_registered_model_tag` / `set_model_version_tag` / `delete_model_version_tag`.

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: wire tag denormalization into registry store writes"
```

### Task 7: Wire FTS into tracking store writes

Modify `tracking_store.py` to write FTS items during:
- `create_experiment` / `rename_experiment` (experiment name: word + trigram + GSI2 cross-partition)
- `create_run` / `update_run_info` (run name: word + trigram)
- `log_batch` params (word + trigram if configured)
- `set_tag` / `delete_tag` (word + trigram if configured, with diff for updates)

- [ ] **Step 1: Write failing tests**

Test that after creating an experiment, FTS items exist in DynamoDB (both W# and 3# prefixed). Test that after renaming, old tokens deleted and new tokens written. Test that experiment name FTS items have GSI2 attributes. Test log_batch params write FTS items.

- [ ] **Step 2: Modify tracking_store.py to call FTS helpers on writes**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: wire FTS writes into tracking store CRUD operations"
```

### Task 8: Wire FTS into registry store writes

Modify `registry_store.py` to write FTS items for model names on `create_registered_model` / `rename_registered_model` (word + trigram + GSI2 cross-partition).

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Modify registry_store.py**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: wire FTS writes into registry store CRUD operations"
```

### Task 9: NAME_REV materialized items for GSI5 suffix ILIKE

Modify tracking_store.py and registry_store.py:
- `create_experiment` / `rename_experiment`: Write/update `E#NAME_REV` item with `gsi5sk = REV#<rev(lower(name))>#<id>`
- `create_registered_model` / `rename_registered_model`: Write/update `M#NAME_REV` item with `gsi5sk = REV#<rev(lower(name))>`

- [ ] **Step 1: Write failing tests**

Test NAME_REV item exists after create. Test GSI5 query `begins_with("REV#...")` returns matching items. Test rename updates NAME_REV.

- [ ] **Step 2: Add NAME_REV writes**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add NAME_REV materialized items for suffix ILIKE"
```

---

## Chunk 3: Filter Parser, Query Planner & Pagination

### Task 10: MLflow filter parser adapter

**Files:**
- Create: `src/mlflow_dynamodbstore/dynamodb/search.py`
- Create: `tests/unit/dynamodb/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/dynamodb/test_search.py
from mlflow_dynamodbstore.dynamodb.search import parse_run_filter, parse_experiment_filter, FilterPredicate

class TestRunFilterParser:
    def test_empty_filter(self):
        predicates = parse_run_filter("")
        assert predicates == []

    def test_attribute_equals(self):
        predicates = parse_run_filter("status = 'FINISHED'")
        assert len(predicates) == 1
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "status"
        assert predicates[0].op == "="
        assert predicates[0].value == "FINISHED"

    def test_metric_comparison(self):
        predicates = parse_run_filter("metric.accuracy > 0.9")
        assert predicates[0].field_type == "metric"
        assert predicates[0].key == "accuracy"
        assert predicates[0].op == ">"
        assert predicates[0].value == 0.9

    def test_param_like(self):
        predicates = parse_run_filter("param.model LIKE '%transformer%'")
        assert predicates[0].field_type == "param"
        assert predicates[0].op == "LIKE"

    def test_tag_equals(self):
        predicates = parse_run_filter("tag.env = 'prod'")
        assert predicates[0].field_type == "tag"

    def test_compound_and(self):
        predicates = parse_run_filter("metric.acc > 0.9 AND param.lr = '0.01'")
        assert len(predicates) == 2

    def test_dataset_filter(self):
        predicates = parse_run_filter("dataset.name = 'my_data'")
        assert predicates[0].field_type == "dataset"

    def test_run_name_like(self):
        predicates = parse_run_filter("run_name LIKE '%pipeline%'")
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "run_name"

class TestExperimentFilterParser:
    def test_name_like(self):
        predicates = parse_experiment_filter("name LIKE 'prod%'")
        assert predicates[0].field_type == "attribute"
        assert predicates[0].key == "name"
        assert predicates[0].op == "LIKE"

    def test_tag_filter(self):
        predicates = parse_experiment_filter("tag.team = 'ml'")
        assert predicates[0].field_type == "tag"
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement filter parsers**

Use MLflow's existing `SearchUtils` / `SearchExperimentsUtils` to parse the filter string. Our `parse_run_filter` / `parse_experiment_filter` are adapters that normalize MLflow's internal parsed representation into our `FilterPredicate` dataclass.

```python
@dataclass(frozen=True)
class FilterPredicate:
    field_type: str  # "attribute", "metric", "param", "tag", "dataset"
    key: str
    op: str  # "=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "NOT IN", "IS NULL", "IS NOT NULL"
    value: Any
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add MLflow filter parser adapters for runs and experiments"
```

### Task 11: Query planner

- [ ] **Step 1: Write failing tests**

```python
from mlflow_dynamodbstore.dynamodb.search import plan_run_query, QueryPlan

class TestRunQueryPlanner:
    def test_no_filter_uses_base_sk(self):
        plan = plan_run_query(predicates=[], order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi1"  # lifecycle filter
        assert plan.sk_prefix == "active#"

    def test_status_filter_uses_lsi3(self):
        preds = parse_run_filter("status = 'FINISHED'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi3"
        assert plan.sk_prefix == "FINISHED#"

    def test_metric_order_uses_rank(self):
        plan = plan_run_query(predicates=[], order_by=["metric.accuracy DESC"], view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "rank"
        assert plan.rank_key == "accuracy"

    def test_dataset_filter_uses_dlink(self):
        preds = parse_run_filter("dataset.name = 'my_data'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "dlink"

    def test_run_name_like_uses_fts(self):
        preds = parse_run_filter("run_name LIKE '%pipeline%'")
        plan = plan_run_query(predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY)
        assert plan.strategy == "fts"

    def test_tag_filter_denormalized_uses_filter_expression(self):
        preds = parse_run_filter("tag.mlflow.user = 'alice'")
        plan = plan_run_query(
            predicates=preds, order_by=None, view_type=ViewType.ACTIVE_ONLY,
            denormalized_patterns=["mlflow.*"],
        )
        assert any(fe for fe in plan.filter_expressions)

    def test_order_by_end_time_uses_lsi2(self):
        plan = plan_run_query(predicates=[], order_by=["end_time ASC"], view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi2"

    def test_order_by_duration_uses_lsi5(self):
        plan = plan_run_query(predicates=[], order_by=["duration DESC"], view_type=ViewType.ACTIVE_ONLY)
        assert plan.index == "lsi5"
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement query planner**

`plan_run_query()` analyzes predicates and order_by to select the optimal execution strategy:
1. Check for specialized strategies: RANK (metric/param order), DLINK (dataset filter), FTS (LIKE '%word%')
2. Check order_by → LSI mapping: end_time→LSI2, status→LSI3, run_name→LSI4, duration→LSI5
3. Default: lifecycle filter via LSI1 (active#/deleted#)
4. Classify remaining predicates: FilterExpressions (denormalized tags) vs post-filters (BatchGetItem for non-denormalized tags, params, metrics)
5. Return `QueryPlan` dataclass

```python
@dataclass
class QueryPlan:
    strategy: str  # "index", "rank", "dlink", "fts"
    index: str | None  # LSI/GSI name
    sk_prefix: str | None  # begins_with condition
    scan_forward: bool
    filter_expressions: list[str]  # DynamoDB FilterExpression clauses
    post_filters: list[FilterPredicate]  # Applied in Python after fetch
    rank_key: str | None  # For strategy="rank"
    fts_query: str | None  # For strategy="fts"
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add DynamoDB query planner for MLflow search"
```

### Task 12: Pagination engine

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

Base64-encoded JSON. Encode DynamoDB's `LastEvaluatedKey`, experiment index for multi-experiment queries, accumulated result count.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add pagination token encoding/decoding"
```

### Task 13: Query executor

Add a `execute_query(table, plan, pk, max_results, page_token)` function that:
1. Executes the DynamoDB query based on the plan (index selection, key conditions, FilterExpressions)
2. For RANK strategy: query RANK items, extract run IDs, BatchGetItem run METAs
3. For DLINK strategy: query DLINK items, extract run IDs, BatchGetItem run METAs
4. For FTS strategy: query FTS items (word first, fallback to trigram), extract entity IDs, BatchGetItem METAs
5. Applies post-filters in Python (non-denormalized tag lookups via BatchGetItem, param/metric value checks)
6. Handles pagination (LastEvaluatedKey → page token)

- [ ] **Step 1: Write failing tests**

Test each execution strategy with actual DynamoDB items (via `@mock_aws`).

- [ ] **Step 2: Implement query executor**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add query executor for all search strategies"
```

---

## Chunk 4: Wire Search into Stores

### Task 14: Full _search_runs implementation

Modify `tracking_store.py`: replace the Plan 1 basic `_search_runs` with the full implementation using parse → plan → execute pipeline.

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
        assert runs[0].info.run_id == run2.info.run_id

    def test_search_by_denormalized_tag(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.set_tag(run1.info.run_id, RunTag("mlflow.user", "alice"))
        runs, _ = tracking_store._search_runs([exp_id], "tag.mlflow.user = 'alice'", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 1

    def test_search_by_dataset(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        run1 = tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="r1")
        # log_inputs with dataset...
        # search by dataset.name = 'my_data'...

    def test_search_run_name_like_word(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        tracking_store.create_run(exp_id, user_id="u", start_time=1000, tags=[], run_name="my-pipeline-v1")
        tracking_store.create_run(exp_id, user_id="u", start_time=1001, tags=[], run_name="other-run")
        runs, _ = tracking_store._search_runs([exp_id], "run_name LIKE '%pipeline%'", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 1

    def test_search_pagination(self, tracking_store):
        exp_id = tracking_store.create_experiment("exp", artifact_location="s3://b")
        for i in range(30):
            tracking_store.create_run(exp_id, user_id="u", start_time=1000 + i, tags=[], run_name=f"r{i}")
        runs, token = tracking_store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 10, None, None)
        assert len(runs) == 10
        assert token is not None
        runs2, token2 = tracking_store._search_runs([exp_id], "", ViewType.ACTIVE_ONLY, 10, None, token)
        assert len(runs2) == 10
        # All run IDs should be different
        ids1 = {r.info.run_id for r in runs}
        ids2 = {r.info.run_id for r in runs2}
        assert ids1.isdisjoint(ids2)

    def test_multi_experiment_search(self, tracking_store):
        exp1 = tracking_store.create_experiment("exp1", artifact_location="s3://b")
        exp2 = tracking_store.create_experiment("exp2", artifact_location="s3://b")
        tracking_store.create_run(exp1, user_id="u", start_time=1000, tags=[], run_name="r1")
        tracking_store.create_run(exp2, user_id="u", start_time=1001, tags=[], run_name="r2")
        runs, _ = tracking_store._search_runs([exp1, exp2], "", ViewType.ACTIVE_ONLY, 100, None, None)
        assert len(runs) == 2
```

- [ ] **Step 2: Replace _search_runs implementation**

1. Parse filter_string via `parse_run_filter`
2. Plan query via `plan_run_query`
3. For each experiment_id: execute query via `execute_query`
4. Merge results across experiments
5. Handle pagination tokens
6. Return `(runs, next_page_token)`

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Run Plan 1 tests to verify no regressions**

```bash
uv run pytest tests/unit/test_tracking_store.py tests/integration/test_tracking_crud.py -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: implement full _search_runs with filter parser and query planner"
```

### Task 15: Full search_experiments implementation

Upgrade `search_experiments` to support `filter_string` and `order_by`.

- [ ] **Step 1: Write failing tests**

Test filter by name LIKE 'prefix%' (GSI5 FWD#), LIKE '%suffix' (GSI5 REV#), LIKE '%word%' (FTS via GSI2 FTS_NAMES#). Filter by tag (denormalized: FilterExpression; non-denormalized: BatchGetItem). Order by name (LSI3), creation_time (ULID/GSI2), last_update_time (LSI2). Pagination.

- [ ] **Step 2: Implement using query planner**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: upgrade search_experiments with full filter/order_by support"
```

### Task 16: Full search_registered_models and search_model_versions

- [ ] **Step 1: Write failing tests**

Test filter by name LIKE (GSI5 + FTS), filter by tag, order by name/last_update_time. For model versions: filter by run_id (GSI1 cross-model or LSI5 within-model), order by version/creation_time (LSI1)/last_update_time (LSI2). Pagination.

- [ ] **Step 2: Implement using query planner**
- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add full search to registry store"
```

---

## Chunk 5: Integration Tests + Verification

### Task 17: Search integration tests

**Files:**
- Create: `tests/integration/test_search_runs.py`
- Create: `tests/integration/test_search_models.py`

- [ ] **Step 1: Write integration tests**

End-to-end search scenarios via moto server:
- Create experiment + 50 runs with varied metrics/params/tags → search with filters → verify results
- Pagination: search with max_results=10, follow page_tokens through all results
- Order by metric DESC → verify RANK item ordering
- LIKE '%word%' → verify FTS query (word-level)
- LIKE '%part%' → verify FTS trigram query
- LIKE 'prefix%' / '%suffix' → verify GSI5 FWD#/REV# queries
- Dataset filter → verify DLINK query
- Tag filter (denormalized mlflow.*) → verify FilterExpression
- Tag filter (non-denormalized) → verify BatchGetItem + Python filter
- Multi-experiment search → verify merge
- Cross-partition experiment name LIKE → verify GSI2 FTS_NAMES#

- [ ] **Step 2: Run integration tests**

```bash
uv run pytest tests/integration/test_search_runs.py tests/integration/test_search_models.py -v
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test: add search integration tests"
```

### Task 18: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --cov=mlflow_dynamodbstore --cov-report=term-missing
```

- [ ] **Step 2: Run linters**

```bash
uv run ruff check src/ tests/
uv run mypy src/mlflow_dynamodbstore/
```

- [ ] **Step 3: Run MLflow compatibility tests**

```bash
uv run pytest tests/compatibility/ -v --tb=short
```

Check if any previously-failing compatibility tests now pass with search support.

- [ ] **Step 4: Commit any fixes**

```bash
git commit -m "chore: fix lint and type errors from Plan 2 verification"
```
