"""Tests for full-text search tokenizers."""

from __future__ import annotations

from mlflow_dynamodbstore.dynamodb.fts import (
    fts_diff,
    fts_items_for_text,
    tokenize_trigrams,
    tokenize_words,
)


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


class TestFTSKeyBuilders:
    def test_experiment_name_fts_items_structure(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="my data pipeline",
            levels=("W", "3"),
            workspace="default",
        )
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        reverse_items = [i for i in items if i["SK"].startswith("FTS_REV#")]
        assert len(forward_items) > 0
        assert len(forward_items) == len(reverse_items)

    def test_experiment_name_fts_items_have_gsi2(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="my data pipeline",
            levels=("W", "3"),
            workspace="default",
        )
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        for item in forward_items:
            assert "gsi2pk" in item
            assert item["gsi2pk"].startswith("FTS_NAMES#")
            assert "gsi2sk" in item

    def test_model_name_fts_items_have_gsi2(self):
        items = fts_items_for_text(
            pk="RM#01JQABC",
            entity_type="M",
            entity_id="01JQABC",
            field=None,
            text="bert classifier",
            levels=("W",),
            workspace="prod",
        )
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        for item in forward_items:
            assert "gsi2pk" in item
            assert item["gsi2pk"] == "FTS_NAMES#prod"

    def test_run_name_fts_items_no_gsi2(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="R",
            entity_id="01JRABC",
            field=None,
            text="training run",
            levels=("W", "3"),
        )
        for item in items:
            assert "gsi2pk" not in item
            assert "gsi2sk" not in item

    def test_forward_sk_pattern(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="pipeline",
            levels=("W",),
            workspace="default",
        )
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        # Forward SK = FTS#<level>#<entity_type>#<token>#<entity_id>
        for item in forward_items:
            parts = item["SK"].split("#")
            assert parts[0] == "FTS"
            assert parts[1] == "W"
            assert parts[2] == "E"
            assert parts[4] == "01JQXYZ"

    def test_reverse_sk_pattern(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="pipeline",
            levels=("W",),
            workspace="default",
        )
        reverse_items = [i for i in items if i["SK"].startswith("FTS_REV#")]
        # Reverse SK = FTS_REV#<entity_type>#<entity_id>#<level>#<token>
        for item in reverse_items:
            parts = item["SK"].split("#")
            assert parts[0] == "FTS_REV"
            assert parts[1] == "E"
            assert parts[2] == "01JQXYZ"
            assert parts[3] == "W"

    def test_fts_items_with_field(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="R",
            entity_id="01JRABC",
            field="my.tag",
            text="test value",
            levels=("W",),
        )
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        # SK should include the field: FTS#W#<token>#R#01JRABC#my.tag
        for item in forward_items:
            assert "my.tag" in item["SK"]

    def test_pk_attribute_set(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="pipeline",
            levels=("W",),
            workspace="default",
        )
        for item in items:
            assert item["PK"] == "EXP#01JQXYZ"

    def test_trigram_level_items(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="pipeline",
            levels=("3",),
            workspace="default",
        )
        forward_items = [i for i in items if i["SK"].startswith("FTS#")]
        assert len(forward_items) > 0
        for item in forward_items:
            assert item["SK"].startswith("FTS#3#")

    def test_empty_text_produces_no_items(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="",
            levels=("W", "3"),
            workspace="default",
        )
        assert items == []

    def test_all_stop_words_produces_no_word_items(self):
        items = fts_items_for_text(
            pk="EXP#01JQXYZ",
            entity_type="E",
            entity_id="01JQXYZ",
            field=None,
            text="the a an",
            levels=("W",),
            workspace="default",
        )
        assert items == []


class TestFTSDiff:
    def test_diff_add_and_remove(self):
        to_add, to_remove = fts_diff("old pipeline", "new pipeline", levels=("W",))
        assert len(to_add) > 0
        assert len(to_remove) > 0

    def test_common_tokens_not_in_diff(self):
        to_add, to_remove = fts_diff("old pipeline", "new pipeline", levels=("W",))
        # "pipeline" stem should appear in neither set (it's common)
        from mlflow_dynamodbstore.dynamodb.fts import tokenize_words

        pipeline_stem = next(iter(tokenize_words("pipeline")))
        add_tokens = {t for _, t in to_add}
        remove_tokens = {t for _, t in to_remove}
        assert pipeline_stem not in add_tokens
        assert pipeline_stem not in remove_tokens

    def test_diff_with_none_old_text(self):
        to_add, to_remove = fts_diff(None, "new pipeline", levels=("W",))
        assert len(to_add) > 0
        assert len(to_remove) == 0

    def test_diff_returns_level_token_tuples(self):
        to_add, to_remove = fts_diff("old value", "new value", levels=("W", "3"))
        for level, token in to_add:
            assert level in ("W", "3")
            assert isinstance(token, str)
        for level, token in to_remove:
            assert level in ("W", "3")
            assert isinstance(token, str)

    def test_diff_identical_text_no_changes(self):
        to_add, to_remove = fts_diff("same text here", "same text here", levels=("W", "3"))
        assert len(to_add) == 0
        assert len(to_remove) == 0

    def test_diff_disjoint_texts(self):
        to_add, to_remove = fts_diff("apple banana", "cherry dragon", levels=("W",))
        # All old tokens should be removed, all new tokens should be added
        assert len(to_add) > 0
        assert len(to_remove) > 0
        add_tokens = {t for _, t in to_add}
        remove_tokens = {t for _, t in to_remove}
        assert add_tokens.isdisjoint(remove_tokens)
