"""Tests for full-text search tokenizers."""

from mlflow_dynamodbstore.dynamodb.fts import tokenize_words


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
