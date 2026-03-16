"""Tests for full-text search tokenizers."""

from mlflow_dynamodbstore.dynamodb.fts import tokenize_trigrams, tokenize_words


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
