"""Full-text search tokenizers: word-level (stemmed) and trigram-level."""

from __future__ import annotations

import re
from typing import Any

import snowballstemmer

from mlflow_dynamodbstore.dynamodb.schema import (
    GSI2_FTS_NAMES_PREFIX,
    GSI2_PK,
    GSI2_SK,
    SK_FTS_PREFIX,
    SK_FTS_REV_PREFIX,
)

STOP_WORDS = frozenset(
    {
        # Articles & determiners
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # Pronouns
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        # Prepositions
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "out",
        "off",
        "down",
        "against",
        "until",
        "while",
        # Conjunctions
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "not",
        "only",
        "than",
        "when",
        "if",
        "because",
        "as",
        "while",
        "although",
        "though",
        # Be verbs
        "be",
        "am",
        "is",
        "are",
        "was",
        "were",
        "been",
        "being",
        # Have verbs
        "has",
        "have",
        "had",
        "having",
        # Do verbs
        "do",
        "does",
        "did",
        "doing",
        # Modal verbs
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "must",
        # Common adverbs
        "no",
        "not",
        "very",
        "too",
        "also",
        "just",
        "more",
        "most",
        "now",
        "then",
        "here",
        "there",
        "where",
        "how",
        "all",
        "each",
        "every",
        "any",
        "few",
        "some",
        "such",
        "own",
        "same",
        "other",
        "much",
        "many",
        "well",
        "back",
        "even",
        "still",
        "already",
        # Common verbs (low semantic value)
        "get",
        "got",
        "gets",
        "make",
        "made",
        "let",
        # Misc
        "no",
        "yes",
        "one",
        "two",
    }
)

_stemmer = snowballstemmer.stemmer("english")


def tokenize_words(text: str) -> set[str]:
    """Stemmed whole-word tokens for LIKE '%complete_word%' matches."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return set(_stemmer.stemWords(words))


def tokenize_trigrams(text: str) -> set[str]:
    """Character trigrams for LIKE '%partial%' matches."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    grams: set[str] = set()
    for word in words:
        for i in range(len(word) - 2):
            grams.add(word[i : i + 3])
    return grams


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Entity types that require cross-partition GSI2 FTS attributes.
_GSI2_ENTITY_TYPES = frozenset({"E", "M"})


def _tokens_for_level(level: str, text: str) -> set[str]:
    """Return the token set for a given level identifier."""
    if level == "W":
        return tokenize_words(text)
    if level == "3":
        return tokenize_trigrams(text)
    if level == "2":
        return tokenize_tail_bigrams(text)
    raise ValueError(f"Unknown FTS level: {level!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fts_items_for_text(
    pk: str,
    entity_type: str,
    entity_id: str,
    field: str | None,
    text: str,
    levels: tuple[str, ...] = ("W", "3", "2"),
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    """Return DynamoDB item dicts (forward + reverse) for every FTS token.

    Key patterns
    ~~~~~~~~~~~~
    Forward SK : ``FTS#<level>#<entity_type>#<token>#<entity_id>[#<field>]``
    Reverse SK : ``FTS_REV#<entity_type>#<entity_id>[#<field>]#<level>#<token>``

    For experiment (``entity_type="E"``) and model (``entity_type="M"``)
    names the forward items also carry ``gsi2pk`` / ``gsi2sk`` so that a
    single GSI2 query can search across partitions.
    """
    add_gsi2 = entity_type in _GSI2_ENTITY_TYPES

    # Build the optional field suffix once.
    field_suffix = f"#{field}" if field else ""

    items: list[dict[str, Any]] = []

    for level in levels:
        tokens = _tokens_for_level(level, text)
        for token in tokens:
            forward_sk = f"{SK_FTS_PREFIX}{level}#{entity_type}#{token}#{entity_id}{field_suffix}"
            reverse_sk = (
                f"{SK_FTS_REV_PREFIX}{entity_type}#{entity_id}{field_suffix}#{level}#{token}"
            )

            forward: dict[str, Any] = {"PK": pk, "SK": forward_sk}
            reverse: dict[str, Any] = {"PK": pk, "SK": reverse_sk}

            if add_gsi2:
                gsi2pk_val = f"{GSI2_FTS_NAMES_PREFIX}{workspace}"
                gsi2sk_val = f"{level}#{entity_type}#{token}#{entity_id}{field_suffix}"
                forward[GSI2_PK] = gsi2pk_val
                forward[GSI2_SK] = gsi2sk_val

            items.append(forward)
            items.append(reverse)

    return items


def fts_diff(
    old_text: str | None,
    new_text: str,
    levels: tuple[str, ...] = ("W", "3", "2"),
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Compute the token-level diff between *old_text* and *new_text*.

    Returns ``(tokens_to_add, tokens_to_remove)`` where each element is a
    ``set`` of ``(level, token)`` tuples.  Common tokens appear in neither
    set.
    """
    old_tokens: set[tuple[str, str]] = set()
    new_tokens: set[tuple[str, str]] = set()

    for level in levels:
        if old_text is not None:
            for token in _tokens_for_level(level, old_text):
                old_tokens.add((level, token))
        for token in _tokens_for_level(level, new_text):
            new_tokens.add((level, token))

    tokens_to_add = new_tokens - old_tokens
    tokens_to_remove = old_tokens - new_tokens
    return tokens_to_add, tokens_to_remove
