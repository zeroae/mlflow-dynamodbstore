"""Full-text search tokenizers: word-level (stemmed) and trigram-level."""

from __future__ import annotations

import re

import snowballstemmer

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
