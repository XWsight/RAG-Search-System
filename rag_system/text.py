"""Text normalization and lightweight lexical scoring utilities.

The dense embedding model remains the primary semantic retriever.  These
functions add an inexpensive lexical signal, which is especially useful for
exact names, acronyms, identifiers, and error codes that embeddings can miss.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable


_LATIN_OR_NUMBER = re.compile(r"[a-z0-9][a-z0-9_.+-]*", re.IGNORECASE)
_HAN_SEQUENCE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode and repeated whitespace without changing meaning."""

    normalized = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    return _WHITESPACE.sub(" ", normalized).strip()


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Tokenize mixed Chinese/Latin text without requiring a tokenizer model.

    Chinese sequences contribute both single characters and bigrams.  Latin
    words, model names, versions, and identifiers are kept as complete tokens.
    The function is deterministic so it can be used in tests and evaluations.
    """

    normalized = normalize_text(text).lower()
    tokens: list[str] = list(_LATIN_OR_NUMBER.findall(normalized))

    for sequence in _HAN_SEQUENCE.findall(normalized):
        tokens.extend(sequence)
        if len(sequence) > 1:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))

    return tuple(tokens)


def lexical_relevance(query: str, text: str) -> float:
    """Return a bounded 0..1 lexical relevance score.

    Query-token coverage is weighted more heavily than Jaccard similarity so a
    long document is not unfairly penalized for containing additional terms.
    Repeated terms do not inflate the score without limit.
    """

    query_tokens = Counter(lexical_tokens(query))
    text_tokens = Counter(lexical_tokens(text))
    if not query_tokens or not text_tokens:
        return 0.0

    overlap = sum(min(count, text_tokens[token]) for token, count in query_tokens.items())
    query_total = sum(query_tokens.values())
    coverage = overlap / query_total

    query_set = set(query_tokens)
    text_set = set(text_tokens)
    union = query_set | text_set
    jaccard = len(query_set & text_set) / len(union) if union else 0.0

    query_identifiers = set(_LATIN_OR_NUMBER.findall(normalize_text(query).lower()))
    text_identifiers = set(_LATIN_OR_NUMBER.findall(normalize_text(text).lower()))
    identifier_coverage = (
        len(query_identifiers & text_identifiers) / len(query_identifiers)
        if query_identifiers
        else 0.0
    )

    return min(1.0, 0.65 * coverage + 0.15 * jaccard + 0.2 * identifier_coverage)


def stable_digest(parts: Iterable[str], *, length: int = 24) -> str:
    """Create a stable identifier from ordered text parts."""

    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")

    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:length]


def truncate_text(text: str, max_characters: int) -> str:
    """Truncate text with an explicit marker, preserving an exact upper bound."""

    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if len(text) <= max_characters:
        return text
    if max_characters <= 3:
        return "." * max_characters
    return f"{text[: max_characters - 3]}..."
