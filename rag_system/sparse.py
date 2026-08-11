"""Small, dependency-free BM25 index for lexical candidate retrieval.

The index deliberately works with plain document IDs and strings. This keeps
it independent from vector-store and framework types, while callers can use a
chunk ID as the document ID when indexing document chunks.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from rag_system.text import lexical_tokens, normalize_text, stable_digest


@dataclass(frozen=True, slots=True)
class SparseDocument:
    """A lexical-search document identified by a stable caller-facing ID."""

    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str):
            raise TypeError("document_id must be a string")
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not normalize_text(self.text):
            raise ValueError("text cannot be empty")
        if not lexical_tokens(self.text):
            raise ValueError("text must contain at least one searchable token")


@dataclass(frozen=True, slots=True)
class SparseSearchHit:
    """A BM25 result. Larger scores indicate stronger lexical relevance."""

    document_id: str
    text: str
    score: float


def stable_document_id(text: str, *, namespace: str = "document") -> str:
    """Return a repeatable ID for normalized text within a namespace."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("text cannot be empty")
    if not lexical_tokens(normalized):
        raise ValueError("text must contain at least one searchable token")
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    if not namespace.strip():
        raise ValueError("namespace cannot be empty")
    return f"doc-{stable_digest((namespace.strip(), normalized), length=24)}"


class BM25Index:
    """An immutable in-memory Okapi BM25 index.

    Duplicate IDs with identical text are collapsed. Reusing an ID for
    different text is rejected because silently choosing either version would
    make retrieval results depend on input order.
    """

    def __init__(
        self,
        documents: Iterable[SparseDocument] = (),
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = _finite_number("k1", k1)
        self._b = _finite_number("b", b)
        if self._k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= self._b <= 1:
            raise ValueError("b must be between zero and one")

        if isinstance(documents, (str, bytes)):
            raise TypeError("documents must be an iterable of SparseDocument objects")

        unique_documents: list[SparseDocument] = []
        by_id: dict[str, SparseDocument] = {}
        for document in documents:
            if not isinstance(document, SparseDocument):
                raise TypeError("documents must contain only SparseDocument objects")
            previous = by_id.get(document.document_id)
            if previous is not None:
                if normalize_text(previous.text) != normalize_text(document.text):
                    raise ValueError(
                        f"document_id {document.document_id!r} is used for different text"
                    )
                continue
            by_id[document.document_id] = document
            unique_documents.append(document)

        self._documents = tuple(unique_documents)
        tokenized = tuple(lexical_tokens(document.text) for document in self._documents)
        self._term_frequencies = tuple(Counter(tokens) for tokens in tokenized)
        self._document_lengths = tuple(len(tokens) for tokens in tokenized)
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())

        document_count = len(self._documents)
        self._inverse_document_frequency = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        document_ids: Iterable[str] | None = None,
        namespace: str = "document",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Build an index, generating stable content IDs when IDs are omitted."""

        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be an iterable of strings")
        text_items = tuple(texts)
        if document_ids is None:
            id_items = tuple(stable_document_id(text, namespace=namespace) for text in text_items)
        else:
            if isinstance(document_ids, (str, bytes)):
                raise TypeError("document_ids must be an iterable of strings")
            id_items = tuple(document_ids)
            if len(id_items) != len(text_items):
                raise ValueError("document_ids and texts must have the same length")

        documents = (
            SparseDocument(document_id=document_id, text=text)
            for document_id, text in zip(id_items, text_items, strict=True)
        )
        return cls(documents, k1=k1, b=b)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(document.document_id for document in self._documents)

    def search(self, query: str, top_k: int = 10) -> tuple[SparseSearchHit, ...]:
        """Return up to ``top_k`` positive-scoring documents in stable order."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_terms = tuple(dict.fromkeys(lexical_tokens(query)))
        if not query_terms or not self._documents:
            return ()

        scored: list[SparseSearchHit] = []
        for document, frequencies, document_length in zip(
            self._documents,
            self._term_frequencies,
            self._document_lengths,
            strict=True,
        ):
            score = 0.0
            length_ratio = document_length / self._average_document_length
            normalization = self._k1 * (1 - self._b + self._b * length_ratio)
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                inverse_frequency = self._inverse_document_frequency.get(term, 0.0)
                score += inverse_frequency * (
                    term_frequency * (self._k1 + 1)
                    / (term_frequency + normalization)
                )

            if score > 0:
                scored.append(
                    SparseSearchHit(
                        document_id=document.document_id,
                        text=document.text,
                        score=score,
                    )
                )

        scored.sort(key=lambda hit: (-hit.score, hit.document_id))
        return tuple(scored[:top_k])


def _finite_number(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved
