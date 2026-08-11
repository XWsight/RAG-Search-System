"""Dependency boundaries used by the application layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_system.domain import Chunk, IndexRef, SearchHit, SourceDocument, WebSearchResult


class DocumentLoader(Protocol):
    def load(self, paths: Sequence[str]) -> Sequence[SourceDocument]: ...


class TextSplitter(Protocol):
    def split(self, document: SourceDocument) -> Sequence[Chunk]: ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorIndex(Protocol):
    @property
    def index_ref(self) -> IndexRef: ...

    def search(self, query: str, *, top_k: int) -> Sequence[SearchHit]: ...

    def close(self) -> None: ...

    def delete(self) -> None: ...


class IndexRepository(Protocol):
    def build(self, index_id: str, chunks: Sequence[Chunk]) -> VectorIndex: ...

    def delete(self, index_id: str) -> bool: ...


class ChatModel(Protocol):
    @property
    def available(self) -> bool: ...

    def answer(self, question: str, evidence: Sequence[tuple[str, str]]) -> str: ...


class WebSearchProvider(Protocol):
    @property
    def available(self) -> bool: ...

    def search(self, query: str, *, count: int) -> Sequence[WebSearchResult]: ...


class QueryPlanner(Protocol):
    @property
    def available(self) -> bool: ...

    def plan_queries(self, question: str, *, max_queries: int) -> Sequence[str]: ...


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> Sequence[SearchHit]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        top_k: int,
    ) -> Sequence[SearchHit]: ...
