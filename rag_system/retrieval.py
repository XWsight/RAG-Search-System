"""Dense, sparse, and fused retrieval implementations."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path

from rag_system.config import Settings
from rag_system.domain import Chunk, IndexRef, Route, RouteDecision, SearchHit
from rag_system.ports import IndexRepository, Reranker, VectorIndex
from rag_system.ranking import reciprocal_rank_fusion
from rag_system.reranking import RerankerError
from rag_system.sparse import BM25Index, SparseDocument
from rag_system.text import lexical_relevance


class DependencyUnavailableError(RuntimeError):
    """Raised when an optional runtime dependency is not installed."""


class IndexIntegrityError(RuntimeError):
    """Raised when a persisted collection does not match its manifest."""


class ChromaVectorIndex:
    """A Chroma collection hidden behind framework-neutral domain objects."""

    def __init__(
        self,
        *,
        store: object,
        index_ref: IndexRef,
        chunks: Sequence[Chunk],
        persistent: bool,
        inference_lock: threading.RLock,
    ) -> None:
        self._store = store
        self._index_ref = index_ref
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._persistent = persistent
        self._inference_lock = inference_lock
        self._closed = False
        self._deleted = False

    @property
    def index_ref(self) -> IndexRef:
        return self._index_ref

    def search(self, query: str, *, top_k: int) -> tuple[SearchHit, ...]:
        if self._closed:
            raise RuntimeError("index is closed")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        with self._inference_lock:
            results = self._store.similarity_search_with_score(query, k=top_k)
        hits: list[SearchHit] = []
        for rank, (document, distance_value) in enumerate(results, start=1):
            chunk_id = str(document.metadata.get("chunk_id", ""))
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            distance = max(0.0, float(distance_value))
            cosine_relevance = max(0.0, min(1.0, 1.0 - distance / 2.0))
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=cosine_relevance,
                    dense_rank=rank,
                    dense_distance=distance,
                    reasons=("dense",),
                )
            )
        return tuple(hits)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._persistent:
            self._delete_collection(suppress_errors=True)

    def delete(self) -> None:
        """Permanently delete the collection, including persisted data."""

        self._delete_collection(suppress_errors=False)
        self._closed = True

    def _delete_collection(self, *, suppress_errors: bool) -> None:
        if self._deleted:
            return
        try:
            self._store.delete_collection()
        except Exception:
            if not suppress_errors:
                raise
            # Ephemeral cache eviction is best-effort. Durable deletion uses
            # ``delete()`` above and deliberately propagates failures so the
            # catalog is not removed while vector data is still present.
        self._deleted = True


class ChromaIndexRepository(IndexRepository):
    """Build or reopen isolated cosine-space Chroma collections."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.validate()
        self._embedding_function: object | None = None
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()

    def _embeddings(self) -> object:
        with self._lock:
            if self._embedding_function is None:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError as error:
                    raise DependencyUnavailableError(
                        "缺少 langchain-huggingface，请先安装项目依赖。"
                    ) from error
                self._embedding_function = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
            return self._embedding_function

    def build(self, index_id: str, chunks: Sequence[Chunk]) -> ChromaVectorIndex:
        if not chunks:
            raise ValueError("cannot build an empty index")
        try:
            from langchain_chroma import Chroma
        except ImportError as error:
            raise DependencyUnavailableError("缺少 langchain-chroma，请先安装项目依赖。") from error

        collection_name = self._collection_name(index_id)
        store = self._new_store(Chroma, collection_name)
        expected_ids = {chunk.chunk_id for chunk in chunks}
        existing_ids = self._existing_ids(store)
        if existing_ids and existing_ids != expected_ids:
            # A partial collection can be left behind by a terminated indexing
            # job. Its deterministic name makes a clean rebuild safe.
            try:
                store.delete_collection()
            except Exception as error:
                raise IndexIntegrityError(
                    "persisted index is inconsistent and could not be rebuilt"
                ) from error
            store = self._new_store(Chroma, collection_name)
            existing_ids = set()
        if not existing_ids:
            with self._inference_lock:
                store.add_texts(
                    texts=[chunk.text for chunk in chunks],
                    ids=[chunk.chunk_id for chunk in chunks],
                    metadatas=[
                        {
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "source_name": chunk.source_name,
                            "chunk_index": chunk.chunk_index,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "heading": chunk.heading,
                        }
                        for chunk in chunks
                    ],
                )
        return ChromaVectorIndex(
            store=store,
            index_ref=IndexRef(
                index_id=index_id,
                document_count=len({chunk.document_id for chunk in chunks}),
                chunk_count=len(chunks),
                created_at=time.time(),
            ),
            chunks=chunks,
            persistent=self.settings.persist_data,
            inference_lock=self._inference_lock,
        )

    def delete(self, index_id: str) -> bool:
        """Delete a persisted collection that is not currently in the cache."""

        if not self.settings.persist_data:
            return False
        try:
            import chromadb
            from chromadb.errors import NotFoundError
        except ImportError as error:
            raise DependencyUnavailableError("missing chromadb dependency") from error

        directory = self._persistence_directory()
        client = chromadb.PersistentClient(path=str(directory))
        try:
            client.delete_collection(self._collection_name(index_id))
        except NotFoundError:
            return False
        return True

    def _new_store(self, chroma_type: type, collection_name: str) -> object:
        options: dict[str, object] = {
            "collection_name": collection_name,
            "embedding_function": self._embeddings(),
            "collection_metadata": {"hnsw:space": "cosine"},
        }
        if self.settings.persist_data:
            options["persist_directory"] = str(self._persistence_directory())
        return chroma_type(**options)

    def _persistence_directory(self) -> Path:
        directory = self.settings.storage_root.expanduser().resolve() / "vector"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _collection_name(index_id: str) -> str:
        return f"rag_{index_id.replace('-', '_')}"

    @staticmethod
    def _existing_ids(store: object) -> set[str]:
        result = store.get(include=[])
        if not isinstance(result, dict):
            raise IndexIntegrityError("persisted index returned an invalid manifest")
        ids = result.get("ids", [])
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise IndexIntegrityError("persisted index returned invalid identifiers")
        return set(ids)


class HybridRetriever:
    """Fuse dense and BM25 candidates, then compute an interpretable score."""

    def __init__(
        self,
        vector_index: VectorIndex,
        chunks: Sequence[Chunk],
        settings: Settings,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.settings = settings.validate()
        self.reranker = reranker
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._sparse = BM25Index(
            SparseDocument(document_id=chunk.chunk_id, text=chunk.text) for chunk in chunks
        )

    def search(self, query: str, *, top_k: int) -> tuple[SearchHit, ...]:
        question = (query or "").strip()
        if not question:
            return ()
        if top_k < 1:
            raise ValueError("top_k must be positive")

        dense_hits = tuple(
            self.vector_index.search(question, top_k=self.settings.dense_candidates)
        )
        sparse_hits = self._sparse.search(question, top_k=self.settings.sparse_candidates)
        dense_by_id = {hit.chunk.chunk_id: hit for hit in dense_hits}
        sparse_by_id = {hit.document_id: hit for hit in sparse_hits}
        dense_ids = [hit.chunk.chunk_id for hit in dense_hits]
        sparse_ids = [hit.document_id for hit in sparse_hits]
        fused = reciprocal_rank_fusion({"dense": dense_ids, "sparse": sparse_ids})

        maximum_rrf = 2 / 61
        candidates: list[SearchHit] = []
        for item in fused[: self.settings.fused_candidates]:
            chunk = self._chunks.get(item.item_id)
            if chunk is None:
                continue
            dense_hit = dense_by_id.get(item.item_id)
            sparse_hit = sparse_by_id.get(item.item_id)
            dense_score = dense_hit.score if dense_hit else 0.0
            lexical_score = lexical_relevance(question, chunk.text)
            rrf_score = min(1.0, item.score / maximum_rrf)
            final_score = min(1.0, 0.55 * dense_score + 0.25 * lexical_score + 0.20 * rrf_score)
            reasons = tuple(
                reason
                for reason, present in (("dense", dense_hit is not None), ("sparse", sparse_hit is not None))
                if present
            )
            candidates.append(
                SearchHit(
                    chunk=chunk,
                    score=final_score,
                    dense_rank=dense_hit.dense_rank if dense_hit else None,
                    sparse_rank=(sparse_ids.index(item.item_id) + 1) if sparse_hit else None,
                    dense_distance=dense_hit.dense_distance if dense_hit else None,
                    reasons=reasons,
                )
            )

        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        if self.reranker is not None:
            try:
                candidates = list(
                    self.reranker.rerank(
                        question,
                        candidates,
                        top_k=self.settings.fused_candidates,
                    )
                )
            except RerankerError:
                # Reranking is an optional quality layer; first-stage results
                # remain usable if its model is unavailable at runtime.
                pass
        return tuple(self._diversify(candidates, top_k))

    @staticmethod
    def _diversify(candidates: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        selected: list[SearchHit] = []
        per_document: dict[str, int] = {}
        for hit in candidates:
            document_id = hit.chunk.document_id
            if per_document.get(document_id, 0) >= 2:
                continue
            selected.append(hit)
            per_document[document_id] = per_document.get(document_id, 0) + 1
            if len(selected) >= top_k:
                break
        return selected


class RoutingPolicy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings.validate()

    def decide(self, hits: Sequence[SearchHit], *, allow_web: bool) -> RouteDecision:
        if not hits:
            return RouteDecision(
                route=Route.WEB if allow_web else Route.REFUSED,
                confidence=0.0,
                reason="本地检索没有返回候选证据。",
            )

        confidence = self.confidence(hits)

        if confidence >= self.settings.local_confidence_threshold:
            return RouteDecision(Route.LOCAL, confidence, "本地证据达到置信度阈值。")
        if allow_web and confidence >= self.settings.local_confidence_threshold * 0.72:
            return RouteDecision(Route.HYBRID, confidence, "本地证据不完整，将补充网络来源。")
        if allow_web:
            return RouteDecision(Route.WEB, confidence, "本地证据不足，将使用网络来源。")
        return RouteDecision(Route.REFUSED, confidence, "本地证据不足且联网搜索未开启。")

    @staticmethod
    def confidence(hits: Sequence[SearchHit]) -> float:
        """Return the bounded confidence feature used by the routing policy."""

        if not hits:
            return 0.0
        top_score = max(0.0, min(1.0, hits[0].score))
        second_score = hits[1].score if len(hits) > 1 else 0.0
        margin = max(0.0, top_score - second_score)
        agreement = 1.0 if {"dense", "sparse"}.issubset(hits[0].reasons) else 0.0
        return min(
            1.0,
            0.75 * top_score + 0.15 * agreement + 0.10 * min(1.0, margin * 4),
        )
