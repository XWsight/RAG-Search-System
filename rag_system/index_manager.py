"""Bounded, thread-safe index lifecycle management."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rag_system.config import Settings
from rag_system.domain import IndexRef
from rag_system.ingestion import DocumentIngestor, IngestionResult
from rag_system.ports import IndexRepository
from rag_system.reranking import CrossEncoderReranker
from rag_system.retrieval import HybridRetriever


@dataclass(slots=True)
class ManagedIndex:
    index_ref: IndexRef
    retriever: HybridRetriever
    last_accessed: float
    close: Callable[[], None]
    delete: Callable[[], None]
    active_leases: int = 0
    retire_action: str | None = None


class IndexManager:
    """Prepare, build, cache, release, and permanently delete indexes."""

    def __init__(
        self,
        settings: Settings,
        repository: IndexRepository,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings.validate()
        self.repository = repository
        self.ingestor = DocumentIngestor(settings)
        self.reranker = (
            CrossEncoderReranker(
                settings.reranker_model,
                weight=settings.reranker_weight,
            )
            if settings.reranker_model
            else None
        )
        self.clock = clock
        self._indexes: OrderedDict[str, ManagedIndex] = OrderedDict()
        self._retiring: dict[str, ManagedIndex] = {}
        self._lock = threading.RLock()
        self._lease_changed = threading.Condition(self._lock)
        self._build_locks = tuple(threading.Lock() for _ in range(32))
        self._closed = False

    def prepare(
        self,
        paths: Sequence[str | Path] | None = None,
        *,
        namespace: str = "",
    ) -> IngestionResult:
        """Validate, load, and chunk documents without computing embeddings."""

        with self._lock:
            self._ensure_open()
        return self.ingestor.ingest(paths, namespace=namespace)

    def build(
        self,
        paths: Sequence[str | Path] | None = None,
        *,
        namespace: str = "",
    ) -> IndexRef:
        return self.build_prepared(self.prepare(paths, namespace=namespace))

    def build_prepared(self, ingestion: IngestionResult) -> IndexRef:
        """Build a prepared index without holding the global cache lock."""

        if not isinstance(ingestion, IngestionResult):
            raise TypeError("ingestion must be an IngestionResult")
        build_lock = self._lock_for(ingestion.index_id)
        with build_lock:
            now = self.clock()
            with self._lock:
                self._ensure_open()
                expired = self._take_expired(now)
                existing = self._reactivate(ingestion.index_id, now)
                if existing is not None:
                    evicted = self._take_over_capacity()
                else:
                    evicted = ()
            self._close_many((*expired, *evicted))
            if existing is not None:
                return existing.index_ref

            managed = self._create_managed(ingestion, now)
            with self._lock:
                if self._closed:
                    managed.close()
                    raise RuntimeError("index manager is closed")
                self._indexes[ingestion.index_id] = managed
                evicted = self._take_over_capacity()
            self._close_many(evicted)
            return managed.index_ref

    def get(self, index_id: str) -> HybridRetriever:
        now = self.clock()
        with self._lock:
            self._ensure_open()
            expired = self._take_expired(now)
            managed = self._reactivate(index_id, now)
            if managed is not None:
                evicted = self._take_over_capacity()
            else:
                evicted = ()
        self._close_many((*expired, *evicted))
        if managed is None:
            raise KeyError("知识库索引未加载或已从缓存释放。")
        return managed.retriever

    @contextmanager
    def lease(self, index_id: str) -> Iterator[HybridRetriever]:
        """Keep an index alive for the complete retrieval operation."""

        now = self.clock()
        with self._lock:
            self._ensure_open()
            expired = self._take_expired(now)
            managed = self._reactivate(index_id, now)
            if managed is not None:
                managed.active_leases += 1
                evicted = self._take_over_capacity()
            else:
                evicted = ()
        self._close_many((*expired, *evicted))
        if managed is None:
            raise KeyError("知识库索引未加载或已从缓存释放。")
        try:
            yield managed.retriever
        finally:
            close_now = False
            with self._lease_changed:
                managed.active_leases -= 1
                if managed.active_leases < 0:
                    raise RuntimeError("index lease accounting is inconsistent")
                if managed.active_leases == 0:
                    self._lease_changed.notify_all()
                    if managed.retire_action == "close":
                        managed.retire_action = None
                        if self._retiring.get(index_id) is managed:
                            self._retiring.pop(index_id, None)
                        close_now = True
            if close_now:
                managed.close()

    def delete(self, index_id: str) -> bool:
        """Permanently remove one active or persisted index."""

        with self._lock_for(index_id):
            with self._lease_changed:
                managed = self._indexes.pop(index_id, None)
                if managed is None:
                    managed = self._retiring.get(index_id)
            if managed is not None:
                with self._lease_changed:
                    managed.retire_action = "delete"
                    while managed.active_leases:
                        self._lease_changed.wait()
                    managed.retire_action = None
                    if self._retiring.get(index_id) is managed:
                        self._retiring.pop(index_id, None)
                managed.delete()
                return True
            repository_delete = getattr(self.repository, "delete", None)
            return bool(repository_delete and repository_delete(index_id))

    def close(self) -> None:
        """Release cached indexes without deleting durable collections."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            managed_indexes = tuple(self._indexes.values())
            self._indexes.clear()
            ready = self._stage_close(managed_indexes)
        self._close_many(ready)

    def healthcheck(self) -> bool:
        """Verify the manager and backing repository without loading a model."""

        with self._lock:
            if self._closed:
                return False
        try:
            return self.repository.healthcheck() is True
        except Exception:
            return False

    def stats(self) -> dict[str, int]:
        with self._lock:
            managed_indexes = (*self._indexes.values(), *self._retiring.values())
            return {
                "active_indexes": len(managed_indexes),
                "documents": sum(item.index_ref.document_count for item in managed_indexes),
                "chunks": sum(item.index_ref.chunk_count for item in managed_indexes),
            }

    def _create_managed(self, ingestion: IngestionResult, now: float) -> ManagedIndex:
        vector_index = self.repository.build(ingestion.index_id, ingestion.chunks)
        retriever = HybridRetriever(
            vector_index,
            ingestion.chunks,
            self.settings,
            reranker=self.reranker,
        )
        return ManagedIndex(
            index_ref=vector_index.index_ref,
            retriever=retriever,
            last_accessed=now,
            close=vector_index.close,
            delete=getattr(vector_index, "delete", vector_index.close),
        )

    def _take_expired(self, now: float) -> tuple[ManagedIndex, ...]:
        expired_ids = [
            index_id
            for index_id, managed in self._indexes.items()
            if now - managed.last_accessed >= self.settings.session_ttl_seconds
        ]
        expired = tuple(self._indexes.pop(index_id) for index_id in expired_ids)
        return self._stage_close(expired)

    def _take_over_capacity(self) -> tuple[ManagedIndex, ...]:
        evicted: list[ManagedIndex] = []
        while len(self._indexes) > self.settings.max_sessions:
            _, managed = self._indexes.popitem(last=False)
            evicted.append(managed)
        return self._stage_close(evicted)

    def _stage_close(self, indexes: Sequence[ManagedIndex]) -> tuple[ManagedIndex, ...]:
        """Register leased indexes before they leave the active cache."""

        ready: list[ManagedIndex] = []
        for managed in indexes:
            if managed.retire_action == "delete":
                continue
            if managed.active_leases:
                managed.retire_action = "close"
                self._retiring[managed.index_ref.index_id] = managed
            else:
                managed.retire_action = None
                ready.append(managed)
        return tuple(ready)

    def _reactivate(self, index_id: str, now: float) -> ManagedIndex | None:
        managed = self._indexes.get(index_id)
        if managed is None:
            retiring = self._retiring.get(index_id)
            if retiring is not None and retiring.retire_action == "close":
                managed = self._retiring.pop(index_id)
                managed.retire_action = None
                self._indexes[index_id] = managed
        if managed is not None:
            managed.last_accessed = now
            self._indexes.move_to_end(index_id)
        return managed

    @staticmethod
    def _close_many(indexes: Sequence[ManagedIndex]) -> None:
        for managed in indexes:
            managed.close()

    def _lock_for(self, index_id: str) -> threading.Lock:
        if not isinstance(index_id, str) or not index_id:
            raise ValueError("index_id cannot be empty")
        slot = sum(index_id.encode("utf-8")) % len(self._build_locks)
        return self._build_locks[slot]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("index manager is closed")
