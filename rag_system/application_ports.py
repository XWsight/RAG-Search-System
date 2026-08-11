"""Replaceable persistence and execution ports used by application workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from rag_system.catalog import (
    DocumentManifest,
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)
from rag_system.domain import AnswerRequest, AnswerResult, IndexRef
from rag_system.idempotency import IdempotencyReservation
from rag_system.ingestion import IngestionResult
from rag_system.job_contracts import CancellationToken, JobId, JobSnapshot
from rag_system.ports import Retriever
from rag_system.tenancy import Principal


class KnowledgeBaseRepository(Protocol):
    def create(
        self,
        principal: Principal,
        display_name: str,
        documents: Sequence[DocumentManifest] = (),
        *,
        idempotency_reservation_id: str | None = None,
    ) -> KnowledgeBaseRecord: ...

    def get(self, principal: Principal, resource_id: str) -> KnowledgeBaseRecord: ...

    def find_by_idempotency_reservation(
        self,
        principal: Principal,
        reservation_id: str,
    ) -> KnowledgeBaseRecord | None: ...

    def list(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]: ...

    def replace_manifest(
        self,
        principal: Principal,
        resource_id: str,
        documents: Sequence[DocumentManifest],
    ) -> KnowledgeBaseRecord: ...

    def transition(
        self,
        principal: Principal,
        resource_id: str,
        target: KnowledgeBaseStatus,
        *,
        internal_index_id: str | None = None,
        chunk_count: int | None = None,
        error_code: KnowledgeBaseErrorCode | None = None,
    ) -> KnowledgeBaseRecord: ...

    def delete(
        self,
        principal: Principal,
        resource_id: str,
    ) -> tuple[DocumentManifest, ...]: ...


class IdempotencyRepository(Protocol):
    def reserve(
        self,
        principal: Principal,
        operation: str,
        key: str,
        request_digest: str,
    ) -> IdempotencyReservation: ...

    def bind_result(
        self,
        principal: Principal,
        reservation_id: str,
        resource_id: str,
        job_id: str,
    ) -> IdempotencyReservation: ...

    def recover_binding(
        self,
        principal: Principal,
        reservation_id: str,
        resource_id: str,
        job_id: str,
    ) -> IdempotencyReservation: ...

    def abandon(self, principal: Principal, reservation_id: str) -> bool: ...

    def purge_expired(self) -> int: ...


JobTask = Callable[[CancellationToken], Mapping[str, Any]]


class JobExecutor(Protocol):
    def submit(
        self,
        tenant_id: str,
        task: JobTask,
        *,
        idempotency_key: str,
    ) -> JobId: ...

    def get(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot: ...

    def cancel(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot: ...

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None: ...

    def healthcheck(self) -> bool: ...


class StoredAsset(Protocol):
    @property
    def relative_path(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def sha256(self) -> str: ...


class DocumentStore(Protocol):
    @property
    def root(self) -> Path: ...

    def planned_relative_path(
        self,
        tenant_id: str,
        resource_id: str,
        display_name: str,
    ) -> str: ...

    def healthcheck(self) -> bool: ...

    def save(
        self,
        tenant_id: str,
        resource_id: str,
        display_name: str,
        source: bytes | bytearray | memoryview | BinaryIO,
    ) -> StoredAsset: ...

    def resolve(self, tenant_id: str, resource_id: str) -> Path: ...

    def delete(self, tenant_id: str, resource_id: str) -> bool: ...


class IndexLifecycle(Protocol):
    def get(self, index_id: str) -> Retriever: ...

    def delete(self, index_id: str) -> bool: ...

    def close(self) -> None: ...

    def healthcheck(self) -> bool: ...


class KnowledgeService(Protocol):
    @property
    def index_manager(self) -> IndexLifecycle: ...

    def create_index(
        self,
        paths: Sequence[str] | None = None,
        *,
        namespace: str = "",
    ) -> IndexRef: ...

    def prepare_index(
        self,
        paths: Sequence[str] | None = None,
        *,
        namespace: str = "",
    ) -> IngestionResult: ...

    def create_prepared_index(self, ingestion: IngestionResult) -> IndexRef: ...

    def clear_session(self, session_id: str) -> bool: ...

    def answer(self, index_id: str, request: AnswerRequest) -> AnswerResult: ...

    def close(self) -> None: ...


__all__ = [
    "DocumentStore",
    "IdempotencyRepository",
    "IndexLifecycle",
    "JobExecutor",
    "JobTask",
    "KnowledgeBaseRepository",
    "KnowledgeService",
    "StoredAsset",
]
