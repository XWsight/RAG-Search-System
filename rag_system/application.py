"""Framework-neutral application contract exposed to delivery adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO, Protocol

from rag_system.catalog import KnowledgeBaseRecord
from rag_system.config import Settings
from rag_system.domain import AnswerRequest, AnswerResult
from rag_system.jobs import JobId, JobSnapshot
from rag_system.metrics import OperationalMetrics
from rag_system.tenancy import Principal


class PlatformError(RuntimeError):
    """Base class for application failures safe to classify at a boundary."""

    code = "platform_error"


class PlatformValidationError(PlatformError, ValueError):
    code = "invalid_request"


class KnowledgeBaseNotReadyError(PlatformError):
    code = "knowledge_base_not_ready"


class PlatformIntegrityError(PlatformError):
    code = "storage_integrity_error"


class PlatformUnavailableError(PlatformError):
    code = "platform_unavailable"


class IdempotencyInProgressError(PlatformError):
    code = "idempotency_in_progress"


@dataclass(frozen=True, slots=True)
class UploadDocument:
    """Transport-independent document submitted for ingestion."""

    display_name: str
    source: bytes | bytearray | memoryview | BinaryIO


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSubmission:
    knowledge_base: KnowledgeBaseRecord
    job_id: JobId
    replayed: bool = False


class RagApplication(Protocol):
    """Use cases available to HTTP, CLI, workers, and future agent adapters."""

    settings: Settings
    metrics: OperationalMetrics

    def create_knowledge_base(
        self,
        principal: Principal,
        *,
        display_name: str,
        documents: Sequence[UploadDocument],
        idempotency_key: str,
    ) -> KnowledgeBaseSubmission: ...

    def get_knowledge_base(
        self,
        principal: Principal,
        resource_id: str,
    ) -> KnowledgeBaseRecord: ...

    def list_knowledge_bases(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]: ...

    def delete_knowledge_base(self, principal: Principal, resource_id: str) -> bool: ...

    def get_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot: ...

    def cancel_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot: ...

    def answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult: ...

    def clear_session(
        self,
        principal: Principal,
        resource_id: str,
        session_id: str,
    ) -> bool: ...

    def recover_incomplete(self, principals: Sequence[Principal]) -> int: ...

    def close(self) -> None: ...


__all__ = [
    "IdempotencyInProgressError",
    "KnowledgeBaseNotReadyError",
    "KnowledgeBaseSubmission",
    "PlatformError",
    "PlatformIntegrityError",
    "PlatformUnavailableError",
    "PlatformValidationError",
    "RagApplication",
    "UploadDocument",
]
