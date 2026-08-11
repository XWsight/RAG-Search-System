"""Production application service for tenant-scoped knowledge bases."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

from rag_system.catalog import (
    DocumentManifest,
    KnowledgeBaseCatalog,
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)
from rag_system.config import Settings
from rag_system.domain import AnswerRequest, AnswerResult
from rag_system.file_store import FileStoreError, TenantFileStore
from rag_system.idempotency import (
    IdempotencyConflictError,
    IdempotencyStore,
    IdempotencyUnavailableError,
)
from rag_system.jobs import (
    CancellationToken,
    JobCancelledError,
    JobError,
    JobId,
    JobManager,
    JobNotFoundError,
    JobSnapshot,
)
from rag_system.metrics import OperationalMetrics, create_operational_metrics
from rag_system.security import DocumentValidationError
from rag_system.tenancy import Principal

if TYPE_CHECKING:
    from rag_system.service import RagService


class PlatformError(RuntimeError):
    """Base class for errors that can be safely classified by an API."""

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
    display_name: str
    source: bytes | bytearray | memoryview | BinaryIO


@dataclass(frozen=True, slots=True)
class KnowledgeBaseSubmission:
    knowledge_base: KnowledgeBaseRecord
    job_id: JobId
    replayed: bool = False


class RagPlatform:
    """Coordinate catalog, file storage, jobs, indexes, and answering.

    The platform is deployable as one durable node: SQLite, uploaded files,
    and Chroma collections share one storage volume. Expensive indexing runs
    outside the request thread and all resource operations are tenant-scoped.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        service: RagService,
        catalog: KnowledgeBaseCatalog,
        file_store: TenantFileStore,
        jobs: JobManager,
        idempotency: IdempotencyStore,
        metrics: OperationalMetrics | None = None,
        document_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not callable(document_id_factory):
            raise TypeError("document_id_factory must be callable")
        self.settings = settings.validate()
        self.service = service
        self.catalog = catalog
        self.file_store = file_store
        self.jobs = jobs
        self.idempotency = idempotency
        self.metrics = metrics or create_operational_metrics()
        self._document_id_factory = document_id_factory
        self._resource_locks = tuple(threading.RLock() for _ in range(64))
        self._job_map_lock = threading.RLock()
        self._job_by_resource: dict[tuple[str, str], JobId] = {}
        self._answer_slots = threading.BoundedSemaphore(
            self.settings.max_concurrent_answers
        )

    def create_knowledge_base(
        self,
        principal: Principal,
        *,
        display_name: str,
        documents: Sequence[UploadDocument],
        idempotency_key: str,
    ) -> KnowledgeBaseSubmission:
        uploads = self._materialize_uploads(documents)
        if not uploads:
            raise PlatformValidationError("at least one document is required")
        if len(uploads) > self.settings.max_documents:
            raise PlatformValidationError("document count exceeds the configured limit")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise PlatformValidationError("idempotency key is required")

        request_digest = self._create_request_digest(display_name, uploads)
        reservation = self.idempotency.reserve(
            principal,
            "knowledge_base.create",
            idempotency_key.strip(),
            request_digest,
        )
        if not reservation.created:
            if not reservation.is_bound:
                record = self.catalog.find_by_idempotency_reservation(
                    principal,
                    reservation.reservation_id,
                )
                if record is None:
                    raise IdempotencyInProgressError(
                        "matching request is still in progress"
                    )
                current_job = self._job_for_resource(principal, record.resource_id)
                if (
                    record.status is KnowledgeBaseStatus.CANCELLING
                    and self._job_is_terminal(principal, current_job)
                ):
                    record = self._converge_cancel_intent(principal, record)
                    current_job = None
                if current_job is None and record.status in {
                    KnowledgeBaseStatus.READY,
                    KnowledgeBaseStatus.FAILED,
                }:
                    current_job = self._submit_status_job(principal, record)
                if current_job is None:
                    raise IdempotencyInProgressError(
                        "matching request recovery is still in progress"
                    )
                self._recover_idempotency_binding(principal, record, current_job)
                return KnowledgeBaseSubmission(record, current_job, replayed=True)
            record = self.catalog.get(principal, reservation.resource_id)
            current_job = self._job_for_resource(principal, record.resource_id)
            if current_job is None:
                try:
                    self.jobs.get(
                        principal.tenant_id.value,
                        JobId(reservation.job_id),
                    )
                    current_job = JobId(reservation.job_id)
                except JobNotFoundError:
                    current_job = None
            if (
                record.status is KnowledgeBaseStatus.CANCELLING
                and self._job_is_terminal(principal, current_job)
            ):
                record = self._converge_cancel_intent(principal, record)
                current_job = None
            if current_job is None and record.status in {
                KnowledgeBaseStatus.READY,
                KnowledgeBaseStatus.FAILED,
            }:
                current_job = self._submit_status_job(principal, record)
                self._recover_idempotency_binding(
                    principal,
                    record,
                    current_job,
                )
            if current_job is None:
                raise IdempotencyInProgressError(
                    "matching request recovery is still in progress"
                )
            return KnowledgeBaseSubmission(
                record,
                current_job,
                replayed=True,
            )

        record: KnowledgeBaseRecord | None = None
        job_id: JobId | None = None
        try:
            record = self.catalog.create(
                principal,
                display_name,
                idempotency_reservation_id=reservation.reservation_id,
            )
            planned: list[tuple[str, UploadDocument, DocumentManifest]] = []
            for upload in uploads:
                document_resource_id = self._new_document_id()
                relative_path = self.file_store.planned_relative_path(
                    principal.tenant_id.value,
                    document_resource_id,
                    upload.display_name,
                )
                content = bytes(upload.source)
                manifest_item = DocumentManifest(
                    display_name=PurePosixPath(relative_path).name,
                    relative_path=relative_path,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
                planned.append((document_resource_id, upload, manifest_item))
            record = self.catalog.replace_manifest(
                principal,
                record.resource_id,
                tuple(item for _, _, item in planned),
            )
            for document_resource_id, upload, manifest_item in planned:
                saved = self.file_store.save(
                    principal.tenant_id.value,
                    document_resource_id,
                    upload.display_name,
                    upload.source,
                )
                if (
                    saved.display_name != manifest_item.display_name
                    or saved.relative_path != manifest_item.relative_path
                    or saved.size != manifest_item.size_bytes
                    or saved.sha256 != manifest_item.sha256
                ):
                    raise PlatformIntegrityError("stored document does not match its manifest")
            job_id = self._submit_indexing(
                principal,
                record.resource_id,
                idempotency_key=reservation.reservation_id,
            )
            self.idempotency.bind_result(
                principal,
                reservation.reservation_id,
                record.resource_id,
                job_id.value,
            )
        except Exception:
            if record is not None:
                if job_id is None:
                    self._rollback_create(principal, record)
                else:
                    try:
                        self.delete_knowledge_base(principal, record.resource_id)
                    except Exception:
                        pass
            try:
                self.idempotency.abandon(principal, reservation.reservation_id)
            except Exception:
                pass
            raise
        if record is None or job_id is None:
            raise PlatformUnavailableError("knowledge base submission did not complete")
        return KnowledgeBaseSubmission(record, job_id, replayed=False)

    def get_knowledge_base(
        self,
        principal: Principal,
        resource_id: str,
    ) -> KnowledgeBaseRecord:
        return self.catalog.get(principal, resource_id)

    def list_knowledge_bases(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        return self.catalog.list(principal, limit=limit, offset=offset)

    def get_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        return self.jobs.get(principal.tenant_id.value, job_id)

    def cancel_job(self, principal: Principal, job_id: JobId | str) -> JobSnapshot:
        snapshot = self.jobs.get(principal.tenant_id.value, job_id)
        resource_id = self._resource_for_job(principal, snapshot.job_id)
        if resource_id is not None:
            self._persist_cancel_intent(principal, resource_id)
        snapshot = self.jobs.cancel(principal.tenant_id.value, snapshot.job_id)
        if resource_id is not None:
            self._mark_failed(
                principal,
                resource_id,
                KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
        return snapshot

    def answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        if not self._answer_slots.acquire(blocking=False):
            raise PlatformUnavailableError("answer capacity is temporarily exhausted")
        try:
            return self._answer(principal, resource_id, request)
        finally:
            self._answer_slots.release()

    def _answer(
        self,
        principal: Principal,
        resource_id: str,
        request: AnswerRequest,
    ) -> AnswerResult:
        record = self.catalog.get(principal, resource_id)
        if record.status is not KnowledgeBaseStatus.READY or not record.internal_index_id:
            raise KnowledgeBaseNotReadyError("knowledge base is not ready")

        try:
            self.service.index_manager.get(record.internal_index_id)
        except KeyError:
            with self._lock_for(resource_id):
                record = self.catalog.get(principal, resource_id)
                if record.status is not KnowledgeBaseStatus.READY or not record.internal_index_id:
                    raise KnowledgeBaseNotReadyError("knowledge base is not ready") from None
                paths = self._resolve_documents(principal, record)
                restored = self.service.create_index(
                    [str(path) for path in paths],
                    namespace=self._namespace(principal, resource_id),
                )
                if restored.index_id != record.internal_index_id:
                    self.service.index_manager.delete(restored.index_id)
                    raise PlatformIntegrityError(
                        "stored index identity does not match its catalog"
                    ) from None

        scoped_request = replace(
            request,
            session_id=self._session_id(principal, resource_id, request.session_id),
        )
        try:
            return self.service.answer(record.internal_index_id, scoped_request)
        except KeyError:
            current = self.catalog.get(principal, resource_id)
            if current.status is not KnowledgeBaseStatus.READY:
                raise KnowledgeBaseNotReadyError("knowledge base is not ready") from None
            raise KnowledgeBaseNotReadyError("knowledge base index is being reloaded") from None

    def clear_session(
        self,
        principal: Principal,
        resource_id: str,
        session_id: str,
    ) -> bool:
        self.catalog.get(principal, resource_id)
        return self.service.clear_session(self._session_id(principal, resource_id, session_id))

    def delete_knowledge_base(self, principal: Principal, resource_id: str) -> bool:
        job_id = self._job_for_resource(principal, resource_id)
        if job_id is not None:
            try:
                self.jobs.cancel(principal.tenant_id.value, job_id)
            except JobError:
                pass

        with self._lock_for(resource_id):
            record = self.catalog.get(principal, resource_id)
            if record.status is not KnowledgeBaseStatus.DELETING:
                record = self.catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.DELETING,
                )
            if record.internal_index_id:
                self.service.index_manager.delete(record.internal_index_id)
            for document in record.documents:
                document_id = self._document_resource_id(principal, document)
                self.file_store.delete(principal.tenant_id.value, document_id)
            self.catalog.delete(principal, resource_id)
            with self._job_map_lock:
                self._job_by_resource.pop((principal.tenant_id.value, resource_id), None)
            return True

    def recover_incomplete(self, principals: Sequence[Principal]) -> int:
        """Resubmit durable pending/indexing records after process restart."""

        recovered = 0
        seen_tenants: set[str] = set()
        for principal in principals:
            tenant = principal.tenant_id.value
            if tenant in seen_tenants:
                continue
            seen_tenants.add(tenant)
            tenant_records: list[KnowledgeBaseRecord] = []
            offset = 0
            while True:
                records = self.catalog.list(principal, limit=100, offset=offset)
                tenant_records.extend(records)
                if len(records) < 100:
                    break
                offset += len(records)
            for record in tenant_records:
                if record.status is KnowledgeBaseStatus.DELETING:
                    self.delete_knowledge_base(principal, record.resource_id)
                    recovered += 1
                elif record.status is KnowledgeBaseStatus.CANCELLING:
                    self.catalog.transition(
                        principal,
                        record.resource_id,
                        KnowledgeBaseStatus.FAILED,
                        error_code=KnowledgeBaseErrorCode.INDEX_CANCELLED,
                    )
                    recovered += 1
                elif record.status in {
                    KnowledgeBaseStatus.PENDING,
                    KnowledgeBaseStatus.INDEXING,
                }:
                    job_id = self._submit_indexing(
                        principal,
                        record.resource_id,
                        idempotency_key=f"recovery:{record.resource_id}:{record.version}",
                    )
                    self._recover_idempotency_binding(principal, record, job_id)
                    recovered += 1
        return recovered

    def _submit_status_job(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> JobId:
        def task(token: CancellationToken) -> dict[str, str | int]:
            token.raise_if_cancelled()
            return {
                "knowledge_base_id": record.resource_id,
                "status": record.status.value,
                "document_count": record.document_count,
                "chunk_count": record.chunk_count,
            }

        job_id = self.jobs.submit(
            principal.tenant_id.value,
            task,
            idempotency_key=f"recovered-state:{record.resource_id}:{record.version}",
        )
        with self._job_map_lock:
            self._job_by_resource[
                (principal.tenant_id.value, record.resource_id)
            ] = job_id
        return job_id

    def _recover_idempotency_binding(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
        job_id: JobId,
    ) -> None:
        reservation_id = record.idempotency_reservation_id
        if reservation_id is None:
            return
        try:
            self.idempotency.recover_binding(
                principal,
                reservation_id,
                record.resource_id,
                job_id.value,
            )
        except IdempotencyUnavailableError:
            return
        except IdempotencyConflictError:
            raise PlatformIntegrityError(
                "idempotency binding points to a different durable resource"
            ) from None

    def close(self) -> None:
        try:
            self.jobs.shutdown(wait=True, cancel_pending=True)
        finally:
            try:
                self.service.index_manager.close()
            finally:
                close_service = getattr(self.service, "close", None)
                if callable(close_service):
                    close_service()

    def _submit_indexing(
        self,
        principal: Principal,
        resource_id: str,
        *,
        idempotency_key: str,
    ) -> JobId:
        def task(token: CancellationToken):
            with self._lock_for(resource_id):
                return self._index_task(principal, resource_id, token)

        job_id = self.jobs.submit(
            principal.tenant_id.value,
            task,
            idempotency_key=idempotency_key,
        )
        with self._job_map_lock:
            self._job_by_resource[(principal.tenant_id.value, resource_id)] = job_id
        return job_id

    def _index_task(
        self,
        principal: Principal,
        resource_id: str,
        token: CancellationToken,
    ) -> dict[str, str | int]:
        built_index_id = ""
        try:
            token.raise_if_cancelled()
            record = self.catalog.get(principal, resource_id)
            paths = self._resolve_documents(principal, record)
            prepared = self.service.prepare_index(
                [str(path) for path in paths],
                namespace=self._namespace(principal, resource_id),
            )
            token.raise_if_cancelled()
            if record.status is KnowledgeBaseStatus.PENDING:
                record = self.catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.INDEXING,
                    internal_index_id=prepared.index_id,
                )
            elif record.status is KnowledgeBaseStatus.INDEXING:
                if record.internal_index_id != prepared.index_id:
                    raise PlatformIntegrityError("index identity changed during recovery")
            else:
                raise KnowledgeBaseNotReadyError("knowledge base cannot be indexed")

            built_index_id = prepared.index_id
            index_ref = self.service.create_prepared_index(prepared)
            token.raise_if_cancelled()
            self.catalog.transition(
                principal,
                resource_id,
                KnowledgeBaseStatus.READY,
                chunk_count=index_ref.chunk_count,
            )
            built_index_id = ""
            self.metrics.index_tasks_total.increment(
                labels={"operation": "build", "outcome": "success"}
            )
            return {
                "knowledge_base_id": resource_id,
                "status": KnowledgeBaseStatus.READY.value,
                "document_count": index_ref.document_count,
                "chunk_count": index_ref.chunk_count,
            }
        except DocumentValidationError:
            self._cleanup_uncommitted_index(built_index_id)
            self._mark_failed(
                principal,
                resource_id,
                KnowledgeBaseErrorCode.CONTENT_REJECTED,
            )
            self._record_index_failure()
            raise
        except FileStoreError:
            self._cleanup_uncommitted_index(built_index_id)
            self._mark_failed(
                principal,
                resource_id,
                KnowledgeBaseErrorCode.INDEX_STORAGE_FAILED,
            )
            self._record_index_failure()
            raise
        except JobCancelledError:
            self._cleanup_uncommitted_index(built_index_id)
            self._mark_failed(
                principal,
                resource_id,
                KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
            self._record_index_failure()
            raise
        except Exception:
            self._cleanup_uncommitted_index(built_index_id)
            cancellation_requested = token.cancelled or self._cancel_intent_exists(
                principal,
                resource_id,
            )
            failure_code = (
                KnowledgeBaseErrorCode.INDEX_CANCELLED
                if cancellation_requested
                else KnowledgeBaseErrorCode.INDEX_BUILD_FAILED
            )
            self._mark_failed(
                principal,
                resource_id,
                failure_code,
            )
            self._record_index_failure()
            if cancellation_requested:
                raise JobCancelledError() from None
            raise

    def _cleanup_uncommitted_index(self, index_id: str) -> None:
        if not index_id:
            return
        try:
            self.service.index_manager.delete(index_id)
        except Exception:
            # The FAILED catalog tombstone retains internal_index_id, so an
            # operator-initiated resource delete can retry durable cleanup.
            return

    def _record_index_failure(self) -> None:
        self.metrics.index_tasks_total.increment(
            labels={"operation": "build", "outcome": "error"}
        )

    def _mark_failed(
        self,
        principal: Principal,
        resource_id: str,
        error_code: KnowledgeBaseErrorCode,
    ) -> None:
        try:
            record = self.catalog.get(principal, resource_id)
            if record.status in {
                KnowledgeBaseStatus.PENDING,
                KnowledgeBaseStatus.INDEXING,
                KnowledgeBaseStatus.CANCELLING,
            }:
                self.catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.FAILED,
                    error_code=error_code,
                )
        except Exception:
            return

    def _persist_cancel_intent(
        self,
        principal: Principal,
        resource_id: str,
    ) -> None:
        try:
            record = self.catalog.get(principal, resource_id)
            if record.status in {
                KnowledgeBaseStatus.PENDING,
                KnowledgeBaseStatus.INDEXING,
            }:
                self.catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.CANCELLING,
                )
        except Exception:
            try:
                current = self.catalog.get(principal, resource_id)
            except Exception:
                current = None
            if current is not None and current.status in {
                KnowledgeBaseStatus.CANCELLING,
                KnowledgeBaseStatus.READY,
                KnowledgeBaseStatus.FAILED,
                KnowledgeBaseStatus.DELETING,
            }:
                return
            raise PlatformUnavailableError(
                "knowledge base cancellation could not be persisted"
            ) from None

    def _cancel_intent_exists(self, principal: Principal, resource_id: str) -> bool:
        try:
            return (
                self.catalog.get(principal, resource_id).status
                is KnowledgeBaseStatus.CANCELLING
            )
        except Exception:
            return False

    def _converge_cancel_intent(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> KnowledgeBaseRecord:
        """Finish a durable cancellation whose in-memory job no longer exists."""

        if record.status is not KnowledgeBaseStatus.CANCELLING:
            return record
        try:
            return self.catalog.transition(
                principal,
                record.resource_id,
                KnowledgeBaseStatus.FAILED,
                error_code=KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
        except Exception:
            try:
                current = self.catalog.get(principal, record.resource_id)
            except Exception:
                current = None
            if current is not None and current.status is KnowledgeBaseStatus.FAILED:
                return current
            raise PlatformUnavailableError(
                "knowledge base cancellation recovery did not complete"
            ) from None

    def _resolve_documents(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> tuple[Path, ...]:
        resolved: list[Path] = []
        for document in record.documents:
            document_id = self._document_resource_id(principal, document)
            path = self.file_store.resolve(principal.tenant_id.value, document_id)
            relative = path.resolve(strict=True).relative_to(self.file_store.root).as_posix()
            if relative != document.relative_path or path.name != document.display_name:
                raise PlatformIntegrityError("document path does not match its manifest")
            size, digest = self._file_identity(path)
            if size != document.size_bytes or digest != document.sha256:
                raise PlatformIntegrityError("document content does not match its manifest")
            resolved.append(path)
        if not resolved:
            raise PlatformIntegrityError("knowledge base has no documents")
        return tuple(resolved)

    def _document_resource_id(
        self,
        principal: Principal,
        document: DocumentManifest,
    ) -> str:
        parts = PurePosixPath(document.relative_path).parts
        expected_tenant = "tenant-" + hashlib.sha256(
            principal.tenant_id.value.encode("utf-8")
        ).hexdigest()
        if len(parts) != 3 or parts[0] != expected_tenant:
            raise PlatformIntegrityError("document manifest is outside the tenant boundary")
        return parts[1]

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as source:
                while block := source.read(64 * 1024):
                    size += len(block)
                    digest.update(block)
        except OSError:
            raise PlatformIntegrityError("document could not be verified") from None
        return size, digest.hexdigest()

    def _rollback_create(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> None:
        try:
            current = self.catalog.get(principal, record.resource_id)
            if current.status is not KnowledgeBaseStatus.DELETING:
                current = self.catalog.transition(
                    principal,
                    record.resource_id,
                    KnowledgeBaseStatus.DELETING,
                )
        except Exception:
            return

        cleanup_succeeded = True
        for document in current.documents:
            document_id = self._document_resource_id(principal, document)
            try:
                self.file_store.delete(principal.tenant_id.value, document_id)
            except FileStoreError:
                cleanup_succeeded = False
        if cleanup_succeeded:
            try:
                self.catalog.delete(principal, record.resource_id)
            except Exception:
                return

    def _new_document_id(self) -> str:
        value = self._document_id_factory()
        if not isinstance(value, str):
            raise PlatformUnavailableError("document identifier generation failed")
        normalized = value.replace("-", "")
        if len(normalized) < 16 or not normalized.isalnum():
            raise PlatformUnavailableError("document identifier generation failed")
        return f"doc_{normalized[:48]}"

    def _materialize_uploads(
        self,
        documents: Sequence[UploadDocument],
    ) -> tuple[UploadDocument, ...]:
        uploads = tuple(documents)
        if not uploads:
            return ()
        materialized: list[UploadDocument] = []
        total = 0
        for upload in uploads:
            if not isinstance(upload, UploadDocument):
                raise PlatformValidationError("documents must be UploadDocument values")
            content = self._read_upload(upload.source)
            total += len(content)
            if total > self.settings.max_total_bytes:
                raise PlatformValidationError("total upload size exceeds the configured limit")
            materialized.append(UploadDocument(upload.display_name, content))
        return tuple(materialized)

    def _read_upload(
        self,
        source: bytes | bytearray | memoryview | BinaryIO,
    ) -> bytes:
        if isinstance(source, (bytes, bytearray, memoryview)):
            content = bytes(source)
            if len(content) > self.settings.max_file_bytes:
                raise PlatformValidationError("file size exceeds the configured limit")
            return content
        reader = getattr(source, "read", None)
        if not callable(reader):
            raise PlatformValidationError("upload source must be binary")
        chunks: list[bytes] = []
        size = 0
        try:
            while True:
                block = reader(min(64 * 1024, self.settings.max_file_bytes - size + 1))
                if block in (b"", None):
                    break
                if not isinstance(block, (bytes, bytearray, memoryview)):
                    raise PlatformValidationError("upload source must return bytes")
                normalized = bytes(block)
                size += len(normalized)
                if size > self.settings.max_file_bytes:
                    raise PlatformValidationError("file size exceeds the configured limit")
                chunks.append(normalized)
        except PlatformValidationError:
            raise
        except Exception:
            raise PlatformValidationError("upload could not be read") from None
        return b"".join(chunks)

    @staticmethod
    def _create_request_digest(
        display_name: str,
        uploads: Sequence[UploadDocument],
    ) -> str:
        if not isinstance(display_name, str):
            raise PlatformValidationError("display_name must be a string")
        try:
            collection_name = display_name.encode("utf-8")
            identities = sorted(
                (
                    upload.display_name.encode("utf-8"),
                    hashlib.sha256(bytes(upload.source)).digest(),
                )
                for upload in uploads
            )
        except (AttributeError, TypeError, UnicodeError):
            raise PlatformValidationError("upload metadata is invalid") from None
        digest = hashlib.sha256(b"rag-create-request-v1")
        for value in (collection_name, *(part for identity in identities for part in identity)):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        return digest.hexdigest()

    @staticmethod
    def _namespace(principal: Principal, resource_id: str) -> str:
        return f"{principal.tenant_id.value}:{resource_id}"

    @staticmethod
    def _session_id(principal: Principal, resource_id: str, session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise PlatformValidationError("session_id is required")
        if len(session_id) > 128:
            raise PlatformValidationError("session_id is too long")
        identity = f"{principal.tenant_id.value}\0{resource_id}\0{session_id.strip()}"
        return "session_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _lock_for(self, resource_id: str) -> threading.RLock:
        slot = sum(resource_id.encode("utf-8")) % len(self._resource_locks)
        return self._resource_locks[slot]

    def _job_for_resource(self, principal: Principal, resource_id: str) -> JobId | None:
        identity = (principal.tenant_id.value, resource_id)
        with self._job_map_lock:
            job_id = self._job_by_resource.get(identity)
        if job_id is None:
            return None
        try:
            self.jobs.get(principal.tenant_id.value, job_id)
        except JobNotFoundError:
            with self._job_map_lock:
                if self._job_by_resource.get(identity) == job_id:
                    self._job_by_resource.pop(identity, None)
            return None
        return job_id

    def _job_is_terminal(self, principal: Principal, job_id: JobId | None) -> bool:
        if job_id is None:
            return True
        try:
            return self.jobs.get(principal.tenant_id.value, job_id).status.terminal
        except JobNotFoundError:
            return True

    def _resource_for_job(self, principal: Principal, job_id: JobId) -> str | None:
        with self._job_map_lock:
            for (tenant_id, resource_id), mapped_job in self._job_by_resource.items():
                if tenant_id == principal.tenant_id.value and mapped_job == job_id:
                    return resource_id
        return None
