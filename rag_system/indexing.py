"""Durable indexing state machine and its failure compensation."""

from __future__ import annotations

from rag_system.application import (
    KnowledgeBaseNotReadyError,
    PlatformIntegrityError,
    PlatformUnavailableError,
)
from rag_system.application_ports import KnowledgeBaseRepository, KnowledgeService
from rag_system.assets import KnowledgeBaseAssets
from rag_system.catalog import (
    KnowledgeBaseErrorCode,
    KnowledgeBaseRecord,
    KnowledgeBaseStatus,
)
from rag_system.file_store import FileStoreError
from rag_system.jobs import CancellationToken, JobCancelledError
from rag_system.metrics import OperationalMetrics
from rag_system.security import DocumentValidationError
from rag_system.tenancy import Principal


class KnowledgeBaseIndexer:
    """Build an index while keeping catalog state and cleanup semantics coherent."""

    def __init__(
        self,
        *,
        service: KnowledgeService,
        catalog: KnowledgeBaseRepository,
        assets: KnowledgeBaseAssets,
        metrics: OperationalMetrics,
    ) -> None:
        self._service = service
        self._catalog = catalog
        self._assets = assets
        self._metrics = metrics

    def run(
        self,
        principal: Principal,
        resource_id: str,
        token: CancellationToken,
        *,
        namespace: str,
    ) -> dict[str, str | int]:
        built_index_id = ""
        try:
            token.raise_if_cancelled()
            record = self._catalog.get(principal, resource_id)
            paths = self._assets.resolve(principal, record)
            prepared = self._service.prepare_index(
                [str(path) for path in paths],
                namespace=namespace,
            )
            token.raise_if_cancelled()
            if record.status is KnowledgeBaseStatus.PENDING:
                record = self._catalog.transition(
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
            index_ref = self._service.create_prepared_index(prepared)
            token.raise_if_cancelled()
            self._catalog.transition(
                principal,
                resource_id,
                KnowledgeBaseStatus.READY,
                chunk_count=index_ref.chunk_count,
            )
            built_index_id = ""
            self._record_success()
            return {
                "knowledge_base_id": resource_id,
                "status": KnowledgeBaseStatus.READY.value,
                "document_count": index_ref.document_count,
                "chunk_count": index_ref.chunk_count,
            }
        except DocumentValidationError:
            self._fail(
                principal,
                resource_id,
                built_index_id,
                KnowledgeBaseErrorCode.CONTENT_REJECTED,
            )
            raise
        except FileStoreError:
            self._fail(
                principal,
                resource_id,
                built_index_id,
                KnowledgeBaseErrorCode.INDEX_STORAGE_FAILED,
            )
            raise
        except JobCancelledError:
            self._fail(
                principal,
                resource_id,
                built_index_id,
                KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
            raise
        except Exception:
            cancellation_requested = token.cancelled or self.cancel_intent_exists(
                principal,
                resource_id,
            )
            failure_code = (
                KnowledgeBaseErrorCode.INDEX_CANCELLED
                if cancellation_requested
                else KnowledgeBaseErrorCode.INDEX_BUILD_FAILED
            )
            self._fail(principal, resource_id, built_index_id, failure_code)
            if cancellation_requested:
                raise JobCancelledError() from None
            raise

    def mark_failed(
        self,
        principal: Principal,
        resource_id: str,
        error_code: KnowledgeBaseErrorCode,
    ) -> None:
        try:
            record = self._catalog.get(principal, resource_id)
            if record.status in {
                KnowledgeBaseStatus.PENDING,
                KnowledgeBaseStatus.INDEXING,
                KnowledgeBaseStatus.CANCELLING,
            }:
                self._catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.FAILED,
                    error_code=error_code,
                )
        except Exception:
            return

    def persist_cancel_intent(self, principal: Principal, resource_id: str) -> None:
        try:
            record = self._catalog.get(principal, resource_id)
            if record.status in {
                KnowledgeBaseStatus.PENDING,
                KnowledgeBaseStatus.INDEXING,
            }:
                self._catalog.transition(
                    principal,
                    resource_id,
                    KnowledgeBaseStatus.CANCELLING,
                )
        except Exception:
            try:
                current = self._catalog.get(principal, resource_id)
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

    def cancel_intent_exists(self, principal: Principal, resource_id: str) -> bool:
        try:
            return (
                self._catalog.get(principal, resource_id).status
                is KnowledgeBaseStatus.CANCELLING
            )
        except Exception:
            return False

    def converge_cancel_intent(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> KnowledgeBaseRecord:
        if record.status is not KnowledgeBaseStatus.CANCELLING:
            return record
        try:
            return self._catalog.transition(
                principal,
                record.resource_id,
                KnowledgeBaseStatus.FAILED,
                error_code=KnowledgeBaseErrorCode.INDEX_CANCELLED,
            )
        except Exception:
            try:
                current = self._catalog.get(principal, record.resource_id)
            except Exception:
                current = None
            if current is not None and current.status is KnowledgeBaseStatus.FAILED:
                return current
            raise PlatformUnavailableError(
                "knowledge base cancellation recovery did not complete"
            ) from None

    def cleanup_uncommitted_index(self, index_id: str) -> None:
        if not index_id:
            return
        try:
            self._service.index_manager.delete(index_id)
        except Exception:
            # The FAILED catalog tombstone retains internal_index_id, allowing
            # an operator-initiated resource delete to retry durable cleanup.
            return

    def _fail(
        self,
        principal: Principal,
        resource_id: str,
        index_id: str,
        error_code: KnowledgeBaseErrorCode,
    ) -> None:
        self.cleanup_uncommitted_index(index_id)
        self.mark_failed(principal, resource_id, error_code)
        self._metrics.index_tasks_total.increment(
            labels={"operation": "build", "outcome": "error"}
        )

    def _record_success(self) -> None:
        self._metrics.index_tasks_total.increment(
            labels={"operation": "build", "outcome": "success"}
        )


__all__ = ["KnowledgeBaseIndexer"]
