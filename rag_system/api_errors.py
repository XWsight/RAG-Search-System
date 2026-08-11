"""Safe mapping from application failures to the public HTTP error contract."""

from __future__ import annotations

from rag_system.application import (
    IdempotencyInProgressError,
    KnowledgeBaseNotReadyError,
    PlatformError,
    PlatformIntegrityError,
    PlatformUnavailableError,
    PlatformValidationError,
)
from rag_system.catalog import (
    CatalogSchemaError,
    CatalogStorageError,
    CatalogValidationError,
    InvalidStatusTransitionError,
    KnowledgeBaseUnavailableError,
)
from rag_system.file_store import (
    DuplicateResourceError,
    FileStoreError,
    FileStoreIOError,
    FileStoreSecurityError,
    InvalidFileNameError,
    InvalidResourceIdError,
    ResourceNotFoundError,
    StorageLimitError,
)
from rag_system.idempotency import (
    IdempotencyCapacityError,
    IdempotencyConflictError,
    IdempotencyError,
    IdempotencySchemaError,
    IdempotencyStorageError,
    IdempotencyUnavailableError,
    IdempotencyValidationError,
)
from rag_system.jobs import (
    JobCapacityError,
    JobError,
    JobManagerShutdownError,
    JobNotFoundError,
    JobSubmissionError,
)
from rag_system.loaders import DocumentLoadError
from rag_system.provider_errors import ProviderError
from rag_system.security import DocumentValidationError
from rag_system.tenancy import AuthenticationError, AuthorizationError


class ApiBoundaryError(RuntimeError):
    """An error whose code and generic message are safe for clients."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.safe_message = message
        self.headers = dict(headers or {})


APPLICATION_ERROR_TYPES: tuple[type[Exception], ...] = (
    AuthenticationError,
    AuthorizationError,
    PlatformError,
    CatalogValidationError,
    CatalogSchemaError,
    CatalogStorageError,
    KnowledgeBaseUnavailableError,
    FileStoreError,
    IdempotencyError,
    JobError,
    ProviderError,
    DocumentValidationError,
)


def classify_application_error(error: Exception) -> tuple[int, str, str]:
    if isinstance(error, AuthenticationError):
        return 401, "authentication_failed", "Authentication failed."
    if isinstance(error, AuthorizationError):
        return 404, "resource_unavailable", "Resource is unavailable."
    if isinstance(
        error,
        (KnowledgeBaseUnavailableError, JobNotFoundError, ResourceNotFoundError),
    ):
        return 404, "resource_unavailable", "Resource is unavailable."
    if isinstance(error, IdempotencyConflictError):
        return 409, "idempotency_conflict", "The idempotency key conflicts with this request."
    if isinstance(error, IdempotencyInProgressError):
        return 409, "idempotency_in_progress", "The matching request is still in progress."
    if isinstance(error, (DuplicateResourceError, InvalidStatusTransitionError)):
        return 409, "resource_conflict", "The resource cannot be changed in its current state."
    if isinstance(error, KnowledgeBaseNotReadyError):
        return 409, "knowledge_base_not_ready", "The knowledge base is not ready."
    if isinstance(error, StorageLimitError):
        return 413, "storage_limit_exceeded", "The storage limit was exceeded."
    if isinstance(
        error,
        (
            PlatformValidationError,
            CatalogValidationError,
            FileStoreSecurityError,
            IdempotencyValidationError,
            InvalidFileNameError,
            InvalidResourceIdError,
            DocumentLoadError,
            DocumentValidationError,
        ),
    ):
        return 422, "invalid_request", "The request could not be validated."
    if isinstance(
        error,
        (
            IdempotencyCapacityError,
            IdempotencyUnavailableError,
            IdempotencySchemaError,
            IdempotencyStorageError,
            JobCapacityError,
            JobSubmissionError,
            JobManagerShutdownError,
            PlatformUnavailableError,
            CatalogSchemaError,
            CatalogStorageError,
            FileStoreIOError,
            ProviderError,
        ),
    ):
        return 503, "service_unavailable", "The service is temporarily unavailable."
    if isinstance(error, PlatformIntegrityError):
        return 500, "internal_error", "The request could not be completed."
    return 500, "internal_error", "The request could not be completed."


def classify_http_error(status_code: int) -> tuple[str, str]:
    if status_code == 404:
        return "resource_unavailable", "Resource is unavailable."
    if status_code == 405:
        return "method_not_allowed", "The method is not allowed for this resource."
    if status_code == 413:
        return "upload_limit_exceeded", "The upload exceeds the configured limits."
    if status_code in {400, 422}:
        return "invalid_request", "The request could not be validated."
    if status_code == 401:
        return "authentication_failed", "Authentication failed."
    if status_code == 403:
        return "forbidden", "The operation is not permitted."
    return "request_failed", "The request could not be completed."


__all__ = [
    "APPLICATION_ERROR_TYPES",
    "ApiBoundaryError",
    "classify_application_error",
    "classify_http_error",
]
