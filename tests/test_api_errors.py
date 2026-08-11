from __future__ import annotations

import unittest

from rag_system.api_errors import (
    APPLICATION_ERROR_TYPES,
    ApiBoundaryError,
    classify_application_error,
    classify_http_error,
)
from rag_system.application import (
    IdempotencyInProgressError,
    KnowledgeBaseNotReadyError,
    PlatformIntegrityError,
    PlatformUnavailableError,
    PlatformValidationError,
)
from rag_system.catalog import KnowledgeBaseUnavailableError
from rag_system.file_store import DuplicateResourceError, StorageLimitError
from rag_system.idempotency import IdempotencyConflictError
from rag_system.tenancy import AuthenticationError, AuthorizationError


class ApiErrorContractTests(unittest.TestCase):
    def test_application_failures_have_stable_safe_classifications(self) -> None:
        cases = (
            (AuthenticationError(), (401, "authentication_failed")),
            (AuthorizationError(), (404, "resource_unavailable")),
            (KnowledgeBaseUnavailableError(), (404, "resource_unavailable")),
            (IdempotencyConflictError(), (409, "idempotency_conflict")),
            (IdempotencyInProgressError("busy"), (409, "idempotency_in_progress")),
            (DuplicateResourceError("exists"), (409, "resource_conflict")),
            (KnowledgeBaseNotReadyError("pending"), (409, "knowledge_base_not_ready")),
            (StorageLimitError("large"), (413, "storage_limit_exceeded")),
            (PlatformValidationError("bad"), (422, "invalid_request")),
            (PlatformUnavailableError("down"), (503, "service_unavailable")),
            (PlatformIntegrityError("corrupt"), (500, "internal_error")),
            (RuntimeError("unknown"), (500, "internal_error")),
        )

        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                status, code, message = classify_application_error(error)
                self.assertEqual((status, code), expected)
                self.assertLessEqual(len(message), 256)
        self.assertNotIn(
            "sensitive-upstream-detail",
            classify_application_error(RuntimeError("sensitive-upstream-detail"))[2],
        )

    def test_registered_error_families_are_unique_exception_types(self) -> None:
        self.assertEqual(len(APPLICATION_ERROR_TYPES), len(set(APPLICATION_ERROR_TYPES)))
        self.assertTrue(all(issubclass(item, Exception) for item in APPLICATION_ERROR_TYPES))

    def test_http_framework_errors_use_a_closed_public_vocabulary(self) -> None:
        expected_codes = {
            400: "invalid_request",
            401: "authentication_failed",
            403: "forbidden",
            404: "resource_unavailable",
            405: "method_not_allowed",
            413: "upload_limit_exceeded",
            422: "invalid_request",
            418: "request_failed",
        }
        for status, expected_code in expected_codes.items():
            code, message = classify_http_error(status)
            self.assertEqual(code, expected_code)
            self.assertTrue(message)

    def test_boundary_error_copies_headers(self) -> None:
        headers = {"Retry-After": "1"}
        error = ApiBoundaryError(429, "rate_limit_exceeded", "Slow down.", headers=headers)
        headers["Retry-After"] = "999"

        self.assertEqual(error.headers, {"Retry-After": "1"})
        self.assertEqual(error.status_code, 429)
        self.assertEqual(str(error), "rate_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
