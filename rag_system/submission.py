"""Validation and canonicalization for knowledge-base submissions."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Sequence

from rag_system.application import (
    PlatformUnavailableError,
    PlatformValidationError,
    UploadDocument,
)


class UploadBatchPreparer:
    """Materialize bounded uploads and derive stable submission identities."""

    _READ_SIZE = 64 * 1024

    def __init__(
        self,
        *,
        max_file_bytes: int,
        max_total_bytes: int,
        max_documents: int,
        document_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        if not isinstance(max_total_bytes, int) or max_total_bytes < max_file_bytes:
            raise ValueError("max_total_bytes must be at least max_file_bytes")
        if not isinstance(max_documents, int) or max_documents <= 0:
            raise ValueError("max_documents must be a positive integer")
        if not callable(document_id_factory):
            raise TypeError("document_id_factory must be callable")
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._max_documents = max_documents
        self._document_id_factory = document_id_factory

    def prepare(self, documents: Sequence[UploadDocument]) -> tuple[UploadDocument, ...]:
        uploads = tuple(documents)
        if not uploads:
            raise PlatformValidationError("at least one document is required")
        if len(uploads) > self._max_documents:
            raise PlatformValidationError("document count exceeds the configured limit")

        materialized: list[UploadDocument] = []
        total = 0
        for upload in uploads:
            if not isinstance(upload, UploadDocument):
                raise PlatformValidationError("documents must be UploadDocument values")
            content = self._read(upload.source)
            total += len(content)
            if total > self._max_total_bytes:
                raise PlatformValidationError("total upload size exceeds the configured limit")
            materialized.append(UploadDocument(upload.display_name, content))
        return tuple(materialized)

    def new_document_id(self) -> str:
        value = self._document_id_factory()
        if not isinstance(value, str):
            raise PlatformUnavailableError("document identifier generation failed")
        normalized = value.replace("-", "")
        if len(normalized) < 16 or not normalized.isalnum():
            raise PlatformUnavailableError("document identifier generation failed")
        return f"doc_{normalized[:48]}"

    @staticmethod
    def request_digest(
        display_name: str,
        uploads: Sequence[UploadDocument],
    ) -> str:
        if not isinstance(display_name, str):
            raise PlatformValidationError("display_name must be a string")
        try:
            collection_name = display_name.encode("utf-8")
            identities: list[tuple[bytes, bytes]] = []
            for upload in uploads:
                if not isinstance(upload.source, (bytes, bytearray, memoryview)):
                    raise TypeError
                identities.append(
                    (
                        upload.display_name.encode("utf-8"),
                        hashlib.sha256(bytes(upload.source)).digest(),
                    )
                )
            identities.sort()
        except (AttributeError, TypeError, UnicodeError):
            raise PlatformValidationError("upload metadata is invalid") from None
        digest = hashlib.sha256(b"rag-create-request-v1")
        values = (collection_name, *(part for identity in identities for part in identity))
        for value in values:
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        return digest.hexdigest()

    def _read(self, source: object) -> bytes:
        if isinstance(source, (bytes, bytearray, memoryview)):
            content = bytes(source)
            if len(content) > self._max_file_bytes:
                raise PlatformValidationError("file size exceeds the configured limit")
            return content
        reader = getattr(source, "read", None)
        if not callable(reader):
            raise PlatformValidationError("upload source must be binary")
        chunks: list[bytes] = []
        size = 0
        try:
            while True:
                block = reader(min(self._READ_SIZE, self._max_file_bytes - size + 1))
                if block in (b"", None):
                    break
                if not isinstance(block, (bytes, bytearray, memoryview)):
                    raise PlatformValidationError("upload source must return bytes")
                normalized = bytes(block)
                size += len(normalized)
                if size > self._max_file_bytes:
                    raise PlatformValidationError("file size exceeds the configured limit")
                chunks.append(normalized)
        except PlatformValidationError:
            raise
        except Exception:
            raise PlatformValidationError("upload could not be read") from None
        return b"".join(chunks)


__all__ = ["UploadBatchPreparer"]
