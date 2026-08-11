"""Tenant-scoped document asset planning, persistence, and verification."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rag_system.application import PlatformIntegrityError, PlatformValidationError, UploadDocument
from rag_system.application_ports import DocumentStore
from rag_system.catalog import DocumentManifest, KnowledgeBaseRecord
from rag_system.tenancy import Principal


@dataclass(frozen=True, slots=True)
class PlannedDocument:
    resource_id: str
    upload: UploadDocument
    manifest: DocumentManifest


class KnowledgeBaseAssets:
    """Own the consistency boundary between catalog manifests and stored files."""

    def __init__(self, file_store: DocumentStore) -> None:
        self._file_store = file_store

    def plan(
        self,
        principal: Principal,
        uploads: Sequence[UploadDocument],
        *,
        new_document_id: Callable[[], str],
    ) -> tuple[PlannedDocument, ...]:
        if not callable(new_document_id):
            raise TypeError("new_document_id must be callable")
        planned: list[PlannedDocument] = []
        for upload in uploads:
            if not isinstance(upload.source, bytes):
                raise PlatformValidationError("uploads must be materialized before planning")
            document_id = new_document_id()
            relative_path = self._file_store.planned_relative_path(
                principal.tenant_id.value,
                document_id,
                upload.display_name,
            )
            content = upload.source
            manifest = DocumentManifest(
                display_name=PurePosixPath(relative_path).name,
                relative_path=relative_path,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            planned.append(PlannedDocument(document_id, upload, manifest))
        return tuple(planned)

    def store(self, principal: Principal, documents: Sequence[PlannedDocument]) -> None:
        for planned in documents:
            saved = self._file_store.save(
                principal.tenant_id.value,
                planned.resource_id,
                planned.upload.display_name,
                planned.upload.source,
            )
            manifest = planned.manifest
            if (
                saved.display_name != manifest.display_name
                or saved.relative_path != manifest.relative_path
                or saved.size != manifest.size_bytes
                or saved.sha256 != manifest.sha256
            ):
                raise PlatformIntegrityError("stored document does not match its manifest")

    def resolve(
        self,
        principal: Principal,
        record: KnowledgeBaseRecord,
    ) -> tuple[Path, ...]:
        resolved: list[Path] = []
        for document in record.documents:
            document_id = self.document_resource_id(principal, document)
            path = self._file_store.resolve(principal.tenant_id.value, document_id)
            relative = path.resolve(strict=True).relative_to(self._file_store.root).as_posix()
            if relative != document.relative_path or path.name != document.display_name:
                raise PlatformIntegrityError("document path does not match its manifest")
            size, digest = self._file_identity(path)
            if size != document.size_bytes or digest != document.sha256:
                raise PlatformIntegrityError("document content does not match its manifest")
            resolved.append(path)
        if not resolved:
            raise PlatformIntegrityError("knowledge base has no documents")
        return tuple(resolved)

    def delete(self, principal: Principal, record: KnowledgeBaseRecord) -> None:
        for document in record.documents:
            document_id = self.document_resource_id(principal, document)
            self._file_store.delete(principal.tenant_id.value, document_id)

    @staticmethod
    def document_resource_id(
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


__all__ = ["KnowledgeBaseAssets", "PlannedDocument"]
