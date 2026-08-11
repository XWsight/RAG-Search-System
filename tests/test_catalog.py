from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_system.catalog import (
    CatalogSchemaError,
    CatalogStorageError,
    CatalogValidationError,
    DocumentManifest,
    InvalidStatusTransitionError,
    KnowledgeBaseCatalog,
    KnowledgeBaseErrorCode,
    KnowledgeBaseStatus,
    KnowledgeBaseUnavailableError,
)
from rag_system.tenancy import Principal, TenantId


def make_principal(tenant: str) -> Principal:
    return Principal(f"user-{tenant}", TenantId(tenant), frozenset({"reader"}))


def make_document(name: str = "guide.txt", content: bytes = b"RAG knowledge") -> DocumentManifest:
    return DocumentManifest(
        display_name=name,
        relative_path=f"documents/{name}",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name, "catalog.sqlite3")
        self.catalog = KnowledgeBaseCatalog(self.database)
        self.tenant_a = make_principal("tenant-a")
        self.tenant_b = make_principal("tenant-b")

    def test_restart_recovers_record_and_manifest(self) -> None:
        document = make_document()
        created = self.catalog.create(self.tenant_a, "技术资料", [document])
        reopened = KnowledgeBaseCatalog(self.database)
        restored = reopened.get(self.tenant_a, created.resource_id)

        self.assertEqual(restored, created)
        self.assertEqual(restored.documents, (document,))
        self.assertEqual(restored.document_count, 1)
        self.assertEqual(restored.total_bytes, document.size_bytes)
        self.assertNotIn("tenant-a", restored.resource_id)
        self.assertNotIn("技术资料", restored.resource_id)

    def test_valid_state_machine_and_delete_returns_manifest(self) -> None:
        document = make_document()
        record = self.catalog.create(self.tenant_a, "资料", [document])
        record = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.INDEXING,
            internal_index_id="index-v1",
        )
        self.assertEqual(record.status, KnowledgeBaseStatus.INDEXING)
        record = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.READY,
            chunk_count=12,
        )
        self.assertEqual(record.chunk_count, 12)

        record = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.INDEXING,
            internal_index_id="index-v2",
        )
        record = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.FAILED,
            error_code=KnowledgeBaseErrorCode.INDEX_BUILD_FAILED,
        )
        self.assertEqual(record.error_code, KnowledgeBaseErrorCode.INDEX_BUILD_FAILED)
        record = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.DELETING,
        )
        self.assertIsNone(record.error_code)
        self.assertEqual(self.catalog.delete(self.tenant_a, record.resource_id), (document,))
        with self.assertRaises(KnowledgeBaseUnavailableError):
            self.catalog.get(self.tenant_a, record.resource_id)

    def test_illegal_transitions_are_rejected_without_mutation(self) -> None:
        record = self.catalog.create(self.tenant_a, "资料", [make_document()])
        with self.assertRaises(InvalidStatusTransitionError):
            self.catalog.transition(
                self.tenant_a,
                record.resource_id,
                KnowledgeBaseStatus.READY,
                chunk_count=1,
            )
        self.assertEqual(
            self.catalog.get(self.tenant_a, record.resource_id).status,
            KnowledgeBaseStatus.PENDING,
        )

        with self.assertRaises(InvalidStatusTransitionError):
            self.catalog.delete(self.tenant_a, record.resource_id)

    def test_pending_ingestion_can_fail_with_a_safe_code(self) -> None:
        record = self.catalog.create(self.tenant_a, "invalid upload", [make_document()])
        failed = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.FAILED,
            error_code=KnowledgeBaseErrorCode.CONTENT_REJECTED,
        )
        self.assertEqual(failed.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(failed.error_code, KnowledgeBaseErrorCode.CONTENT_REJECTED)

    def test_cancellation_intent_is_durable_and_only_converges_to_failure(self) -> None:
        record = self.catalog.create(self.tenant_a, "cancelled upload", [make_document()])
        cancelling = self.catalog.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.CANCELLING,
        )
        self.assertEqual(cancelling.status, KnowledgeBaseStatus.CANCELLING)
        self.assertIsNone(cancelling.error_code)
        reopened = KnowledgeBaseCatalog(self.database)
        self.assertEqual(
            reopened.get(self.tenant_a, record.resource_id).status,
            KnowledgeBaseStatus.CANCELLING,
        )
        with self.assertRaises(InvalidStatusTransitionError):
            reopened.transition(
                self.tenant_a,
                record.resource_id,
                KnowledgeBaseStatus.READY,
                chunk_count=1,
            )
        failed = reopened.transition(
            self.tenant_a,
            record.resource_id,
            KnowledgeBaseStatus.FAILED,
            error_code=KnowledgeBaseErrorCode.INDEX_CANCELLED,
        )
        self.assertEqual(failed.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(failed.error_code, KnowledgeBaseErrorCode.INDEX_CANCELLED)

    def test_cross_tenant_and_missing_have_identical_errors(self) -> None:
        own = self.catalog.create(self.tenant_a, "私有资料", [make_document()])
        messages: list[str] = []
        for resource_id in (own.resource_id, "kb_" + "x" * 32, "invalid"):
            with self.assertRaises(KnowledgeBaseUnavailableError) as raised:
                self.catalog.get(self.tenant_b, resource_id)
            messages.append(str(raised.exception))
            self.assertNotIn(resource_id, str(raised.exception))
        self.assertEqual(len(set(messages)), 1)
        self.assertEqual(self.catalog.list(self.tenant_b), ())
        self.assertEqual(len(self.catalog.list(self.tenant_a)), 1)

    def test_idempotency_reservation_lookup_is_durable_and_tenant_scoped(self) -> None:
        reservation_id = "idem_0123456789abcdef0123456789abcdef"
        record = self.catalog.create(
            self.tenant_a,
            "可恢复资料",
            idempotency_reservation_id=reservation_id,
        )
        reopened = KnowledgeBaseCatalog(self.database)
        self.assertEqual(
            reopened.find_by_idempotency_reservation(
                self.tenant_a,
                reservation_id,
            ),
            record,
        )
        self.assertIsNone(
            reopened.find_by_idempotency_reservation(
                self.tenant_b,
                reservation_id,
            )
        )

    def test_list_is_bounded_and_tenant_filtered(self) -> None:
        for index in range(12):
            principal = self.tenant_a if index % 2 == 0 else self.tenant_b
            self.catalog.create(principal, f"资料 {index}")
        self.assertEqual(len(self.catalog.list(self.tenant_a, limit=3)), 3)
        self.assertEqual(len(self.catalog.list(self.tenant_a, limit=100)), 6)
        for bad_limit in (0, 101, True):
            with self.subTest(limit=bad_limit), self.assertRaises(CatalogValidationError):
                self.catalog.list(self.tenant_a, limit=bad_limit)

    def test_manifest_paths_and_json_are_strict(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        for path in ("/absolute.txt", "../escape.txt", "a/../b.txt", "a//b.txt", "C:/file.txt"):
            with self.subTest(path=path), self.assertRaises(CatalogValidationError):
                DocumentManifest("file.txt", path, 1, digest)

        with self.assertRaises(CatalogValidationError):
            DocumentManifest("file.txt", "documents/file.txt", True, digest)
        with self.assertRaises(CatalogValidationError):
            DocumentManifest("file.txt", "documents/file.txt", 1, digest.upper())

        record = self.catalog.create(self.tenant_a, "资料", [make_document()])
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "UPDATE knowledge_bases SET manifest_json = ? WHERE resource_id = ?",
                ('[{"display_name":"x","display_name":"y"}]', record.resource_id),
            )
            connection.commit()
        with self.assertRaises(CatalogSchemaError):
            self.catalog.get(self.tenant_a, record.resource_id)

    def test_schema_version_wal_and_foreign_keys(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
        connection = self.catalog._connect()
        try:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        finally:
            connection.close()

        incompatible = Path(self.directory.name, "future.sqlite3")
        with closing(sqlite3.connect(incompatible)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaises(CatalogSchemaError):
            KnowledgeBaseCatalog(incompatible)

    def test_schema_v2_is_migrated_without_losing_records(self) -> None:
        record = self.catalog.create(self.tenant_a, "migration record", [make_document()])
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("PRAGMA user_version = 2")
            connection.commit()

        migrated = KnowledgeBaseCatalog(self.database)
        self.assertEqual(migrated.get(self.tenant_a, record.resource_id), record)
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)

    def test_connection_is_closed_when_pragma_initialization_fails(self) -> None:
        connection = MagicMock()
        connection.execute.side_effect = sqlite3.OperationalError("disk unavailable")
        with patch("rag_system.catalog.sqlite3.connect", return_value=connection):
            with self.assertRaises(CatalogStorageError):
                self.catalog._connect()
        connection.close.assert_called_once_with()

    def test_concurrent_creates_are_unique_and_recoverable(self) -> None:
        def create(index: int) -> tuple[str, str]:
            principal = self.tenant_a if index % 2 == 0 else self.tenant_b
            record = self.catalog.create(principal, f"并发资料 {index}")
            return principal.tenant_id.value, record.resource_id

        with ThreadPoolExecutor(max_workers=10) as pool:
            created = list(pool.map(create, range(60)))

        resource_ids = [resource_id for _, resource_id in created]
        self.assertEqual(len(resource_ids), len(set(resource_ids)))
        self.assertEqual(len(self.catalog.list(self.tenant_a, limit=100)), 30)
        self.assertEqual(len(self.catalog.list(self.tenant_b, limit=100)), 30)
        self.assertEqual(len(KnowledgeBaseCatalog(self.database).list(self.tenant_a, limit=100)), 30)


if __name__ == "__main__":
    unittest.main()
