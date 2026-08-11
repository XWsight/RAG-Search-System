"""Durable, tenant-isolated SQLite catalog for knowledge bases."""

from __future__ import annotations

import json
import math
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, cast

from .tenancy import Principal, TenantId


_SCHEMA_VERSION = 4
_MAX_LIST_LIMIT = 100
_MAX_LIST_OFFSET = 10_000
_MAX_MANIFEST_ITEMS = 10_000
_MAX_MANIFEST_JSON_BYTES = 8 * 1024 * 1024
_RESOURCE_ID_PATTERN = re.compile(r"kb_[A-Za-z0-9_-]{32}")
_INDEX_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,190}[A-Za-z0-9])?")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDEMPOTENCY_RESERVATION_PATTERN = re.compile(r"idem_[0-9a-f]{32}")


class CatalogError(Exception):
    """Base class for catalog failures safe to classify at an API boundary."""


class CatalogValidationError(CatalogError, ValueError):
    """Catalog input failed strict validation."""


class CatalogSchemaError(CatalogError):
    """The on-disk schema or stored data violates the catalog contract."""

    def __init__(self) -> None:
        super().__init__("Catalog schema or stored data is invalid.")


class CatalogStorageError(CatalogError):
    """A sanitized SQLite operation failure."""

    def __init__(self) -> None:
        super().__init__("Catalog storage operation failed.")


class KnowledgeBaseUnavailableError(CatalogError):
    """The same denial for missing and cross-tenant resources."""

    def __init__(self) -> None:
        super().__init__("Knowledge base is unavailable.")


class InvalidStatusTransitionError(CatalogError):
    def __init__(self, current: KnowledgeBaseStatus, target: KnowledgeBaseStatus) -> None:
        super().__init__(f"Invalid knowledge base status transition: {current.value} -> {target.value}.")


class KnowledgeBaseStatus(StrEnum):
    PREPARING = "preparing"
    PENDING = "pending"
    INDEXING = "indexing"
    CANCELLING = "cancelling"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class KnowledgeBaseErrorCode(StrEnum):
    CONTENT_REJECTED = "content_rejected"
    INGESTION_FAILED = "ingestion_failed"
    INDEX_BUILD_FAILED = "index_build_failed"
    INDEX_STORAGE_FAILED = "index_storage_failed"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    INTERNAL_ERROR = "internal_error"
    INDEX_CANCELLED = "index_cancelled"


_ALLOWED_TRANSITIONS: Mapping[KnowledgeBaseStatus, frozenset[KnowledgeBaseStatus]] = {
    KnowledgeBaseStatus.PREPARING: frozenset(
        {
            KnowledgeBaseStatus.PENDING,
            KnowledgeBaseStatus.FAILED,
            KnowledgeBaseStatus.DELETING,
        }
    ),
    KnowledgeBaseStatus.PENDING: frozenset(
        {
            KnowledgeBaseStatus.INDEXING,
            KnowledgeBaseStatus.CANCELLING,
            KnowledgeBaseStatus.FAILED,
            KnowledgeBaseStatus.DELETING,
        }
    ),
    KnowledgeBaseStatus.INDEXING: frozenset(
        {
            KnowledgeBaseStatus.READY,
            KnowledgeBaseStatus.CANCELLING,
            KnowledgeBaseStatus.FAILED,
            KnowledgeBaseStatus.DELETING,
        }
    ),
    KnowledgeBaseStatus.READY: frozenset(
        {KnowledgeBaseStatus.INDEXING, KnowledgeBaseStatus.DELETING}
    ),
    KnowledgeBaseStatus.CANCELLING: frozenset(
        {KnowledgeBaseStatus.FAILED, KnowledgeBaseStatus.DELETING}
    ),
    KnowledgeBaseStatus.FAILED: frozenset(
        {KnowledgeBaseStatus.INDEXING, KnowledgeBaseStatus.DELETING}
    ),
    KnowledgeBaseStatus.DELETING: frozenset(),
}


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    display_name: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_name", _validate_display_name(self.display_name, 255))
        object.__setattr__(self, "relative_path", _validate_relative_path(self.relative_path))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise CatalogValidationError("Document size must be an integer.")
        if not 0 <= self.size_bytes <= 2**63 - 1:
            raise CatalogValidationError("Document size is outside the supported range.")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise CatalogValidationError("Document SHA-256 must be 64 lowercase hexadecimal characters.")


DocumentManifestEntry = DocumentManifest


@dataclass(frozen=True, slots=True)
class KnowledgeBaseRecord:
    resource_id: str
    tenant_id: TenantId
    display_name: str
    status: KnowledgeBaseStatus
    internal_index_id: str | None
    documents: tuple[DocumentManifest, ...]
    document_count: int
    total_bytes: int
    chunk_count: int
    error_code: KnowledgeBaseErrorCode | None
    created_at: float
    updated_at: float
    version: int
    idempotency_reservation_id: str | None = None

    def __post_init__(self) -> None:
        _validate_resource_id(self.resource_id)
        if not isinstance(self.tenant_id, TenantId):
            raise CatalogValidationError("tenant_id must be a TenantId.")
        object.__setattr__(self, "display_name", _validate_display_name(self.display_name, 200))
        if not isinstance(self.status, KnowledgeBaseStatus):
            raise CatalogValidationError("Invalid knowledge base status.")
        if self.internal_index_id is not None:
            object.__setattr__(
                self,
                "internal_index_id",
                _validate_internal_index_id(self.internal_index_id),
            )
        object.__setattr__(self, "documents", _normalize_manifest(self.documents))
        expected_total = sum(item.size_bytes for item in self.documents)
        if self.document_count != len(self.documents) or self.total_bytes != expected_total:
            raise CatalogValidationError("Stored document counts do not match the manifest.")
        if isinstance(self.chunk_count, bool) or not isinstance(self.chunk_count, int) or self.chunk_count < 0:
            raise CatalogValidationError("chunk_count must be a non-negative integer.")
        if self.status in {
            KnowledgeBaseStatus.PENDING,
            KnowledgeBaseStatus.INDEXING,
            KnowledgeBaseStatus.CANCELLING,
            KnowledgeBaseStatus.READY,
        } and not self.documents:
            raise CatalogValidationError(
                "An active knowledge base requires an attached document manifest."
            )
        if self.status in {
            KnowledgeBaseStatus.PREPARING,
            KnowledgeBaseStatus.PENDING,
        } and (self.internal_index_id is not None or self.chunk_count != 0):
            raise CatalogValidationError(
                "Preparing and pending knowledge bases cannot contain index results."
            )
        if (
            self.status is KnowledgeBaseStatus.INDEXING
            and self.internal_index_id is None
        ):
            raise CatalogValidationError(
                "An indexing knowledge base requires an internal index ID."
            )
        if self.status is KnowledgeBaseStatus.READY and self.internal_index_id is None:
            raise CatalogValidationError("A ready knowledge base requires an internal index ID.")
        if self.status is KnowledgeBaseStatus.FAILED:
            if not isinstance(self.error_code, KnowledgeBaseErrorCode):
                raise CatalogValidationError("A failed knowledge base requires a safe error code.")
        elif self.error_code is not None:
            raise CatalogValidationError("Only failed knowledge bases may contain an error code.")
        if not _valid_timestamp(self.created_at) or not _valid_timestamp(self.updated_at):
            raise CatalogValidationError("Catalog timestamps must be finite and non-negative.")
        if self.updated_at < self.created_at:
            raise CatalogValidationError("updated_at cannot precede created_at.")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise CatalogValidationError("version must be a positive integer.")
        if self.idempotency_reservation_id is not None:
            object.__setattr__(
                self,
                "idempotency_reservation_id",
                _validate_idempotency_reservation_id(
                    self.idempotency_reservation_id
                ),
            )


class KnowledgeBaseCatalog:
    """A thread-safe catalog using short SQLite transactions per operation.

    WAL provides process-safe concurrent readers while ``BEGIN IMMEDIATE``
    serializes state transitions.  The object keeps no database connection or
    plaintext document content in memory between calls.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 60:
            raise CatalogValidationError("timeout_seconds must be between 0 and 60.")
        path = Path(database_path)
        if path.exists() and not path.is_file():
            raise CatalogValidationError("database_path must reference a file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = path.resolve()
        self._clock = clock
        self._timeout_seconds = float(timeout_seconds)
        self._write_lock = RLock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def create(
        self,
        principal: Principal,
        display_name: str,
        *,
        idempotency_reservation_id: str | None = None,
    ) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_name = _validate_display_name(display_name, 200)
        manifest_json = _encode_manifest(())
        created_at = self._now()
        reservation_id = (
            _validate_idempotency_reservation_id(idempotency_reservation_id)
            if idempotency_reservation_id is not None
            else None
        )

        with self._write_lock, self._write_transaction() as connection:
            for _ in range(8):
                resource_id = _new_resource_id()
                try:
                    connection.execute(
                        """
                        INSERT INTO knowledge_bases (
                            resource_id, tenant_id, display_name, status,
                            internal_index_id, manifest_json, document_count,
                            total_bytes, chunk_count, error_code, created_at,
                            updated_at, version, idempotency_reservation_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            resource_id,
                            tenant_id.value,
                            clean_name,
                            KnowledgeBaseStatus.PREPARING.value,
                            None,
                            manifest_json,
                            0,
                            0,
                            0,
                            None,
                            created_at,
                            created_at,
                            1,
                            reservation_id,
                        ),
                    )
                except sqlite3.IntegrityError:
                    continue
                row = self._select_row(connection, tenant_id, resource_id)
                if row is None:
                    raise CatalogStorageError()
                return _record_from_row(row)
        raise CatalogStorageError()

    def get(self, principal: Principal, resource_id: str) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        with self._read_connection() as connection:
            row = self._select_row(connection, tenant_id, clean_id)
        if row is None:
            raise KnowledgeBaseUnavailableError()
        return _record_from_row(row)

    def find_by_idempotency_reservation(
        self,
        principal: Principal,
        reservation_id: str,
    ) -> KnowledgeBaseRecord | None:
        """Find the tenant-owned resource for crash recovery, if it exists."""

        tenant_id = _principal_tenant(principal)
        clean_reservation_id = _validate_idempotency_reservation_id(reservation_id)
        with self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE tenant_id = ? AND idempotency_reservation_id = ?
                """,
                (tenant_id.value, clean_reservation_id),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list(
        self,
        principal: Principal,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[KnowledgeBaseRecord, ...]:
        tenant_id = _principal_tenant(principal)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise CatalogValidationError(f"limit must be between 1 and {_MAX_LIST_LIMIT}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= _MAX_LIST_OFFSET:
            raise CatalogValidationError(f"offset must be between 0 and {_MAX_LIST_OFFSET}.")
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE tenant_id = ?
                ORDER BY updated_at DESC, resource_id DESC
                LIMIT ? OFFSET ?
                """,
                (tenant_id.value, limit, offset),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def attach_manifest(
        self,
        principal: Principal,
        resource_id: str,
        documents: Sequence[DocumentManifest],
    ) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        manifest = _normalize_manifest(documents)
        if not manifest:
            raise CatalogValidationError("Document manifest cannot be empty.")
        manifest_json = _encode_manifest(manifest)
        with self._write_lock, self._write_transaction() as connection:
            current = self._owned_record(connection, tenant_id, clean_id)
            if current.status is not KnowledgeBaseStatus.PREPARING:
                raise InvalidStatusTransitionError(
                    current.status,
                    KnowledgeBaseStatus.PREPARING,
                )
            if current.documents:
                if current.documents == manifest:
                    return current
                raise CatalogValidationError(
                    "Document manifest is immutable once attached."
                )
            connection.execute(
                """
                UPDATE knowledge_bases
                SET manifest_json = ?, document_count = ?, total_bytes = ?,
                    updated_at = ?, version = version + 1
                WHERE resource_id = ? AND tenant_id = ?
                """,
                (
                    manifest_json,
                    len(manifest),
                    sum(item.size_bytes for item in manifest),
                    self._now(),
                    clean_id,
                    tenant_id.value,
                ),
            )
            return self._owned_record(connection, tenant_id, clean_id)

    def transition(
        self,
        principal: Principal,
        resource_id: str,
        target: KnowledgeBaseStatus,
        *,
        internal_index_id: str | None = None,
        chunk_count: int | None = None,
        error_code: KnowledgeBaseErrorCode | None = None,
    ) -> KnowledgeBaseRecord:
        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        if not isinstance(target, KnowledgeBaseStatus):
            raise CatalogValidationError("target must be a KnowledgeBaseStatus.")

        with self._write_lock, self._write_transaction() as connection:
            current = self._owned_record(connection, tenant_id, clean_id)
            if target not in _ALLOWED_TRANSITIONS[current.status]:
                raise InvalidStatusTransitionError(current.status, target)

            new_index_id = current.internal_index_id
            new_chunk_count = current.chunk_count
            new_error_code: KnowledgeBaseErrorCode | None = None

            if target is KnowledgeBaseStatus.PENDING:
                if internal_index_id is not None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Pending rejects index, chunk, and error overrides."
                    )
                if not current.documents:
                    raise CatalogValidationError(
                        "Pending requires an attached document manifest."
                    )
                new_index_id = None
                new_chunk_count = 0
            elif target is KnowledgeBaseStatus.INDEXING:
                if internal_index_id is None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Indexing requires internal_index_id and rejects chunk_count/error_code."
                    )
                new_index_id = _validate_internal_index_id(internal_index_id)
                new_chunk_count = 0
            elif target is KnowledgeBaseStatus.READY:
                if internal_index_id is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Ready uses the existing internal index and rejects error_code."
                    )
                if current.internal_index_id is None:
                    raise CatalogValidationError("Ready requires an existing internal index ID.")
                if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 0:
                    raise CatalogValidationError("Ready requires a non-negative chunk_count.")
                new_chunk_count = chunk_count
            elif target is KnowledgeBaseStatus.FAILED:
                if internal_index_id is not None or chunk_count is not None:
                    raise CatalogValidationError("Failed rejects index and chunk overrides.")
                if not isinstance(error_code, KnowledgeBaseErrorCode):
                    raise CatalogValidationError("Failed requires a safe KnowledgeBaseErrorCode.")
                new_error_code = error_code
            elif target is KnowledgeBaseStatus.CANCELLING:
                if internal_index_id is not None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError(
                        "Cancelling rejects index, chunk, and error overrides."
                    )
            elif target is KnowledgeBaseStatus.DELETING:
                if internal_index_id is not None or chunk_count is not None or error_code is not None:
                    raise CatalogValidationError("Deleting rejects index, chunk, and error overrides.")

            connection.execute(
                """
                UPDATE knowledge_bases
                SET status = ?, internal_index_id = ?, chunk_count = ?,
                    error_code = ?, updated_at = ?, version = version + 1
                WHERE resource_id = ? AND tenant_id = ?
                """,
                (
                    target.value,
                    new_index_id,
                    new_chunk_count,
                    new_error_code.value if new_error_code else None,
                    self._now(),
                    clean_id,
                    tenant_id.value,
                ),
            )
            return self._owned_record(connection, tenant_id, clean_id)

    def delete(
        self,
        principal: Principal,
        resource_id: str,
    ) -> tuple[DocumentManifest, ...]:
        """Delete a DELETING record and return its immutable cleanup manifest."""

        tenant_id = _principal_tenant(principal)
        clean_id = _safe_resource_lookup_id(resource_id)
        with self._write_lock, self._write_transaction() as connection:
            current = self._owned_record(connection, tenant_id, clean_id)
            if current.status is not KnowledgeBaseStatus.DELETING:
                raise InvalidStatusTransitionError(current.status, KnowledgeBaseStatus.DELETING)
            cursor = connection.execute(
                "DELETE FROM knowledge_bases WHERE resource_id = ? AND tenant_id = ?",
                (clean_id, tenant_id.value),
            )
            if cursor.rowcount != 1:
                raise KnowledgeBaseUnavailableError()
            return current.documents

    def _initialize(self) -> None:
        with self._write_lock, self._write_transaction(validate_schema=False) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                existing = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'knowledge_bases'"
                ).fetchone()
                if existing is not None:
                    raise CatalogSchemaError()
                connection.execute(_CREATE_TABLE_SQL)
                self._create_indexes(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version in {2, 3}:
                self._migrate_to_v4(connection, source_version=version)
            elif version != _SCHEMA_VERSION:
                raise CatalogSchemaError()
            self._validate_schema(connection)

    def _migrate_to_v4(
        self,
        connection: sqlite3.Connection,
        *,
        source_version: int,
    ) -> None:
        """Rebuild a supported legacy table with the explicit preparing state."""

        self._validate_schema(connection)
        names = tuple(name for name, *_ in _EXPECTED_COLUMNS)
        column_names = ", ".join(names)
        selected_columns = ", ".join(
            (
                "CASE WHEN status = 'pending' AND manifest_json = '[]' "
                "THEN 'preparing' ELSE status END AS status"
                if name == "status"
                else name
            )
            for name in names
        )
        legacy_table = f"knowledge_bases_v{source_version}"
        connection.execute(
            f"ALTER TABLE knowledge_bases RENAME TO {legacy_table}"
        )
        connection.execute(_CREATE_TABLE_SQL)
        connection.execute(
            f"INSERT INTO knowledge_bases ({column_names}) "
            f"SELECT {selected_columns} FROM {legacy_table}"
        )
        connection.execute(f"DROP TABLE {legacy_table}")
        self._create_indexes(connection)
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _create_indexes(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE INDEX idx_knowledge_bases_tenant_updated "
            "ON knowledge_bases (tenant_id, updated_at DESC, resource_id DESC)"
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_knowledge_bases_idempotency "
            "ON knowledge_bases (tenant_id, idempotency_reservation_id) "
            "WHERE idempotency_reservation_id IS NOT NULL"
        )

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(knowledge_bases)").fetchall()
        actual = tuple(
            (row["name"], row["type"].upper(), int(row["notnull"]), int(row["pk"]))
            for row in rows
        )
        if actual != _EXPECTED_COLUMNS:
            raise CatalogSchemaError()

    def _owned_record(
        self,
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        resource_id: str,
    ) -> KnowledgeBaseRecord:
        row = self._select_row(connection, tenant_id, resource_id)
        if row is None:
            raise KnowledgeBaseUnavailableError()
        return _record_from_row(row)

    @staticmethod
    def _select_row(
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        resource_id: str,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                "SELECT * FROM knowledge_bases WHERE resource_id = ? AND tenant_id = ?",
                (resource_id, tenant_id.value),
            ).fetchone(),
        )

    def _now(self) -> float:
        value = float(self._clock())
        if not _valid_timestamp(value):
            raise CatalogValidationError("clock returned an invalid timestamp.")
        return value

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=self._timeout_seconds,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}")
            mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            if mode != "wal":
                raise CatalogStorageError()
            return connection
        except CatalogStorageError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise CatalogStorageError() from exc

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        except sqlite3.Error as exc:
            raise CatalogStorageError() from exc
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(
        self,
        *,
        validate_schema: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if validate_schema:
                self._validate_schema(connection)
            yield connection
            connection.commit()
        except CatalogError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise CatalogStorageError() from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


_CREATE_TABLE_SQL = """
CREATE TABLE knowledge_bases (
    resource_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'preparing', 'pending', 'indexing', 'cancelling',
            'ready', 'failed', 'deleting'
        )
    ),
    internal_index_id TEXT,
    manifest_json TEXT NOT NULL,
    document_count INTEGER NOT NULL CHECK (document_count >= 0),
    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
    error_code TEXT CHECK (
        error_code IS NULL OR error_code IN (
            'content_rejected', 'ingestion_failed', 'index_build_failed',
            'index_storage_failed', 'upstream_unavailable', 'internal_error',
            'index_cancelled'
        )
    ),
    created_at REAL NOT NULL CHECK (created_at >= 0),
    updated_at REAL NOT NULL CHECK (updated_at >= created_at),
    version INTEGER NOT NULL CHECK (version >= 1),
    idempotency_reservation_id TEXT
)
"""


_EXPECTED_COLUMNS = (
    ("resource_id", "TEXT", 1, 1),
    ("tenant_id", "TEXT", 1, 0),
    ("display_name", "TEXT", 1, 0),
    ("status", "TEXT", 1, 0),
    ("internal_index_id", "TEXT", 0, 0),
    ("manifest_json", "TEXT", 1, 0),
    ("document_count", "INTEGER", 1, 0),
    ("total_bytes", "INTEGER", 1, 0),
    ("chunk_count", "INTEGER", 1, 0),
    ("error_code", "TEXT", 0, 0),
    ("created_at", "REAL", 1, 0),
    ("updated_at", "REAL", 1, 0),
    ("version", "INTEGER", 1, 0),
    ("idempotency_reservation_id", "TEXT", 0, 0),
)


def _principal_tenant(principal: Principal) -> TenantId:
    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    return principal.tenant_id


def _new_resource_id() -> str:
    value = f"kb_{secrets.token_hex(16)}"
    _validate_resource_id(value)
    return value


def _validate_resource_id(value: object) -> str:
    if not isinstance(value, str) or _RESOURCE_ID_PATTERN.fullmatch(value) is None:
        raise CatalogValidationError("Invalid knowledge base resource ID.")
    return value


def _safe_resource_lookup_id(value: object) -> str:
    try:
        return _validate_resource_id(value)
    except CatalogValidationError:
        raise KnowledgeBaseUnavailableError() from None


def _validate_display_name(value: object, max_length: int) -> str:
    if not isinstance(value, str):
        raise CatalogValidationError("Display name must be text.")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise CatalogValidationError("Display name has an invalid length.")
    if any(ord(character) < 32 for character in normalized) or any(
        character in '/\\<>:"|?*' for character in normalized
    ):
        raise CatalogValidationError("Display name contains unsafe characters.")
    if normalized.endswith((".", " ")):
        raise CatalogValidationError("Display name has an unsafe ending.")
    return normalized


def _validate_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise CatalogValidationError("Manifest path has an invalid length.")
    if "\\" in value or "\x00" in value or value.startswith("/"):
        raise CatalogValidationError("Manifest path must be a safe POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or not path.parts:
        raise CatalogValidationError("Manifest path must be normalized and relative.")
    for part in path.parts:
        if part in {"", ".", ".."} or part.endswith((".", " ")):
            raise CatalogValidationError("Manifest path contains an unsafe segment.")
        if any(ord(character) < 32 for character in part) or any(
            character in '<>:"|?*' for character in part
        ):
            raise CatalogValidationError("Manifest path contains unsafe characters.")
    return value


def _validate_internal_index_id(value: object) -> str:
    if not isinstance(value, str) or _INDEX_ID_PATTERN.fullmatch(value) is None:
        raise CatalogValidationError("Internal index ID has an invalid format.")
    return value


def _validate_idempotency_reservation_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or _IDEMPOTENCY_RESERVATION_PATTERN.fullmatch(value) is None
    ):
        raise CatalogValidationError("Idempotency reservation ID has an invalid format.")
    return value


def _normalize_manifest(documents: Sequence[DocumentManifest]) -> tuple[DocumentManifest, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise CatalogValidationError("documents must be a sequence of DocumentManifest values.")
    if len(documents) > _MAX_MANIFEST_ITEMS:
        raise CatalogValidationError("Document manifest exceeds the item limit.")
    normalized = tuple(documents)
    if any(not isinstance(item, DocumentManifest) for item in normalized):
        raise CatalogValidationError("Document manifest contains an invalid item.")
    paths = [item.relative_path for item in normalized]
    if len(paths) != len(set(paths)):
        raise CatalogValidationError("Document manifest contains duplicate relative paths.")
    return normalized


def _encode_manifest(documents: tuple[DocumentManifest, ...]) -> str:
    payload = [
        {
            "display_name": item.display_name,
            "relative_path": item.relative_path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in documents
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > _MAX_MANIFEST_JSON_BYTES:
        raise CatalogValidationError("Encoded document manifest exceeds the size limit.")
    return encoded


def _decode_manifest(value: object) -> tuple[DocumentManifest, ...]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_MANIFEST_JSON_BYTES:
        raise CatalogSchemaError()
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda _constant: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(payload, list) or len(payload) > _MAX_MANIFEST_ITEMS:
            raise ValueError
        documents: list[DocumentManifest] = []
        required_keys = {"display_name", "relative_path", "size_bytes", "sha256"}
        for item in payload:
            if not isinstance(item, dict) or set(item) != required_keys:
                raise ValueError
            documents.append(
                DocumentManifest(
                    display_name=item["display_name"],
                    relative_path=item["relative_path"],
                    size_bytes=item["size_bytes"],
                    sha256=item["sha256"],
                )
            )
        return _normalize_manifest(documents)
    except (CatalogError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CatalogSchemaError() from exc


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _record_from_row(row: sqlite3.Row) -> KnowledgeBaseRecord:
    try:
        documents = _decode_manifest(row["manifest_json"])
        error_value = row["error_code"]
        return KnowledgeBaseRecord(
            resource_id=row["resource_id"],
            tenant_id=TenantId(row["tenant_id"]),
            display_name=row["display_name"],
            status=KnowledgeBaseStatus(row["status"]),
            internal_index_id=row["internal_index_id"],
            documents=documents,
            document_count=row["document_count"],
            total_bytes=row["total_bytes"],
            chunk_count=row["chunk_count"],
            error_code=KnowledgeBaseErrorCode(error_value) if error_value is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=row["version"],
            idempotency_reservation_id=row["idempotency_reservation_id"],
        )
    except CatalogSchemaError:
        raise
    except (CatalogError, KeyError, TypeError, ValueError) as exc:
        raise CatalogSchemaError() from exc


def _valid_timestamp(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


__all__ = [
    "CatalogError",
    "CatalogSchemaError",
    "CatalogStorageError",
    "CatalogValidationError",
    "DocumentManifest",
    "DocumentManifestEntry",
    "InvalidStatusTransitionError",
    "KnowledgeBaseCatalog",
    "KnowledgeBaseErrorCode",
    "KnowledgeBaseRecord",
    "KnowledgeBaseStatus",
    "KnowledgeBaseUnavailableError",
]
