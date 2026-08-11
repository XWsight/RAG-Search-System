"""Durable, tenant-isolated idempotency reservations backed by SQLite."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .tenancy import Principal, TenantId


_SCHEMA_VERSION = 1
_MAX_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_RECORDS_PER_TENANT = 1_000_000
_OPERATION_PATTERN = re.compile(r"[a-z][a-z0-9._:-]{0,63}")
_KEY_PATTERN = re.compile(r"[!-~]{8,255}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RESERVATION_ID_PATTERN = re.compile(r"idem_[0-9a-f]{32}")
_RESULT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class IdempotencyError(Exception):
    """Base class for errors safe to classify at an API boundary."""


class IdempotencyValidationError(IdempotencyError, ValueError):
    """An input violates the idempotency store contract."""


class IdempotencyConflictError(IdempotencyError):
    """A key was reused for a different request or result."""

    def __init__(self) -> None:
        super().__init__("Idempotency request conflicts with an existing reservation.")


class IdempotencyUnavailableError(IdempotencyError):
    """The same denial for expired, missing, and cross-tenant reservations."""

    def __init__(self) -> None:
        super().__init__("Idempotency reservation is unavailable.")


class IdempotencyCapacityError(IdempotencyError):
    """A tenant has reached its bounded retained-reservation capacity."""

    def __init__(self) -> None:
        super().__init__("Idempotency reservation capacity is exhausted.")


class IdempotencySchemaError(IdempotencyError):
    """The on-disk schema or stored data violates the component contract."""

    def __init__(self) -> None:
        super().__init__("Idempotency storage schema or data is invalid.")


class IdempotencyStorageError(IdempotencyError):
    """A sanitized SQLite operation failure."""

    def __init__(self) -> None:
        super().__init__("Idempotency storage operation failed.")


@dataclass(frozen=True, slots=True)
class IdempotencyReservation:
    """A safe reservation snapshot that never exposes keys or request digests."""

    reservation_id: str
    operation: str
    created: bool
    resource_id: str | None
    job_id: str | None
    created_at: float
    updated_at: float
    expires_at: float

    def __post_init__(self) -> None:
        _validate_reservation_id(self.reservation_id)
        _validate_operation(self.operation)
        if not isinstance(self.created, bool):
            raise IdempotencySchemaError()
        if (self.resource_id is None) != (self.job_id is None):
            raise IdempotencySchemaError()
        if self.resource_id is not None:
            _validate_result_id(self.resource_id, "resource_id", schema_error=True)
            _validate_result_id(self.job_id, "job_id", schema_error=True)
        if not all(_valid_timestamp(value) for value in self.timestamps):
            raise IdempotencySchemaError()
        if self.updated_at < self.created_at or self.expires_at <= self.created_at:
            raise IdempotencySchemaError()

    @property
    def is_bound(self) -> bool:
        return self.resource_id is not None

    @property
    def timestamps(self) -> tuple[float, float, float]:
        return self.created_at, self.updated_at, self.expires_at


class IdempotencyStore:
    """Persist bounded idempotency reservations with strict tenant isolation.

    Each operation opens and closes its own SQLite connection. WAL mode permits
    concurrent readers, while ``BEGIN IMMEDIATE`` makes reserve/bind operations
    atomic across threads and store instances. Plaintext idempotency keys and
    supplied request digests are context-hashed before persistence.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        ttl_seconds: float = 24 * 60 * 60,
        max_records_per_tenant: int = 10_000,
        timeout_seconds: float = 5.0,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = lambda: f"idem_{secrets.token_hex(16)}",
    ) -> None:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise IdempotencyValidationError("ttl_seconds must be a real number.")
        normalized_ttl = float(ttl_seconds)
        if not math.isfinite(normalized_ttl) or not 1 <= normalized_ttl <= _MAX_TTL_SECONDS:
            raise IdempotencyValidationError(
                f"ttl_seconds must be between 1 and {_MAX_TTL_SECONDS}."
            )
        if isinstance(max_records_per_tenant, bool) or not isinstance(
            max_records_per_tenant, int
        ):
            raise IdempotencyValidationError("max_records_per_tenant must be an integer.")
        if not 1 <= max_records_per_tenant <= _MAX_RECORDS_PER_TENANT:
            raise IdempotencyValidationError(
                "max_records_per_tenant is outside the supported range."
            )
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise IdempotencyValidationError("timeout_seconds must be a real number.")
        normalized_timeout = float(timeout_seconds)
        if not math.isfinite(normalized_timeout) or not 0 < normalized_timeout <= 60:
            raise IdempotencyValidationError("timeout_seconds must be between 0 and 60.")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if not isinstance(database_path, (str, Path)):
            raise TypeError("database_path must be a string or Path")

        path = Path(database_path)
        if path.exists() and not path.is_file():
            raise IdempotencyValidationError("database_path must reference a file.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path = path.resolve()
        except OSError as exc:
            raise IdempotencyStorageError() from exc

        self._database_path = resolved_path
        self._ttl_seconds = normalized_ttl
        self._max_records_per_tenant = max_records_per_tenant
        self._timeout_seconds = normalized_timeout
        self._clock = clock
        self._id_factory = id_factory
        self._write_lock = RLock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def reserve(
        self,
        principal: Principal,
        operation: str,
        key: str,
        request_digest: str,
    ) -> IdempotencyReservation:
        """Atomically create a reservation or replay the matching reservation.

        Reusing a tenant/operation/key tuple with a different request digest
        raises ``IdempotencyConflictError`` without revealing the prior digest.
        """

        tenant_id = _principal_tenant(principal)
        clean_operation = _validate_operation(operation)
        clean_key = _validate_key(key)
        clean_digest = _validate_request_digest(request_digest)
        key_hash = _context_hash(tenant_id, clean_operation, clean_key)
        request_fingerprint = _context_hash(tenant_id, clean_operation, clean_digest)
        now = self._now()

        with self._write_lock, self._write_transaction() as connection:
            self._purge_expired(connection, now)
            row = self._select_identity(
                connection,
                tenant_id,
                clean_operation,
                key_hash,
            )
            if row is not None:
                _validate_stored_row(row)
                if not hmac.compare_digest(row["request_fingerprint"], request_fingerprint):
                    raise IdempotencyConflictError()
                return _reservation_from_row(row, created=False)

            retained = int(
                connection.execute(
                    "SELECT COUNT(*) FROM idempotency_entries WHERE tenant_id = ?",
                    (tenant_id.value,),
                ).fetchone()[0]
            )
            if retained >= self._max_records_per_tenant:
                raise IdempotencyCapacityError()

            expires_at = now + self._ttl_seconds
            for _ in range(8):
                reservation_id = self._new_reservation_id()
                try:
                    connection.execute(
                        """
                        INSERT INTO idempotency_entries (
                            reservation_id, tenant_id, operation, key_hash,
                            request_fingerprint, resource_id, job_id, created_at,
                            updated_at, expires_at
                        ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                        """,
                        (
                            reservation_id,
                            tenant_id.value,
                            clean_operation,
                            key_hash,
                            request_fingerprint,
                            now,
                            now,
                            expires_at,
                        ),
                    )
                except sqlite3.IntegrityError:
                    continue
                row = self._select_reservation(connection, tenant_id, reservation_id)
                if row is None:
                    raise IdempotencyStorageError()
                return _reservation_from_row(row, created=True)
            raise IdempotencyStorageError()

    def bind_result(
        self,
        principal: Principal,
        reservation_id: str,
        resource_id: str,
        job_id: str,
    ) -> IdempotencyReservation:
        """Bind the durable resource/job pair to an owned reservation.

        Rebinding the same pair is idempotent. A different pair conflicts.
        Missing, expired, malformed, and foreign reservation IDs all produce
        the same non-enumerating error.
        """

        tenant_id = _principal_tenant(principal)
        clean_reservation_id = _safe_reservation_lookup_id(reservation_id)
        clean_resource_id = _validate_result_id(resource_id, "resource_id")
        clean_job_id = _validate_result_id(job_id, "job_id")
        now = self._now()

        with self._write_lock, self._write_transaction() as connection:
            self._purge_expired(connection, now)
            row = self._select_reservation(connection, tenant_id, clean_reservation_id)
            if row is None:
                raise IdempotencyUnavailableError()
            _validate_stored_row(row)

            existing_resource = row["resource_id"]
            existing_job = row["job_id"]
            if existing_resource is not None:
                if not (
                    hmac.compare_digest(existing_resource, clean_resource_id)
                    and hmac.compare_digest(existing_job, clean_job_id)
                ):
                    raise IdempotencyConflictError()
                return _reservation_from_row(row, created=False)

            connection.execute(
                """
                UPDATE idempotency_entries
                SET resource_id = ?, job_id = ?, updated_at = ?, expires_at = ?
                WHERE reservation_id = ? AND tenant_id = ?
                """,
                (
                    clean_resource_id,
                    clean_job_id,
                    now,
                    now + self._ttl_seconds,
                    clean_reservation_id,
                    tenant_id.value,
                ),
            )
            row = self._select_reservation(connection, tenant_id, clean_reservation_id)
            if row is None:
                raise IdempotencyUnavailableError()
            return _reservation_from_row(row, created=False)

    def recover_binding(
        self,
        principal: Principal,
        reservation_id: str,
        resource_id: str,
        job_id: str,
    ) -> IdempotencyReservation:
        """Repair a crash-interrupted binding or rotate its ephemeral job ID.

        Recovery may change only the job associated with the same durable
        resource. It can never repoint an idempotency key to another resource.
        """

        tenant_id = _principal_tenant(principal)
        clean_reservation_id = _safe_reservation_lookup_id(reservation_id)
        clean_resource_id = _validate_result_id(resource_id, "resource_id")
        clean_job_id = _validate_result_id(job_id, "job_id")
        now = self._now()

        with self._write_lock, self._write_transaction() as connection:
            self._purge_expired(connection, now)
            row = self._select_reservation(connection, tenant_id, clean_reservation_id)
            if row is None:
                raise IdempotencyUnavailableError()
            _validate_stored_row(row)
            existing_resource = row["resource_id"]
            if existing_resource is not None and not hmac.compare_digest(
                existing_resource,
                clean_resource_id,
            ):
                raise IdempotencyConflictError()
            connection.execute(
                """
                UPDATE idempotency_entries
                SET resource_id = ?, job_id = ?, updated_at = ?, expires_at = ?
                WHERE reservation_id = ? AND tenant_id = ?
                """,
                (
                    clean_resource_id,
                    clean_job_id,
                    now,
                    now + self._ttl_seconds,
                    clean_reservation_id,
                    tenant_id.value,
                ),
            )
            recovered = self._select_reservation(
                connection,
                tenant_id,
                clean_reservation_id,
            )
            if recovered is None:
                raise IdempotencyUnavailableError()
            return _reservation_from_row(recovered, created=False)

    def abandon(
        self,
        principal: Principal,
        reservation_id: str,
    ) -> bool:
        """Atomically remove an owned, unbound reservation.

        This is intentionally limited to failures before any durable resource
        or job has been attached. Bound reservations conflict, while missing,
        expired, malformed, and foreign IDs share one non-enumerating denial.
        """

        tenant_id = _principal_tenant(principal)
        clean_reservation_id = _safe_reservation_lookup_id(reservation_id)
        now = self._now()

        with self._write_lock, self._write_transaction() as connection:
            self._purge_expired(connection, now)
            row = self._select_reservation(connection, tenant_id, clean_reservation_id)
            if row is None:
                raise IdempotencyUnavailableError()
            _validate_stored_row(row)
            if row["resource_id"] is not None:
                raise IdempotencyConflictError()

            cursor = connection.execute(
                """
                DELETE FROM idempotency_entries
                WHERE reservation_id = ? AND tenant_id = ?
                    AND resource_id IS NULL AND job_id IS NULL
                """,
                (clean_reservation_id, tenant_id.value),
            )
            if cursor.rowcount != 1:
                raise IdempotencyUnavailableError()
            return True

    def purge_expired(self) -> int:
        """Delete expired reservations and return the exact number removed."""

        now = self._now()
        with self._write_lock, self._write_transaction() as connection:
            return self._purge_expired(connection, now)

    def _initialize(self) -> None:
        with self._write_lock, self._write_transaction(validate_schema=False) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name IN ('idempotency_meta', 'idempotency_entries')
                    """
                ).fetchall()
            }
            if not tables:
                connection.execute(_CREATE_META_SQL)
                connection.execute(_CREATE_ENTRIES_SQL)
                connection.execute(
                    "CREATE INDEX idx_idempotency_expiry ON idempotency_entries (expires_at)"
                )
                connection.execute(
                    "CREATE INDEX idx_idempotency_tenant ON idempotency_entries (tenant_id)"
                )
                connection.execute(
                    "INSERT INTO idempotency_meta (singleton, schema_version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif tables != {"idempotency_meta", "idempotency_entries"}:
                raise IdempotencySchemaError()
            self._validate_schema(connection)

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        meta_columns = _table_columns(connection, "idempotency_meta")
        entry_columns = _table_columns(connection, "idempotency_entries")
        if meta_columns != _EXPECTED_META_COLUMNS or entry_columns != _EXPECTED_ENTRY_COLUMNS:
            raise IdempotencySchemaError()
        versions = connection.execute(
            "SELECT singleton, schema_version FROM idempotency_meta"
        ).fetchall()
        if len(versions) != 1 or tuple(versions[0]) != (1, _SCHEMA_VERSION):
            raise IdempotencySchemaError()

    def _new_reservation_id(self) -> str:
        try:
            value = self._id_factory()
        except Exception as exc:
            raise IdempotencyStorageError() from exc
        try:
            return _validate_reservation_id(value)
        except (TypeError, ValueError) as exc:
            raise IdempotencyStorageError() from exc

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise IdempotencyValidationError("clock returned an invalid timestamp.") from exc
        if not _valid_timestamp(value):
            raise IdempotencyValidationError("clock returned an invalid timestamp.")
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
                raise IdempotencyStorageError()
            connection.execute("PRAGMA synchronous = NORMAL")
            return connection
        except IdempotencyStorageError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise IdempotencyStorageError() from exc

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
        except IdempotencyError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise IdempotencyStorageError() from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _select_identity(
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        operation: str,
        key_hash: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM idempotency_entries
            WHERE tenant_id = ? AND operation = ? AND key_hash = ?
            """,
            (tenant_id.value, operation, key_hash),
        ).fetchone()

    @staticmethod
    def _select_reservation(
        connection: sqlite3.Connection,
        tenant_id: TenantId,
        reservation_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM idempotency_entries
            WHERE reservation_id = ? AND tenant_id = ?
            """,
            (reservation_id, tenant_id.value),
        ).fetchone()

    @staticmethod
    def _purge_expired(connection: sqlite3.Connection, now: float) -> int:
        cursor = connection.execute(
            "DELETE FROM idempotency_entries WHERE expires_at <= ?",
            (now,),
        )
        return max(0, cursor.rowcount)


_CREATE_META_SQL = """
CREATE TABLE idempotency_meta (
    singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
)
"""


_CREATE_ENTRIES_SQL = """
CREATE TABLE idempotency_entries (
    reservation_id TEXT PRIMARY KEY NOT NULL,
    tenant_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    resource_id TEXT,
    job_id TEXT,
    created_at REAL NOT NULL CHECK (created_at >= 0),
    updated_at REAL NOT NULL CHECK (updated_at >= created_at),
    expires_at REAL NOT NULL CHECK (expires_at > created_at),
    UNIQUE (tenant_id, operation, key_hash),
    CHECK (
        (resource_id IS NULL AND job_id IS NULL)
        OR (resource_id IS NOT NULL AND job_id IS NOT NULL)
    )
)
"""


_EXPECTED_META_COLUMNS = (
    ("singleton", "INTEGER", 1, 1),
    ("schema_version", "INTEGER", 1, 0),
)


_EXPECTED_ENTRY_COLUMNS = (
    ("reservation_id", "TEXT", 1, 1),
    ("tenant_id", "TEXT", 1, 0),
    ("operation", "TEXT", 1, 0),
    ("key_hash", "TEXT", 1, 0),
    ("request_fingerprint", "TEXT", 1, 0),
    ("resource_id", "TEXT", 0, 0),
    ("job_id", "TEXT", 0, 0),
    ("created_at", "REAL", 1, 0),
    ("updated_at", "REAL", 1, 0),
    ("expires_at", "REAL", 1, 0),
)


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[tuple[str, str, int, int], ...]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return tuple(
        (str(row["name"]), str(row["type"]).upper(), int(row["notnull"]), int(row["pk"]))
        for row in rows
    )


def _principal_tenant(principal: Principal) -> TenantId:
    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    return principal.tenant_id


def _validate_operation(value: object) -> str:
    if not isinstance(value, str) or _OPERATION_PATTERN.fullmatch(value) is None:
        raise IdempotencyValidationError(
            "operation must be 1-64 lowercase letters, numbers, dots, colons, underscores, or hyphens."
        )
    return value


def _validate_key(value: object) -> str:
    if not isinstance(value, str) or _KEY_PATTERN.fullmatch(value) is None:
        raise IdempotencyValidationError(
            "key must contain 8-255 visible ASCII characters without whitespace."
        )
    return value


def _validate_request_digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise IdempotencyValidationError(
            "request_digest must be 64 lowercase hexadecimal characters."
        )
    return value


def _validate_reservation_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("reservation_id must be a string")
    if _RESERVATION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("reservation_id has an invalid format")
    return value


def _safe_reservation_lookup_id(value: object) -> str:
    try:
        return _validate_reservation_id(value)
    except (TypeError, ValueError):
        raise IdempotencyUnavailableError() from None


def _validate_result_id(
    value: object,
    field_name: str,
    *,
    schema_error: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or _RESULT_ID_PATTERN.fullmatch(value) is None
        or ".." in value
    ):
        if schema_error:
            raise IdempotencySchemaError()
        raise IdempotencyValidationError(f"{field_name} has an invalid format.")
    return value


def _context_hash(tenant_id: TenantId, operation: str, value: str) -> str:
    payload = f"{tenant_id.value}\0{operation}\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_timestamp(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _validate_stored_row(row: sqlite3.Row) -> None:
    try:
        _validate_reservation_id(row["reservation_id"])
        TenantId(row["tenant_id"])
        _validate_operation(row["operation"])
        if _SHA256_PATTERN.fullmatch(row["key_hash"]) is None:
            raise ValueError
        if _SHA256_PATTERN.fullmatch(row["request_fingerprint"]) is None:
            raise ValueError
        resource_id = row["resource_id"]
        job_id = row["job_id"]
        if (resource_id is None) != (job_id is None):
            raise ValueError
        if resource_id is not None:
            _validate_result_id(resource_id, "resource_id", schema_error=True)
            _validate_result_id(job_id, "job_id", schema_error=True)
        timestamps = (row["created_at"], row["updated_at"], row["expires_at"])
        if not all(_valid_timestamp(value) for value in timestamps):
            raise ValueError
        if float(row["updated_at"]) < float(row["created_at"]):
            raise ValueError
        if float(row["expires_at"]) <= float(row["created_at"]):
            raise ValueError
    except (IdempotencyError, KeyError, TypeError, ValueError, IndexError):
        raise IdempotencySchemaError() from None


def _reservation_from_row(row: sqlite3.Row, *, created: bool) -> IdempotencyReservation:
    _validate_stored_row(row)
    return IdempotencyReservation(
        reservation_id=row["reservation_id"],
        operation=row["operation"],
        created=created,
        resource_id=row["resource_id"],
        job_id=row["job_id"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        expires_at=float(row["expires_at"]),
    )


__all__ = [
    "IdempotencyCapacityError",
    "IdempotencyConflictError",
    "IdempotencyError",
    "IdempotencyReservation",
    "IdempotencySchemaError",
    "IdempotencyStorageError",
    "IdempotencyStore",
    "IdempotencyUnavailableError",
    "IdempotencyValidationError",
]
