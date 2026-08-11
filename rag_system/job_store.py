"""Durable, tenant-isolated SQLite archive for bounded job snapshots."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, cast

from rag_system.job_contracts import (
    JobId,
    JobNotFoundError,
    JobSnapshot,
    JobStatus,
    JobStorageError,
    job_id_value,
    require_job_text,
)


_SCHEMA_VERSION = 1
_EXPECTED_COLUMNS = (
    ("job_id", "TEXT", 1, None, 2),
    ("tenant_id", "TEXT", 1, None, 1),
    ("status", "TEXT", 1, None, 0),
    ("created_at", "REAL", 1, None, 0),
    ("updated_at", "REAL", 1, None, 0),
    ("started_at", "REAL", 0, None, 0),
    ("finished_at", "REAL", 0, None, 0),
    ("result_json", "TEXT", 0, None, 0),
    ("error_code", "TEXT", 1, None, 0),
    ("error_message", "TEXT", 1, None, 0),
)
_TERMINAL_STATUSES = tuple(status.value for status in JobStatus if status.terminal)
_ACTIVE_STATUSES = tuple(status.value for status in JobStatus if not status.terminal)


class SqliteJobSnapshotStore:
    """Persist job history while leaving task scheduling to ``JobManager``."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        ttl_seconds: float = 7 * 24 * 60 * 60,
        max_records_per_tenant: int = 10_000,
        clock: Callable[[], float] = time.time,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(ttl_seconds, (int, float)) or isinstance(ttl_seconds, bool):
            raise TypeError("ttl_seconds must be a real number")
        if not math.isfinite(float(ttl_seconds)) or ttl_seconds < 60:
            raise ValueError("ttl_seconds must be finite and at least 60 seconds")
        if (
            not isinstance(max_records_per_tenant, int)
            or isinstance(max_records_per_tenant, bool)
            or not 1 <= max_records_per_tenant <= 1_000_000
        ):
            raise ValueError("max_records_per_tenant is outside the supported range")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be between 0 and 60")
        path = Path(database_path)
        if path.exists() and not path.is_file():
            raise ValueError("database_path must reference a file")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = path.resolve()
        self._ttl_seconds = float(ttl_seconds)
        self._max_records_per_tenant = max_records_per_tenant
        self._clock = clock
        self._timeout_seconds = float(timeout_seconds)
        self._write_lock = RLock()
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def put(self, tenant_id: str, snapshot: JobSnapshot) -> None:
        tenant = _validate_tenant_id(tenant_id)
        if not isinstance(snapshot, JobSnapshot):
            raise TypeError("snapshot must be a JobSnapshot")
        payload = _encode_result(snapshot.result)
        _validate_safe_error(snapshot.error_code, snapshot.error_message)
        with self._write_lock, self._transaction() as connection:
            existing = connection.execute(
                "SELECT created_at FROM job_snapshots "
                "WHERE job_id = ? AND tenant_id = ?",
                (snapshot.job_id.value, tenant),
            ).fetchone()
            if existing is not None and float(existing["created_at"]) != snapshot.created_at:
                raise JobStorageError("job snapshot identity conflicts with stored data")
            connection.execute(
                """
                INSERT INTO job_snapshots (
                    job_id, tenant_id, status, created_at, updated_at,
                    started_at, finished_at, result_json, error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, job_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    result_json = excluded.result_json,
                    error_code = excluded.error_code,
                    error_message = excluded.error_message
                """,
                (
                    snapshot.job_id.value,
                    tenant,
                    snapshot.status.value,
                    snapshot.created_at,
                    snapshot.updated_at,
                    snapshot.started_at,
                    snapshot.finished_at,
                    payload,
                    snapshot.error_code,
                    snapshot.error_message,
                ),
            )
            self._purge_expired(connection, self._now())
            self._enforce_capacity(connection, tenant)

    def get(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot:
        tenant = _validate_tenant_id(tenant_id)
        resolved_id = job_id_value(job_id)
        try:
            with self._write_lock, self._transaction() as connection:
                self._purge_expired(connection, self._now())
                row = connection.execute(
                    "SELECT * FROM job_snapshots WHERE job_id = ? AND tenant_id = ?",
                    (resolved_id, tenant),
                ).fetchone()
        except sqlite3.Error as error:
            raise JobStorageError("job snapshot storage operation failed") from error
        if row is None:
            raise JobNotFoundError("job not found")
        return _snapshot_from_row(row)

    def delete(self, tenant_id: str, job_id: JobId | str) -> bool:
        tenant = _validate_tenant_id(tenant_id)
        resolved_id = job_id_value(job_id)
        with self._write_lock, self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM job_snapshots WHERE job_id = ? AND tenant_id = ?",
                (resolved_id, tenant),
            )
            return cursor.rowcount == 1

    def healthcheck(self) -> bool:
        try:
            with self._connection() as connection:
                self._validate_schema(connection)
                connection.execute("SELECT 1 FROM job_snapshots LIMIT 1").fetchone()
            return True
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return False

    def _initialize(self) -> None:
        with self._write_lock:
            try:
                with self._connection() as connection:
                    tables = {
                        str(row[0])
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        ).fetchall()
                    }
                    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                    if not tables and version == 0:
                        connection.execute(_CREATE_TABLE_SQL)
                        connection.execute(
                            "CREATE INDEX idx_job_snapshots_tenant_updated "
                            "ON job_snapshots (tenant_id, updated_at)"
                        )
                        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                    elif tables != {"job_snapshots"} or version != _SCHEMA_VERSION:
                        raise JobStorageError("job snapshot schema is invalid")
                    self._validate_schema(connection)
                    connection.commit()
            except JobStorageError:
                raise
            except sqlite3.Error as error:
                raise JobStorageError("job snapshot storage initialization failed") from error

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        columns = tuple(
            (
                str(row[1]),
                str(row[2]).upper(),
                int(row[3]),
                row[4],
                int(row[5]),
            )
            for row in connection.execute("PRAGMA table_info(job_snapshots)").fetchall()
        )
        index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            ("idx_job_snapshots_tenant_updated",),
        ).fetchone()
        if version != _SCHEMA_VERSION or columns != _EXPECTED_COLUMNS or index is None:
            raise JobStorageError("job snapshot schema is invalid")

    def recover_interrupted(self) -> int:
        """Mark work owned by a terminated process as safely failed."""

        now = self._now()
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        with self._write_lock, self._transaction() as connection:
            cursor = connection.execute(
                f"""
                UPDATE job_snapshots
                SET status = ?, updated_at = ?, finished_at = ?, result_json = NULL,
                    error_code = ?, error_message = ?
                WHERE status IN ({placeholders})
                """,
                (
                    JobStatus.FAILED.value,
                    now,
                    now,
                    "worker_restarted",
                    "job execution was interrupted by a process restart",
                    *_ACTIVE_STATUSES,
                ),
            )
            self._purge_expired(connection, now)
            return max(0, cursor.rowcount)

    def _enforce_capacity(self, connection: sqlite3.Connection, tenant_id: str) -> None:
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM job_snapshots WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()[0]
        )
        excess = count - self._max_records_per_tenant
        if excess <= 0:
            return
        placeholders = ",".join("?" for _ in _TERMINAL_STATUSES)
        rows = connection.execute(
            f"""
            SELECT job_id FROM job_snapshots
            WHERE tenant_id = ? AND status IN ({placeholders})
            ORDER BY updated_at ASC, job_id ASC LIMIT ?
            """,
            (tenant_id, *_TERMINAL_STATUSES, excess),
        ).fetchall()
        if len(rows) != excess:
            raise JobStorageError("job snapshot capacity is exhausted")
        connection.executemany(
            "DELETE FROM job_snapshots WHERE job_id = ? AND tenant_id = ?",
            ((str(row["job_id"]), tenant_id) for row in rows),
        )

    def _purge_expired(self, connection: sqlite3.Connection, now: float) -> None:
        cutoff = now - self._ttl_seconds
        placeholders = ",".join("?" for _ in _TERMINAL_STATUSES)
        connection.execute(
            f"DELETE FROM job_snapshots WHERE status IN ({placeholders}) AND updated_at <= ?",
            (*_TERMINAL_STATUSES, cutoff),
        )

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value) or value < 0:
            raise JobStorageError("job snapshot clock returned an invalid timestamp")
        return value

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=self._timeout_seconds,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                except Exception:
                    connection.rollback()
                    raise
                connection.commit()
        except JobStorageError:
            raise
        except sqlite3.Error as error:
            raise JobStorageError("job snapshot storage operation failed") from error


def _validate_tenant_id(value: str) -> str:
    normalized = require_job_text(value, "tenant_id")
    if len(normalized) > 128 or any(ord(character) < 32 for character in normalized):
        raise ValueError("tenant_id is outside the supported boundary")
    return normalized


def _validate_safe_error(code: str, message: str) -> None:
    if not isinstance(code, str) or not isinstance(message, str):
        raise TypeError("job error fields must be strings")
    if len(code) > 64 or len(message) > 512:
        raise ValueError("job error fields exceed the supported boundary")


def _encode_result(result: dict[str, Any] | None) -> str | None:
    if result is None:
        return None
    try:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        raise JobStorageError("job snapshot result is invalid") from None
    if len(encoded.encode("utf-8")) > 32_768:
        raise JobStorageError("job snapshot result exceeds the storage boundary")
    return encoded


def _snapshot_from_row(row: sqlite3.Row) -> JobSnapshot:
    try:
        raw_result = row["result_json"]
        result = json.loads(raw_result) if raw_result is not None else None
        if result is not None and not isinstance(result, dict):
            raise ValueError
        return JobSnapshot(
            job_id=JobId(str(row["job_id"])),
            status=JobStatus(str(row["status"])),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            started_at=cast(float | None, row["started_at"]),
            finished_at=cast(float | None, row["finished_at"]),
            result=result,
            error_code=str(row["error_code"]),
            error_message=str(row["error_message"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise JobStorageError("stored job snapshot is invalid") from error


_CREATE_TABLE_SQL = """
CREATE TABLE job_snapshots (
    job_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued','running','cancelling','succeeded','failed','cancelled')
    ),
    created_at REAL NOT NULL CHECK (created_at >= 0),
    updated_at REAL NOT NULL CHECK (updated_at >= created_at),
    started_at REAL,
    finished_at REAL,
    result_json TEXT,
    error_code TEXT NOT NULL,
    error_message TEXT NOT NULL,
    PRIMARY KEY (tenant_id, job_id)
)
"""


__all__ = ["SqliteJobSnapshotStore"]
