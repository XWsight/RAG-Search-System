"""Bounded, tenant-isolated background job execution."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rag_system.job_contracts import (
    CancellationToken,
    JobCancelledError,
    JobCapacityError,
    JobError,
    JobId,
    JobManagerShutdownError,
    JobNotFoundError,
    JobSnapshot,
    JobSnapshotRepository,
    JobStatus,
    JobStorageError,
    JobSubmissionError,
    job_id_value,
    require_job_text,
)


@dataclass(slots=True)
class _JobRecord:
    job_id: JobId
    tenant_id: str
    idempotency_key: str
    status: JobStatus
    created_at: float
    updated_at: float
    last_accessed_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result_json: str | None = None
    error_code: str = ""
    error_message: str = ""
    cancellation: threading.Event = field(default_factory=threading.Event)
    future: Future[Any] | None = None


Task = Callable[[CancellationToken], Mapping[str, Any]]


class JobManager:
    """Run small background tasks with bounded retention and strict isolation.

    ``max_jobs`` bounds queued, running, and retained terminal jobs together,
    so the default thread pool cannot accumulate an unbounded work queue.
    Terminal jobs are removed at their TTL boundary or evicted in LRU order
    when a new submission needs capacity. Active jobs are never evicted.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        max_jobs: int = 128,
        max_jobs_per_tenant: int | None = None,
        ttl_seconds: float = 3_600.0,
        max_result_bytes: int = 32_768,
        max_result_depth: int = 8,
        max_result_items: int = 512,
        clock: Callable[[], float] = time.time,
        executor: Executor | None = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        snapshot_store: JobSnapshotRepository | None = None,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int):
            raise TypeError("max_jobs must be an integer")
        if max_jobs < max_workers:
            raise ValueError("max_jobs must be at least max_workers")
        tenant_limit = max_jobs if max_jobs_per_tenant is None else max_jobs_per_tenant
        if isinstance(tenant_limit, bool) or not isinstance(tenant_limit, int):
            raise TypeError("max_jobs_per_tenant must be an integer")
        if not 1 <= tenant_limit <= max_jobs:
            raise ValueError("max_jobs_per_tenant must be between 1 and max_jobs")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("ttl_seconds must be a real number")
        if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive and finite")
        for name, value in (
            ("max_result_bytes", max_result_bytes),
            ("max_result_depth", max_result_depth),
            ("max_result_items", max_result_items),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if executor is not None and (
            not callable(getattr(executor, "submit", None))
            or not callable(getattr(executor, "shutdown", None))
        ):
            raise TypeError("executor must provide submit and shutdown")
        if snapshot_store is not None and any(
            not callable(getattr(snapshot_store, method, None))
            for method in ("put", "get", "delete", "healthcheck")
        ):
            raise TypeError("snapshot_store does not implement the job snapshot contract")

        self._max_jobs = max_jobs
        self._max_jobs_per_tenant = tenant_limit
        self._ttl_seconds = float(ttl_seconds)
        self._max_result_bytes = max_result_bytes
        self._max_result_depth = max_result_depth
        self._max_result_items = max_result_items
        self._clock = clock
        self._id_factory = id_factory
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="rag-job",
        )
        self._snapshot_store = snapshot_store
        self._snapshot_healthy = True
        self._jobs: OrderedDict[str, _JobRecord] = OrderedDict()
        self._idempotency: dict[tuple[str, str], str] = {}
        self._shutdown = False
        self._executor_shutdown = False
        self._lock = threading.RLock()

    def submit(
        self,
        tenant_id: str,
        task: Task,
        *,
        idempotency_key: str,
    ) -> JobId:
        """Submit once per tenant/idempotency key and return the stable ID."""

        tenant_key = require_job_text(tenant_id, "tenant_id")
        request_key = require_job_text(idempotency_key, "idempotency_key")
        if not callable(task):
            raise TypeError("task must be callable")

        with self._lock:
            if self._shutdown:
                raise JobManagerShutdownError("job manager is shut down")
            now = self._now()
            self._cleanup_expired_locked(now)
            idempotency_identity = (tenant_key, request_key)
            existing_id = self._idempotency.get(idempotency_identity)
            if existing_id is not None:
                existing = self._jobs.get(existing_id)
                if existing is not None:
                    self._touch_locked(existing, now)
                    return existing.job_id
                self._idempotency.pop(idempotency_identity, None)

            self._make_tenant_capacity_locked(tenant_key)
            self._make_capacity_locked()
            job_id = self._new_job_id_locked()
            record = _JobRecord(
                job_id=job_id,
                tenant_id=tenant_key,
                idempotency_key=request_key,
                status=JobStatus.QUEUED,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
            )
            self._jobs[job_id.value] = record
            self._idempotency[idempotency_identity] = job_id.value

            try:
                self._persist_locked(record, strict=True)
                future = self._executor.submit(self._execute, job_id, task)
            except Exception:
                self._remove_locked(job_id.value)
                if self._snapshot_store is not None:
                    try:
                        self._snapshot_store.delete(tenant_key, job_id)
                    except JobError:
                        self._snapshot_healthy = False
                        pass
                raise JobSubmissionError("job could not be scheduled") from None
            record.future = future
            return job_id

    def get(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot:
        """Return one job without revealing whether another tenant owns it."""

        tenant_key = require_job_text(tenant_id, "tenant_id")
        resolved_id = job_id_value(job_id)
        snapshot_store: JobSnapshotRepository | None = None
        with self._lock:
            now = self._now()
            self._cleanup_expired_locked(now)
            record = self._jobs.get(resolved_id)
            if record is None:
                snapshot_store = self._snapshot_store
            else:
                if record.tenant_id != tenant_key:
                    raise JobNotFoundError("job not found")
                self._touch_locked(record, now)
                return self._snapshot_locked(record)
        if snapshot_store is None:
            raise JobNotFoundError("job not found")
        return snapshot_store.get(tenant_key, resolved_id)

    def cancel(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot:
        """Request cooperative cancellation within the owning tenant."""

        tenant_key = require_job_text(tenant_id, "tenant_id")
        resolved_id = job_id_value(job_id)
        snapshot_store: JobSnapshotRepository | None = None
        with self._lock:
            now = self._now()
            self._cleanup_expired_locked(now)
            record = self._jobs.get(resolved_id)
            if record is None:
                snapshot_store = self._snapshot_store
            else:
                if record.tenant_id != tenant_key:
                    raise JobNotFoundError("job not found")
                if not record.status.terminal:
                    record.cancellation.set()
                    future_cancelled = bool(
                        record.future is not None and record.future.cancel()
                    )
                    if future_cancelled:
                        self._finish_locked(record, JobStatus.CANCELLED, now)
                    elif record.status is not JobStatus.CANCELLING:
                        record.status = JobStatus.CANCELLING
                        record.updated_at = now
                        record.last_accessed_at = now
                        self._jobs.move_to_end(record.job_id.value)
                        self._persist_locked(record)
                else:
                    self._touch_locked(record, now)
                return self._snapshot_locked(record)
        if snapshot_store is None:
            raise JobNotFoundError("job not found")
        return snapshot_store.get(tenant_key, resolved_id)

    def cleanup(self) -> int:
        """Remove terminal jobs whose idle TTL has elapsed."""

        with self._lock:
            return self._cleanup_expired_locked(self._now())

    def stats(self, tenant_id: str) -> dict[str, int]:
        """Return status counts for exactly one tenant."""

        tenant_key = require_job_text(tenant_id, "tenant_id")
        with self._lock:
            self._cleanup_expired_locked(self._now())
            records = [record for record in self._jobs.values() if record.tenant_id == tenant_key]
            return {
                status.value: sum(record.status is status for record in records)
                for status in JobStatus
            }

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        """Reject new work and optionally mark unfinished work as cancelled."""

        if not isinstance(wait, bool) or not isinstance(cancel_pending, bool):
            raise TypeError("wait and cancel_pending must be booleans")
        with self._lock:
            if self._executor_shutdown:
                return
            self._shutdown = True
            if cancel_pending:
                now = self._now()
                for record in self._jobs.values():
                    if record.status.terminal:
                        continue
                    record.cancellation.set()
                    future_cancelled = bool(
                        record.future is not None and record.future.cancel()
                    )
                    if future_cancelled:
                        self._finish_locked(record, JobStatus.CANCELLED, now)
                    elif record.status is not JobStatus.CANCELLING:
                        record.status = JobStatus.CANCELLING
                        record.updated_at = now
                        record.last_accessed_at = now
                        self._jobs.move_to_end(record.job_id.value)
                        self._persist_locked(record)
            self._executor_shutdown = True

        try:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        except TypeError:
            self._executor.shutdown(wait=wait)

    def healthcheck(self) -> bool:
        with self._lock:
            locally_healthy = (
                not self._shutdown
                and not self._executor_shutdown
                and self._snapshot_healthy
            )
        if not locally_healthy:
            return False
        try:
            return self._snapshot_store is None or self._snapshot_store.healthcheck() is True
        except Exception:
            return False

    def __enter__(self) -> JobManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def _execute(self, job_id: JobId, task: Task) -> None:
        with self._lock:
            record = self._jobs.get(job_id.value)
            if record is None:
                return
            if record.status is JobStatus.CANCELLING:
                self._finish_locked(record, JobStatus.CANCELLED, self._now())
                return
            if record.status is not JobStatus.QUEUED:
                return
            now = self._now()
            record.status = JobStatus.RUNNING
            record.started_at = now
            record.updated_at = now
            record.last_accessed_at = now
            self._jobs.move_to_end(job_id.value)
            token = CancellationToken(record.cancellation)
            try:
                self._persist_locked(record, strict=True)
            except JobStorageError:
                record.error_code = "job_storage_failed"
                record.error_message = "job snapshot storage operation failed"
                self._finish_locked(record, JobStatus.FAILED, now, persist=False)
                return

        try:
            token.raise_if_cancelled()
            raw_result = task(token)
            result_json = _canonical_result(
                raw_result,
                max_bytes=self._max_result_bytes,
                max_depth=self._max_result_depth,
                max_items=self._max_result_items,
            )
        except JobCancelledError:
            self._complete_cancelled(job_id)
            return
        except _InvalidResultError:
            self._complete_failed(
                job_id,
                code="invalid_result",
                message="task returned an invalid or oversized result",
            )
            return
        except Exception:
            self._complete_failed(
                job_id,
                code="task_failed",
                message="task execution failed",
            )
            return

        with self._lock:
            record = self._jobs.get(job_id.value)
            if record is None or record.status.terminal:
                return
            if record.status not in {JobStatus.RUNNING, JobStatus.CANCELLING}:
                return
            record.result_json = result_json
            self._finish_locked(record, JobStatus.SUCCEEDED, self._now())

    def _complete_cancelled(self, job_id: JobId) -> None:
        with self._lock:
            record = self._jobs.get(job_id.value)
            if record is None or record.status.terminal:
                return
            self._finish_locked(record, JobStatus.CANCELLED, self._now())

    def _complete_failed(self, job_id: JobId, *, code: str, message: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id.value)
            if record is None or record.status.terminal:
                return
            if record.status is JobStatus.CANCELLING:
                self._finish_locked(record, JobStatus.CANCELLED, self._now())
                return
            if record.status is not JobStatus.RUNNING:
                return
            record.error_code = code
            record.error_message = message
            self._finish_locked(record, JobStatus.FAILED, self._now())

    def _finish_locked(
        self,
        record: _JobRecord,
        status: JobStatus,
        now: float,
        *,
        persist: bool = True,
    ) -> None:
        record.status = status
        record.updated_at = now
        record.finished_at = now
        record.last_accessed_at = now
        self._jobs.move_to_end(record.job_id.value)
        if persist:
            self._persist_locked(record)

    def _snapshot_locked(self, record: _JobRecord) -> JobSnapshot:
        result = json.loads(record.result_json) if record.result_json is not None else None
        return JobSnapshot(
            job_id=record.job_id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            result=result,
            error_code=record.error_code,
            error_message=record.error_message,
        )

    def _persist_locked(self, record: _JobRecord, *, strict: bool = False) -> None:
        if self._snapshot_store is None:
            return
        try:
            self._snapshot_store.put(record.tenant_id, self._snapshot_locked(record))
        except JobStorageError:
            self._snapshot_healthy = False
            if strict:
                raise

    def _touch_locked(self, record: _JobRecord, now: float) -> None:
        record.last_accessed_at = now
        self._jobs.move_to_end(record.job_id.value)

    def _cleanup_expired_locked(self, now: float) -> int:
        expired = [
            job_id
            for job_id, record in self._jobs.items()
            if record.status.terminal
            and now - record.last_accessed_at >= self._ttl_seconds
        ]
        for job_id in expired:
            self._remove_locked(job_id)
        return len(expired)

    def _make_capacity_locked(self) -> None:
        while len(self._jobs) >= self._max_jobs:
            terminal_id = next(
                (
                    job_id
                    for job_id, record in self._jobs.items()
                    if record.status.terminal
                ),
                None,
            )
            if terminal_id is None:
                raise JobCapacityError("job capacity reached")
            self._remove_locked(terminal_id)

    def _make_tenant_capacity_locked(self, tenant_id: str) -> None:
        tenant_jobs = [
            (job_id, record)
            for job_id, record in self._jobs.items()
            if record.tenant_id == tenant_id
        ]
        while len(tenant_jobs) >= self._max_jobs_per_tenant:
            terminal = next(
                (
                    (job_id, record)
                    for job_id, record in tenant_jobs
                    if record.status.terminal
                ),
                None,
            )
            if terminal is None:
                raise JobCapacityError("tenant job capacity reached")
            self._remove_locked(terminal[0])
            tenant_jobs.remove(terminal)

    def _remove_locked(self, job_id: str) -> None:
        record = self._jobs.pop(job_id, None)
        if record is None:
            return
        identity = (record.tenant_id, record.idempotency_key)
        if self._idempotency.get(identity) == job_id:
            self._idempotency.pop(identity, None)

    def _new_job_id_locked(self) -> JobId:
        for _ in range(16):
            value = self._id_factory()
            if not isinstance(value, str) or not value.strip():
                raise JobSubmissionError("job ID generation failed")
            candidate = JobId(value.strip())
            if candidate.value not in self._jobs:
                return candidate
        raise JobSubmissionError("job ID generation failed")

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise RuntimeError("clock returned a non-finite value")
        return value


class _InvalidResultError(ValueError):
    pass


def _canonical_result(
    value: object,
    *,
    max_bytes: int,
    max_depth: int,
    max_items: int,
) -> str:
    if not isinstance(value, Mapping):
        raise _InvalidResultError()
    item_counter = [0]
    normalized = _normalize_json_value(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        item_counter=item_counter,
        active_ids=set(),
    )
    try:
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        raise _InvalidResultError() from None
    if len(payload.encode("utf-8")) > max_bytes:
        raise _InvalidResultError()
    return payload


def _normalize_json_value(
    value: object,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    item_counter: list[int],
    active_ids: set[int],
) -> object:
    if depth > max_depth:
        raise _InvalidResultError()
    item_counter[0] += 1
    if item_counter[0] > max_items:
        raise _InvalidResultError()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidResultError()
        return value
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise _InvalidResultError()
        active_ids.add(identity)
        try:
            if isinstance(value, Mapping):
                normalized_mapping: dict[str, object] = {}
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise _InvalidResultError()
                    normalized_mapping[key] = _normalize_json_value(
                        child,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                        item_counter=item_counter,
                        active_ids=active_ids,
                    )
                return normalized_mapping
            return [
                _normalize_json_value(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    item_counter=item_counter,
                    active_ids=active_ids,
                )
                for child in value
            ]
        finally:
            active_ids.remove(identity)
    raise _InvalidResultError()


__all__ = [
    "CancellationToken",
    "JobCancelledError",
    "JobCapacityError",
    "JobError",
    "JobId",
    "JobManager",
    "JobManagerShutdownError",
    "JobNotFoundError",
    "JobSnapshot",
    "JobStatus",
    "JobStorageError",
    "JobSubmissionError",
]
