"""Framework-neutral contracts for asynchronous application jobs."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


@dataclass(frozen=True, slots=True)
class JobId:
    value: str

    def __post_init__(self) -> None:
        normalized = _require_text(self.value, "job_id")
        if len(normalized) > 128 or any(ord(character) < 33 for character in normalized):
            raise ValueError("job_id is outside the supported boundary")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: JobId
    status: JobStatus
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error_code: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, JobId) or not isinstance(self.status, JobStatus):
            raise TypeError("job snapshot identity and status are invalid")
        timestamps = tuple(
            value
            for value in (
                self.created_at,
                self.updated_at,
                self.started_at,
                self.finished_at,
            )
            if value is not None
        )
        if any(not _valid_timestamp(value) for value in timestamps):
            raise ValueError("job snapshot timestamps must be finite and non-negative")
        if self.updated_at < self.created_at:
            raise ValueError("job snapshot update cannot precede creation")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("job snapshot start cannot precede creation")
        if self.finished_at is not None and self.finished_at < self.created_at:
            raise ValueError("job snapshot finish cannot precede creation")
        if self.started_at is not None and self.started_at > self.updated_at:
            raise ValueError("job snapshot start cannot follow its update")
        if self.finished_at is not None and self.finished_at > self.updated_at:
            raise ValueError("job snapshot finish cannot follow its update")
        if self.status.terminal != (self.finished_at is not None):
            raise ValueError("terminal jobs require exactly one finish timestamp")
        if self.result is not None and not isinstance(self.result, dict):
            raise TypeError("job snapshot result must be an object")
        if self.result is not None and self.status is not JobStatus.SUCCEEDED:
            raise ValueError("only successful jobs may contain results")
        if not isinstance(self.error_code, str) or not isinstance(self.error_message, str):
            raise TypeError("job snapshot error fields must be strings")
        if len(self.error_code) > 64 or len(self.error_message) > 512:
            raise ValueError("job snapshot error fields exceed the supported boundary")


class JobError(RuntimeError):
    """Base class for safe job-management errors."""


class JobNotFoundError(JobError):
    pass


class JobCapacityError(JobError):
    pass


class JobManagerShutdownError(JobError):
    pass


class JobSubmissionError(JobError):
    pass


class JobStorageError(JobError):
    """A sanitized durable job snapshot failure."""


class JobCancelledError(Exception):
    """Raised by cooperative tasks after observing cancellation."""


class CancellationToken:
    """Read-only cooperative cancellation signal passed to each task."""

    def __init__(self, event: threading.Event) -> None:
        if not isinstance(event, threading.Event):
            raise TypeError("cancellation event is invalid")
        self._event = event

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelledError()


class JobSnapshotRepository(Protocol):
    """Durable archive used by an executor without owning task execution."""

    def put(self, tenant_id: str, snapshot: JobSnapshot) -> None: ...

    def get(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot: ...

    def delete(self, tenant_id: str, job_id: JobId | str) -> bool: ...

    def healthcheck(self) -> bool: ...


def require_job_text(value: str, name: str) -> str:
    return _require_text(value, name)


def job_id_value(job_id: JobId | str) -> str:
    if isinstance(job_id, JobId):
        return job_id.value
    return JobId(job_id).value


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _valid_timestamp(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


__all__ = [
    "CancellationToken",
    "JobCancelledError",
    "JobCapacityError",
    "JobError",
    "JobId",
    "JobManagerShutdownError",
    "JobNotFoundError",
    "JobSnapshot",
    "JobSnapshotRepository",
    "JobStatus",
    "JobStorageError",
    "JobSubmissionError",
    "job_id_value",
    "require_job_text",
]
