"""Thread-safe coordination primitives for application use cases."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from rag_system.jobs import JobId


class ResourceLockPool:
    """A bounded set of deterministic locks keyed by durable resource identity."""

    def __init__(self, slots: int = 64) -> None:
        if not isinstance(slots, int) or slots <= 0:
            raise ValueError("slots must be a positive integer")
        self._locks = tuple(threading.RLock() for _ in range(slots))

    @contextmanager
    def hold(self, resource_id: str) -> Iterator[None]:
        lock = self.lock_for(resource_id)
        with lock:
            yield

    def lock_for(self, resource_id: str) -> threading.RLock:
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        digest = hashlib.blake2s(resource_id.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest, "big") % len(self._locks)
        return self._locks[slot]


@dataclass(frozen=True, slots=True)
class ResourceJob:
    tenant_id: str
    resource_id: str
    job_id: JobId


class ResourceJobRegistry:
    """Maintain a consistent bidirectional resource-to-job association."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_resource: dict[tuple[str, str], JobId] = {}
        self._by_job: dict[tuple[str, JobId], str] = {}

    def bind(self, tenant_id: str, resource_id: str, job_id: JobId) -> None:
        identity = self._validate_identity(tenant_id, resource_id)
        if not isinstance(job_id, JobId):
            raise TypeError("job_id must be a JobId")
        with self._lock:
            previous_job = self._by_resource.get(identity)
            if previous_job is not None and previous_job != job_id:
                self._by_job.pop((tenant_id, previous_job), None)
            previous_resource = self._by_job.get((tenant_id, job_id))
            if previous_resource is not None and previous_resource != resource_id:
                self._by_resource.pop((tenant_id, previous_resource), None)
            self._by_resource[identity] = job_id
            self._by_job[(tenant_id, job_id)] = resource_id

    def job_for(self, tenant_id: str, resource_id: str) -> JobId | None:
        identity = self._validate_identity(tenant_id, resource_id)
        with self._lock:
            return self._by_resource.get(identity)

    def resource_for(self, tenant_id: str, job_id: JobId) -> str | None:
        self._validate_tenant(tenant_id)
        if not isinstance(job_id, JobId):
            raise TypeError("job_id must be a JobId")
        with self._lock:
            return self._by_job.get((tenant_id, job_id))

    def unbind_resource(
        self,
        tenant_id: str,
        resource_id: str,
        *,
        expected_job_id: JobId | None = None,
    ) -> bool:
        identity = self._validate_identity(tenant_id, resource_id)
        with self._lock:
            job_id = self._by_resource.get(identity)
            if job_id is None or (
                expected_job_id is not None and job_id != expected_job_id
            ):
                return False
            self._by_resource.pop(identity, None)
            self._by_job.pop((tenant_id, job_id), None)
            return True

    @staticmethod
    def _validate_identity(tenant_id: str, resource_id: str) -> tuple[str, str]:
        ResourceJobRegistry._validate_tenant(tenant_id)
        if not isinstance(resource_id, str) or not resource_id:
            raise ValueError("resource_id must be a non-empty string")
        return tenant_id, resource_id

    @staticmethod
    def _validate_tenant(tenant_id: str) -> None:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")


__all__ = ["ResourceJob", "ResourceJobRegistry", "ResourceLockPool"]
