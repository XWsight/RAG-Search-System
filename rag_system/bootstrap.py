"""Composition roots for the local UI and the durable API service."""

from __future__ import annotations

import json
import logging
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.catalog import KnowledgeBaseCatalog
from rag_system.config import SecretValue, Settings, load_settings
from rag_system.file_store import TenantFileStore
from rag_system.index_manager import IndexManager
from rag_system.idempotency import IdempotencyStore
from rag_system.jobs import JobManager
from rag_system.metrics import create_operational_metrics
from rag_system.observability import JsonEventLogger
from rag_system.platform import RagPlatform
from rag_system.providers import ZhipuChatModel, ZhipuWebSearch
from rag_system.rate_limit import TokenBucketRateLimiter
from rag_system.retrieval import ChromaIndexRepository
from rag_system.service import RagService
from rag_system.tenancy import ApiKeyAuthenticator, Principal, TenantId


@dataclass(frozen=True, slots=True)
class ProductionRuntime:
    settings: Settings
    platform: RagPlatform
    authenticator: ApiKeyAuthenticator
    principals: tuple[Principal, ...]
    rate_limiter: TokenBucketRateLimiter
    event_logger: JsonEventLogger
    storage_lease: StorageRootLease

    def close(self) -> None:
        try:
            self.platform.close()
        finally:
            self.storage_lease.close()

    def ready(self) -> bool:
        """Check durable metadata and document storage without loading models."""

        try:
            if not self.platform.file_store.healthcheck():
                return False
            if not self.platform.jobs.healthcheck():
                return False
            self.platform.catalog.list(self.principals[0], limit=1, offset=0)
        except Exception:
            return False
        return True


class StorageRootLease:
    """Hold an OS-level exclusive lease for one single-node storage root."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    @classmethod
    def acquire(cls, storage_root: Path) -> StorageRootLease:
        lease = cls(storage_root / ".rag-studio.instance")
        lease._acquire()
        return lease

    def _acquire(self) -> None:
        if self._path.is_symlink():
            raise RuntimeError("storage instance lease path is unsafe")
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(self._path, flags, 0o600)
            handle = os.fdopen(descriptor, "r+b", buffering=0)
            descriptor = None
        except (OSError, ValueError):
            if descriptor is not None:
                os.close(descriptor)
            raise RuntimeError("storage instance lease path is unsafe") from None
        try:
            handle_stat = os.fstat(handle.fileno())
            path_stat = os.lstat(self._path)
            if (
                not stat.S_ISREG(handle_stat.st_mode)
                or stat.S_ISLNK(path_stat.st_mode)
                or (handle_stat.st_dev, handle_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise OSError("unsafe storage lease inode")
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RuntimeError("storage root is already in use by another process") from None
        self._handle = handle

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def build_service_from_settings(settings: Settings) -> RagService:
    validated = settings.validate()
    repository = ChromaIndexRepository(validated)
    manager = IndexManager(validated, repository)
    chat_model = ZhipuChatModel(validated)
    return RagService(
        settings=validated,
        index_manager=manager,
        chat_model=chat_model,
        web_search=ZhipuWebSearch(validated),
        query_planner=chat_model,
    )


def build_service(*, dotenv_path: Path | None = None) -> tuple[RagService, Settings]:
    settings = load_settings(dotenv_path=dotenv_path)
    return build_service_from_settings(settings), settings


def parse_api_credentials(
    encoded: SecretValue,
) -> tuple[ApiKeyAuthenticator, tuple[Principal, ...]]:
    """Parse a bounded strict JSON credential map without retaining raw keys."""

    if not isinstance(encoded, SecretValue):
        raise TypeError("encoded credentials must be a SecretValue")
    raw = encoded.reveal()
    if not raw or len(raw.encode("utf-8")) > 64 * 1024:
        raise ValueError("RAG_API_KEYS_JSON is missing or too large")
    try:
        payload = json.loads(raw, object_pairs_hook=_strict_object)
    except (TypeError, ValueError):
        raise ValueError("RAG_API_KEYS_JSON is invalid") from None
    if not isinstance(payload, dict) or not 1 <= len(payload) <= 100:
        raise ValueError("RAG_API_KEYS_JSON must contain 1-100 credentials")

    credentials: dict[str, Principal] = {}
    principals: list[Principal] = []
    try:
        for api_key, descriptor in payload.items():
            if not isinstance(api_key, str) or not isinstance(descriptor, dict):
                raise ValueError
            if set(descriptor) != {"subject", "tenant_id", "roles"}:
                raise ValueError
            roles = descriptor["roles"]
            if (
                not isinstance(roles, list)
                or not 1 <= len(roles) <= 16
                or any(not isinstance(role, str) for role in roles)
            ):
                raise ValueError
            principal = Principal(
                subject=descriptor["subject"],
                tenant_id=TenantId(descriptor["tenant_id"]),
                roles=frozenset(roles),
            )
            credentials[api_key] = principal
            principals.append(principal)
        authenticator = ApiKeyAuthenticator.from_mapping(credentials)
    except (TypeError, ValueError):
        raise ValueError("RAG_API_KEYS_JSON is invalid") from None

    unique_principals = tuple(
        {
            (item.subject, item.tenant_id.value, tuple(sorted(item.roles))): item
            for item in principals
        }.values()
    )
    return authenticator, unique_principals


def build_production_runtime(*, dotenv_path: Path | None = None) -> ProductionRuntime:
    settings = load_settings(dotenv_path=dotenv_path)
    if not settings.persist_data:
        raise ValueError("RAG_PERSIST_DATA must be true for the production API")

    storage_root = settings.storage_root.expanduser().resolve()
    if storage_root == Path(storage_root.anchor):
        raise ValueError("RAG_STORAGE_ROOT cannot be a filesystem root")
    storage_root.mkdir(parents=True, exist_ok=True)

    storage_lease = StorageRootLease.acquire(storage_root)
    platform: RagPlatform | None = None
    try:
        authenticator, principals = parse_api_credentials(settings.api_keys_json)
        metrics = create_operational_metrics()
        platform = RagPlatform(
            settings=settings,
            service=build_service_from_settings(settings),
            catalog=KnowledgeBaseCatalog(storage_root / "catalog.sqlite3"),
            file_store=TenantFileStore(
                storage_root / "documents",
                max_file_bytes=settings.max_file_bytes,
                max_total_bytes=settings.max_tenant_storage_bytes,
                max_files_per_tenant=settings.max_files_per_tenant,
            ),
            jobs=JobManager(
                max_workers=settings.job_workers,
                max_jobs=settings.max_jobs,
                max_jobs_per_tenant=settings.max_jobs_per_tenant,
                ttl_seconds=settings.job_ttl_seconds,
            ),
            idempotency=IdempotencyStore(
                storage_root / "idempotency.sqlite3",
                ttl_seconds=24 * 60 * 60,
                max_records_per_tenant=10_000,
            ),
            metrics=metrics,
        )
        rate_limiter = TokenBucketRateLimiter(
            rate_per_second=settings.rate_limit_per_second,
            capacity=settings.rate_limit_capacity,
            max_keys=settings.rate_limit_max_tenants,
            key_ttl_seconds=settings.job_ttl_seconds,
        )
        event_logger = JsonEventLogger(
            logging.getLogger("rag_system.events"),
            known_secrets=(settings.api_key.reveal(),),
        )
        platform.recover_incomplete(principals)
    except Exception:
        try:
            if platform is not None:
                try:
                    platform.close()
                except Exception:
                    logging.getLogger("rag_system.bootstrap").error(
                        "production runtime cleanup failed during startup"
                    )
        finally:
            storage_lease.close()
        raise
    return ProductionRuntime(
        settings=settings,
        platform=platform,
        authenticator=authenticator,
        principals=principals,
        rate_limiter=rate_limiter,
        event_logger=event_logger,
        storage_lease=storage_lease,
    )


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result
