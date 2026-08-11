"""Tenant-isolated, bounded storage for untrusted uploaded files."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
import unicodedata
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


_RESOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"/\\|?*')
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class FileStoreError(RuntimeError):
    """Base class for safe storage errors."""


class FileStoreSecurityError(FileStoreError):
    pass


class InvalidResourceIdError(FileStoreError):
    pass


class InvalidFileNameError(FileStoreError):
    pass


class DuplicateResourceError(FileStoreError):
    pass


class ResourceNotFoundError(FileStoreError):
    pass


class StorageLimitError(FileStoreError):
    pass


class FileStoreIOError(FileStoreError):
    pass


@dataclass(frozen=True, slots=True)
class StoredFile:
    relative_path: str
    display_name: str
    size: int
    sha256: str


class TenantFileStore:
    """Store one ordinary file in each tenant/resource directory.

    Tenant IDs never become path components directly. Every operation derives
    a fixed SHA-256 tenant directory and a strictly validated resource ID.
    Directories containing links, reparse points, or unexpected entries are
    rejected instead of traversed or recursively removed.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_file_bytes: int = 20 * 1024 * 1024,
        max_total_bytes: int = 100 * 1024 * 1024,
        max_files_per_tenant: int = 32,
        copy_buffer_size: int = 64 * 1024,
    ) -> None:
        for name, value in (
            ("max_file_bytes", max_file_bytes),
            ("max_total_bytes", max_total_bytes),
            ("max_files_per_tenant", max_files_per_tenant),
            ("copy_buffer_size", copy_buffer_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")

        requested_root = Path(root).absolute()
        if _lexists(requested_root):
            _assert_directory(requested_root, "storage root")
        else:
            try:
                requested_root.mkdir(parents=True, exist_ok=False)
            except OSError:
                raise FileStoreIOError("storage root could not be created") from None
            _assert_directory(requested_root, "storage root")

        try:
            resolved_root = requested_root.resolve(strict=True)
            root_stat = os.lstat(resolved_root)
        except OSError:
            raise FileStoreSecurityError("storage root could not be verified") from None

        self._root = resolved_root
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._max_files_per_tenant = max_files_per_tenant
        self._copy_buffer_size = copy_buffer_size
        self._lock = threading.RLock()

    @property
    def root(self) -> Path:
        return self._root

    def planned_relative_path(
        self,
        tenant_id: str,
        resource_id: str,
        display_name: str,
    ) -> str:
        """Return the validated durable path before bytes are committed."""

        tenant_key = _require_text(tenant_id, "tenant_id")
        resource_key = _validate_resource_id(resource_id)
        safe_name = _validate_display_name(display_name)
        with self._lock:
            self._verify_root()
            tenant_directory = self._tenant_directory(tenant_key)
            return PurePosixPath(
                tenant_directory.name,
                resource_key,
                safe_name,
            ).as_posix()

    def healthcheck(self) -> bool:
        """Verify the configured root still refers to the original directory."""

        with self._lock:
            self._verify_root()
        return True

    def save(
        self,
        tenant_id: str,
        resource_id: str,
        display_name: str,
        source: bytes | bytearray | memoryview | BinaryIO,
    ) -> StoredFile:
        """Stream and atomically commit one new resource."""

        tenant_key = _require_text(tenant_id, "tenant_id")
        resource_key = _validate_resource_id(resource_id)
        safe_name = _validate_display_name(display_name)
        if not isinstance(source, (bytes, bytearray, memoryview)) and not callable(
            getattr(source, "read", None)
        ):
            raise TypeError("source must be bytes or a binary file-like object")

        with self._lock:
            self._verify_root()
            tenant_directory = self._tenant_directory(tenant_key)
            self._ensure_tenant_directory(tenant_directory)
            resource_directory = tenant_directory / resource_key
            self._assert_derived_path(resource_directory, tenant_directory)
            if _lexists(resource_directory):
                if _is_reparse(resource_directory):
                    raise FileStoreSecurityError("resource path is a link or reparse point")
                raise DuplicateResourceError("resource already exists")

            file_count, total_bytes = self._tenant_usage(tenant_directory)
            if file_count >= self._max_files_per_tenant:
                raise StorageLimitError("tenant file count limit reached")
            if total_bytes >= self._max_total_bytes:
                raise StorageLimitError("tenant storage limit reached")

            try:
                resource_directory.mkdir(exist_ok=False)
            except FileExistsError:
                raise DuplicateResourceError("resource already exists") from None
            except OSError:
                raise FileStoreIOError("resource directory could not be created") from None
            _assert_directory(resource_directory, "resource directory")

            temporary = resource_directory / f".upload-{uuid.uuid4().hex}.tmp"
            target = resource_directory / safe_name
            digest = hashlib.sha256()
            size = 0
            committed = False
            try:
                with temporary.open("xb") as destination:
                    for chunk in self._source_chunks(source):
                        next_size = size + len(chunk)
                        if next_size > self._max_file_bytes:
                            raise StorageLimitError("single file size limit exceeded")
                        if total_bytes + next_size > self._max_total_bytes:
                            raise StorageLimitError("tenant storage limit exceeded")
                        destination.write(chunk)
                        digest.update(chunk)
                        size = next_size
                    destination.flush()
                    os.fsync(destination.fileno())

                if _lexists(target):
                    raise DuplicateResourceError("target file already exists")
                os.replace(temporary, target)
                committed = True
                _assert_regular_file(target, "stored file")
                self._assert_derived_path(target.resolve(strict=True), resource_directory)
            except (StorageLimitError, DuplicateResourceError, FileStoreSecurityError):
                self._rollback_resource(resource_directory, temporary, target, committed)
                raise
            except Exception:
                self._rollback_resource(resource_directory, temporary, target, committed)
                raise FileStoreIOError("resource could not be saved") from None

            relative_path = PurePosixPath(
                tenant_directory.name,
                resource_key,
                safe_name,
            ).as_posix()
            return StoredFile(
                relative_path=relative_path,
                display_name=safe_name,
                size=size,
                sha256=digest.hexdigest(),
            )

    def resolve(self, tenant_id: str, resource_id: str) -> Path:
        """Resolve one resource only after validating its complete path."""

        tenant_key = _require_text(tenant_id, "tenant_id")
        resource_key = _validate_resource_id(resource_id)
        with self._lock:
            self._verify_root()
            tenant_directory = self._tenant_directory(tenant_key)
            resource_directory = tenant_directory / resource_key
            return self._resolve_resource_file(tenant_directory, resource_directory)

    def delete(self, tenant_id: str, resource_id: str) -> bool:
        """Delete one exact resource directory without recursive traversal."""

        tenant_key = _require_text(tenant_id, "tenant_id")
        resource_key = _validate_resource_id(resource_id)
        with self._lock:
            self._verify_root()
            tenant_directory = self._tenant_directory(tenant_key)
            resource_directory = tenant_directory / resource_key
            if not _lexists(tenant_directory) or not _lexists(resource_directory):
                return False

            _assert_directory(tenant_directory, "tenant directory")
            _assert_directory(resource_directory, "resource directory")
            self._assert_derived_path(resource_directory, tenant_directory)
            try:
                entries = tuple(resource_directory.iterdir())
            except OSError:
                raise FileStoreSecurityError(
                    "resource directory could not be inspected"
                ) from None
            if not entries:
                try:
                    resource_directory.rmdir()
                except OSError:
                    raise FileStoreIOError("resource could not be deleted") from None
                return True

            stored_path = self._resolve_resource_file(tenant_directory, resource_directory)
            _assert_regular_file(stored_path, "stored file")
            try:
                stored_path.unlink()
                resource_directory.rmdir()
            except OSError:
                raise FileStoreIOError("resource could not be deleted") from None
            return True

    def _tenant_directory(self, tenant_id: str) -> Path:
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        tenant_directory = self._root / f"tenant-{digest}"
        self._assert_derived_path(tenant_directory, self._root)
        return tenant_directory

    def _ensure_tenant_directory(self, tenant_directory: Path) -> None:
        if _lexists(tenant_directory):
            _assert_directory(tenant_directory, "tenant directory")
            return
        try:
            tenant_directory.mkdir(exist_ok=False)
        except FileExistsError:
            _assert_directory(tenant_directory, "tenant directory")
        except OSError:
            raise FileStoreIOError("tenant directory could not be created") from None
        _assert_directory(tenant_directory, "tenant directory")

    def _tenant_usage(self, tenant_directory: Path) -> tuple[int, int]:
        _assert_directory(tenant_directory, "tenant directory")
        file_count = 0
        total_bytes = 0
        try:
            entries = tuple(tenant_directory.iterdir())
        except OSError:
            raise FileStoreSecurityError("tenant directory could not be inspected") from None
        for resource_directory in entries:
            if not _RESOURCE_ID.fullmatch(resource_directory.name):
                raise FileStoreSecurityError("tenant directory contains an unexpected entry")
            _assert_directory(resource_directory, "resource directory")
            stored_path = self._resolve_resource_file(tenant_directory, resource_directory)
            file_count += 1
            total_bytes += os.lstat(stored_path).st_size
        return file_count, total_bytes

    def _resolve_resource_file(
        self,
        tenant_directory: Path,
        resource_directory: Path,
    ) -> Path:
        if not _lexists(tenant_directory) or not _lexists(resource_directory):
            raise ResourceNotFoundError("resource not found")
        _assert_directory(tenant_directory, "tenant directory")
        _assert_directory(resource_directory, "resource directory")
        self._assert_derived_path(resource_directory, tenant_directory)
        try:
            entries = tuple(resource_directory.iterdir())
        except OSError:
            raise FileStoreSecurityError("resource directory could not be inspected") from None
        if len(entries) != 1:
            raise FileStoreSecurityError("resource directory must contain exactly one file")
        stored_path = entries[0]
        _assert_regular_file(stored_path, "stored file")
        try:
            resolved_file = stored_path.resolve(strict=True)
        except OSError:
            raise FileStoreSecurityError("stored file could not be resolved") from None
        self._assert_derived_path(resolved_file, resource_directory)
        self._assert_derived_path(resolved_file, self._root)
        return stored_path

    def _source_chunks(
        self,
        source: bytes | bytearray | memoryview | BinaryIO,
    ) -> Iterator[bytes]:
        if isinstance(source, (bytes, bytearray, memoryview)):
            view = memoryview(source).cast("B")
            for offset in range(0, len(view), self._copy_buffer_size):
                yield bytes(view[offset : offset + self._copy_buffer_size])
            return

        while True:
            chunk = source.read(self._copy_buffer_size)
            if chunk in (b"", None):
                if chunk is None:
                    raise TypeError("binary stream read returned None")
                return
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise TypeError("binary stream must return bytes")
            yield bytes(chunk)

    def _rollback_resource(
        self,
        resource_directory: Path,
        temporary: Path,
        target: Path,
        committed: bool,
    ) -> None:
        candidate = target if committed else temporary
        try:
            if _lexists(candidate) and not _is_reparse(candidate):
                candidate_stat = os.lstat(candidate)
                if stat.S_ISREG(candidate_stat.st_mode):
                    candidate.unlink()
            if _lexists(resource_directory) and not _is_reparse(resource_directory):
                resource_directory.rmdir()
        except OSError:
            # Never broaden cleanup or recurse after a rollback failure.
            pass

    def _verify_root(self) -> None:
        _assert_directory(self._root, "storage root")
        try:
            current = os.lstat(self._root)
        except OSError:
            raise FileStoreSecurityError("storage root could not be verified") from None
        if (current.st_dev, current.st_ino) != self._root_identity:
            raise FileStoreSecurityError("storage root identity changed")

    @staticmethod
    def _assert_derived_path(path: Path, parent: Path) -> None:
        try:
            path.relative_to(parent)
        except ValueError:
            raise FileStoreSecurityError("path escapes its storage boundary") from None


def _validate_resource_id(value: str) -> str:
    resource_id = _require_text(value, "resource_id")
    if resource_id != value or not _RESOURCE_ID.fullmatch(resource_id):
        raise InvalidResourceIdError("resource_id has an invalid format")
    if resource_id.upper() in _WINDOWS_RESERVED:
        raise InvalidResourceIdError("resource_id is reserved")
    return resource_id


def _validate_display_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("display_name must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized in {".", ".."}:
        raise InvalidFileNameError("display_name cannot be empty or relative")
    if normalized != normalized.strip() or normalized.endswith((".", " ")):
        raise InvalidFileNameError("display_name has unsafe leading or trailing characters")
    try:
        encoded_name = normalized.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidFileNameError("display_name contains invalid Unicode") from None
    if len(normalized) > 240 or len(encoded_name) > 255:
        raise InvalidFileNameError("display_name is too long")
    if any(character in _WINDOWS_INVALID_CHARACTERS for character in normalized):
        raise InvalidFileNameError("display_name contains a path or reserved character")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise InvalidFileNameError("display_name contains a control character")
    if Path(normalized).name != normalized or normalized.startswith(".upload-"):
        raise InvalidFileNameError("display_name is not a safe single filename")
    device_name = normalized.split(".", 1)[0].upper()
    if device_name in _WINDOWS_RESERVED:
        raise InvalidFileNameError("display_name is reserved")
    return normalized


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} contains invalid Unicode") from None
    return normalized


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_reparse(path: Path) -> bool:
    try:
        path_stat = os.lstat(path)
    except OSError:
        raise FileStoreSecurityError("path could not be inspected") from None
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    )


def _assert_directory(path: Path, label: str) -> None:
    try:
        path_stat = os.lstat(path)
    except OSError:
        raise FileStoreSecurityError(f"{label} could not be inspected") from None
    if stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        raise FileStoreSecurityError(f"{label} cannot be a link or reparse point")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise FileStoreSecurityError(f"{label} must be a directory")


def _assert_regular_file(path: Path, label: str) -> None:
    try:
        path_stat = os.lstat(path)
    except OSError:
        raise FileStoreSecurityError(f"{label} could not be inspected") from None
    if stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        raise FileStoreSecurityError(f"{label} cannot be a link or reparse point")
    if not stat.S_ISREG(path_stat.st_mode):
        raise FileStoreSecurityError(f"{label} must be a regular file")
    if getattr(path_stat, "st_nlink", 1) != 1:
        raise FileStoreSecurityError(f"{label} cannot be a hard link")


__all__ = [
    "DuplicateResourceError",
    "FileStoreError",
    "FileStoreIOError",
    "FileStoreSecurityError",
    "InvalidFileNameError",
    "InvalidResourceIdError",
    "ResourceNotFoundError",
    "StorageLimitError",
    "StoredFile",
    "TenantFileStore",
]
