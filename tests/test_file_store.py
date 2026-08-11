from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from rag_system.file_store import (
    DuplicateResourceError,
    FileStoreIOError,
    FileStoreSecurityError,
    InvalidFileNameError,
    InvalidResourceIdError,
    ResourceNotFoundError,
    StorageLimitError,
    TenantFileStore,
)


def _tenant_directory(root: Path, tenant_id: str) -> Path:
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return root / f"tenant-{digest}"


class RecordingStream(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.requested_sizes: list[int] = []

    def read(self, size=-1):
        self.requested_sizes.append(size)
        if size < 0:
            raise AssertionError("file store must use bounded reads")
        return super().read(size)


class FailingStream:
    def __init__(self) -> None:
        self.read_count = 0

    def read(self, size: int) -> bytes:
        self.read_count += 1
        if self.read_count == 1:
            return b"partial"
        raise OSError("private storage detail")


class TenantFileStoreTests(unittest.TestCase):
    def test_save_resolve_hash_and_immutable_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(Path(directory, "uploads"), copy_buffer_size=3)
            stream = RecordingStream("向量资料".encode())
            stored = store.save("student-1", "resource-1", "资料.txt", stream)

            self.assertEqual(stored.display_name, "资料.txt")
            self.assertEqual(stored.size, len("向量资料".encode()))
            self.assertEqual(
                stored.sha256,
                hashlib.sha256("向量资料".encode()).hexdigest(),
            )
            self.assertNotIn("student-1", stored.relative_path)
            self.assertTrue(all(size == 3 for size in stream.requested_sizes))
            resolved = store.resolve("student-1", "resource-1")
            self.assertEqual(resolved.read_bytes(), "向量资料".encode())
            self.assertTrue(store.healthcheck())
            with self.assertRaises(FrozenInstanceError):
                stored.size = 0

    def test_same_resource_id_is_isolated_between_tenants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            first = store.save("tenant-a", "shared", "a.txt", b"alpha")
            second = store.save("tenant-b", "shared", "b.txt", b"beta")
            self.assertNotEqual(first.relative_path, second.relative_path)
            self.assertEqual(store.resolve("tenant-a", "shared").read_bytes(), b"alpha")
            self.assertEqual(store.resolve("tenant-b", "shared").read_bytes(), b"beta")

    def test_duplicate_resource_and_unsafe_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            store.save("tenant", "safe-id", "guide.txt", b"first")
            with self.assertRaises(DuplicateResourceError):
                store.save("tenant", "safe-id", "other.txt", b"second")

            for resource_id in ("../escape", "folder/name", ".", "CON", "has space"):
                with self.subTest(resource_id=resource_id), self.assertRaises(
                    InvalidResourceIdError
                ):
                    store.save("tenant", resource_id, "safe.txt", b"x")

            for name in (
                "../escape.txt",
                "folder/file.txt",
                "folder\\file.txt",
                "CON.txt",
                "trailing. ",
                ".upload-collision.tmp",
                "bad\x00name.txt",
                "bad\ud800name.txt",
            ):
                with self.subTest(name=name), self.assertRaises(InvalidFileNameError):
                    store.save("tenant", "new-id", name, b"x")

    def test_file_count_single_file_and_total_limits_roll_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            file_store = TenantFileStore(
                Path(directory, "single"),
                max_file_bytes=3,
                max_total_bytes=10,
                max_files_per_tenant=2,
            )
            with self.assertRaises(StorageLimitError):
                file_store.save("t", "large", "large.bin", b"1234")
            self.assertFalse(_tenant_directory(file_store.root, "t").joinpath("large").exists())

            total_store = TenantFileStore(
                Path(directory, "total"),
                max_file_bytes=5,
                max_total_bytes=5,
                max_files_per_tenant=2,
            )
            total_store.save("t", "one", "one.bin", b"123")
            with self.assertRaises(StorageLimitError):
                total_store.save("t", "two", "two.bin", b"456")
            self.assertFalse(_tenant_directory(total_store.root, "t").joinpath("two").exists())

            count_store = TenantFileStore(
                Path(directory, "count"),
                max_files_per_tenant=1,
            )
            count_store.save("t", "one", "one.bin", b"1")
            with self.assertRaises(StorageLimitError):
                count_store.save("t", "two", "two.bin", b"2")

    def test_stream_and_atomic_replace_failures_roll_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            tenant_directory = _tenant_directory(store.root, "t")
            with self.assertRaises(FileStoreIOError) as stream_error:
                store.save("t", "stream-failure", "data.bin", FailingStream())
            self.assertNotIn("private", str(stream_error.exception))
            self.assertFalse(tenant_directory.joinpath("stream-failure").exists())

            with patch("rag_system.file_store.os.replace", side_effect=OSError("disk detail")):
                with self.assertRaises(FileStoreIOError) as replace_error:
                    store.save("t", "replace-failure", "data.bin", b"data")
            self.assertNotIn("disk detail", str(replace_error.exception))
            self.assertFalse(tenant_directory.joinpath("replace-failure").exists())

    def test_resolve_rejects_unexpected_entries_and_non_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            store.save("t", "resource", "data.bin", b"data")
            resource_directory = _tenant_directory(store.root, "t") / "resource"
            resource_directory.joinpath("unexpected.bin").write_bytes(b"unexpected")
            with self.assertRaises(FileStoreSecurityError):
                store.resolve("t", "resource")
            with self.assertRaises(FileStoreSecurityError):
                store.delete("t", "resource")

            with self.assertRaises(ResourceNotFoundError):
                store.resolve("t", "missing")

    def test_delete_removes_only_the_exact_resource(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            store.save("t", "first", "first.bin", b"1")
            store.save("t", "second", "second.bin", b"2")
            tenant_directory = _tenant_directory(store.root, "t")

            self.assertTrue(store.delete("t", "first"))
            self.assertFalse(tenant_directory.joinpath("first").exists())
            self.assertEqual(store.resolve("t", "second").read_bytes(), b"2")
            self.assertTrue(tenant_directory.exists())
            self.assertFalse(store.delete("t", "first"))

    def test_delete_can_retry_after_file_was_removed_but_rmdir_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            store.save("t", "resource", "data.bin", b"data")
            resource_directory = _tenant_directory(store.root, "t") / "resource"
            original_rmdir = Path.rmdir

            def fail_resource_rmdir(path: Path) -> None:
                if path == resource_directory:
                    raise OSError("directory temporarily busy")
                original_rmdir(path)

            with patch.object(Path, "rmdir", fail_resource_rmdir):
                with self.assertRaises(FileStoreIOError):
                    store.delete("t", "resource")

            self.assertTrue(resource_directory.is_dir())
            self.assertEqual(tuple(resource_directory.iterdir()), ())
            self.assertTrue(store.delete("t", "resource"))
            self.assertFalse(resource_directory.exists())

    def test_symlink_is_never_resolved_or_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(Path(directory, "root"))
            outside = Path(directory, "outside.bin")
            outside.write_bytes(b"outside")
            tenant_directory = _tenant_directory(store.root, "t")
            tenant_directory.mkdir()
            resource_directory = tenant_directory / "linked"
            resource_directory.mkdir()
            link = resource_directory / "link.bin"
            try:
                os.symlink(outside, link)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            with self.assertRaises(FileStoreSecurityError):
                store.resolve("t", "linked")
            with self.assertRaises(FileStoreSecurityError):
                store.delete("t", "linked")
            self.assertEqual(outside.read_bytes(), b"outside")
            self.assertTrue(link.exists())

    def test_reparse_attribute_is_rejected_without_following_the_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TenantFileStore(directory)
            store.save("t", "resource", "data.bin", b"data")
            stored_path = store.resolve("t", "resource")
            original_lstat = os.lstat

            class ReparseStat:
                def __init__(self, original) -> None:
                    self._original = original
                    self.st_file_attributes = (
                        getattr(original, "st_file_attributes", 0) | 0x400
                    )

                def __getattr__(self, name):
                    return getattr(self._original, name)

            def marked_lstat(path):
                result = original_lstat(path)
                if Path(path) == stored_path:
                    return ReparseStat(result)
                return result

            with patch("rag_system.file_store.os.lstat", side_effect=marked_lstat):
                with self.assertRaises(FileStoreSecurityError):
                    store.resolve("t", "resource")
                with self.assertRaises(FileStoreSecurityError):
                    store.delete("t", "resource")

            self.assertEqual(stored_path.read_bytes(), b"data")

    def test_root_link_and_invalid_configuration_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            real_root = Path(directory, "real")
            real_root.mkdir()
            root_link = Path(directory, "root-link")
            try:
                os.symlink(real_root, root_link, target_is_directory=True)
            except OSError:
                root_link = None
            if root_link is not None:
                with self.assertRaises(FileStoreSecurityError):
                    TenantFileStore(root_link)

            for options in (
                {"max_file_bytes": 0},
                {"max_total_bytes": 0},
                {"max_files_per_tenant": 0},
                {"copy_buffer_size": 0},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    TenantFileStore(Path(directory, uuid_name(options)), **options)


def uuid_name(value: object) -> str:
    return "store-" + hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:8]


if __name__ == "__main__":
    unittest.main()
