from __future__ import annotations

import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from rag_system.config import Settings
from rag_system.domain import IndexRef, SearchHit
from rag_system.index_manager import IndexManager


class FakeVectorIndex:
    def __init__(self, index_ref: IndexRef) -> None:
        self.index_ref = index_ref
        self.closed = False
        self.deleted = False

    def search(self, query: str, *, top_k: int) -> tuple[SearchHit, ...]:
        return ()

    def close(self) -> None:
        self.closed = True

    def delete(self) -> None:
        self.closed = True
        self.deleted = True


class FakeRepository:
    def __init__(self) -> None:
        self.build_count = 0
        self.indexes: list[FakeVectorIndex] = []
        self.deleted_ids: list[str] = []

    def build(self, index_id, chunks):
        self.build_count += 1
        index = FakeVectorIndex(IndexRef(index_id, len({c.document_id for c in chunks}), len(chunks), 0.0))
        self.indexes.append(index)
        return index

    def delete(self, index_id):
        self.deleted_ids.append(index_id)
        return True

    def healthcheck(self) -> bool:
        return True


class IndexManagerTests(unittest.TestCase):
    def test_healthcheck_reflects_manager_lifecycle(self) -> None:
        repository = FakeRepository()
        manager = IndexManager(Settings(), repository)

        self.assertTrue(manager.healthcheck())
        manager.close()
        self.assertFalse(manager.healthcheck())

    def test_identical_documents_reuse_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory, "guide.txt")
            document.write_text("RAG 使用外部知识库。" * 40, encoding="utf-8")
            repository = FakeRepository()
            manager = IndexManager(Settings(), repository)
            first = manager.build([document])
            second = manager.build([document])
            self.assertEqual(first.index_id, second.index_id)
            self.assertEqual(repository.build_count, 1)
            self.assertEqual(manager.stats()["active_indexes"], 1)

    def test_ttl_cleanup_and_capacity_close_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clock_value = [0.0]
            settings = replace(Settings(), max_sessions=1, session_ttl_seconds=60)
            repository = FakeRepository()
            manager = IndexManager(settings, repository, clock=lambda: clock_value[0])
            first_path = Path(directory, "first.txt")
            second_path = Path(directory, "second.txt")
            first_path.write_text("第一份知识。" * 40, encoding="utf-8")
            second_path.write_text("第二份知识。" * 40, encoding="utf-8")
            first = manager.build([first_path])
            manager.build([second_path])
            self.assertTrue(repository.indexes[0].closed)
            with self.assertRaises(KeyError):
                manager.get(first.index_id)

            clock_value[0] = 100.0
            with self.assertRaises(KeyError):
                manager.get("missing")
            self.assertTrue(repository.indexes[1].closed)

    def test_namespaces_do_not_share_cached_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory, "guide.txt")
            document.write_text("tenant scoped knowledge " * 20, encoding="utf-8")
            repository = FakeRepository()
            manager = IndexManager(Settings(), repository)

            first = manager.build([document], namespace="tenant-a:kb")
            second = manager.build([document], namespace="tenant-b:kb")

            self.assertNotEqual(first.index_id, second.index_id)
            self.assertEqual(repository.build_count, 2)

    def test_explicit_delete_is_distinct_from_cache_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory, "guide.txt")
            document.write_text("deletion lifecycle " * 20, encoding="utf-8")
            repository = FakeRepository()
            manager = IndexManager(Settings(), repository)
            index_ref = manager.build([document])

            self.assertTrue(manager.delete(index_ref.index_id))
            self.assertTrue(repository.indexes[0].deleted)
            self.assertTrue(manager.delete("idx-not-loaded"))
            self.assertEqual(repository.deleted_ids, ["idx-not-loaded"])

    def test_delete_waits_for_active_retrieval_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory, "guide.txt")
            document.write_text("leased retrieval " * 20, encoding="utf-8")
            repository = FakeRepository()
            manager = IndexManager(Settings(), repository)
            index_ref = manager.build([document])
            delete_started = threading.Event()
            delete_finished = threading.Event()

            def delete_index() -> None:
                delete_started.set()
                manager.delete(index_ref.index_id)
                delete_finished.set()

            with manager.lease(index_ref.index_id):
                worker = threading.Thread(target=delete_index)
                worker.start()
                self.assertTrue(delete_started.wait(1))
                self.assertFalse(delete_finished.wait(0.05))
                self.assertFalse(repository.indexes[0].deleted)

            worker.join(1)
            self.assertTrue(delete_finished.is_set())
            self.assertTrue(repository.indexes[0].deleted)

    def test_capacity_eviction_defers_close_until_lease_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), max_sessions=1)
            repository = FakeRepository()
            manager = IndexManager(settings, repository)
            first_path = Path(directory, "first.txt")
            second_path = Path(directory, "second.txt")
            first_path.write_text("first leased index " * 20, encoding="utf-8")
            second_path.write_text("second index " * 20, encoding="utf-8")
            first = manager.build([first_path])

            with manager.lease(first.index_id):
                manager.build([second_path])
                self.assertFalse(repository.indexes[0].closed)

            self.assertTrue(repository.indexes[0].closed)

    def test_delete_upgrades_an_evicted_active_lease_to_durable_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), max_sessions=1)
            repository = FakeRepository()
            manager = IndexManager(settings, repository)
            first_path = Path(directory, "first.txt")
            second_path = Path(directory, "second.txt")
            first_path.write_text("first leased index " * 20, encoding="utf-8")
            second_path.write_text("second index " * 20, encoding="utf-8")
            first = manager.build([first_path])
            delete_finished = threading.Event()

            with manager.lease(first.index_id):
                manager.build([second_path])
                worker = threading.Thread(
                    target=lambda: (
                        manager.delete(first.index_id),
                        delete_finished.set(),
                    )
                )
                worker.start()
                self.assertFalse(delete_finished.wait(0.05))
                self.assertFalse(repository.indexes[0].deleted)

            worker.join(1)
            self.assertTrue(delete_finished.is_set())
            self.assertTrue(repository.indexes[0].deleted)

    def test_rebuild_reactivates_an_evicted_lease_without_duplicate_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(Settings(), max_sessions=1)
            repository = FakeRepository()
            manager = IndexManager(settings, repository)
            first_path = Path(directory, "first.txt")
            second_path = Path(directory, "second.txt")
            first_path.write_text("first leased index " * 20, encoding="utf-8")
            second_path.write_text("second index " * 20, encoding="utf-8")
            first = manager.build([first_path])

            with manager.lease(first.index_id):
                manager.build([second_path])
                rebuilt = manager.build([first_path])
                self.assertEqual(rebuilt.index_id, first.index_id)
                self.assertEqual(repository.build_count, 2)

            self.assertFalse(repository.indexes[0].closed)
            self.assertEqual(manager.stats()["active_indexes"], 1)


if __name__ == "__main__":
    unittest.main()
