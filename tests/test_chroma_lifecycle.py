from __future__ import annotations

import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag_system.config import Settings
from rag_system.domain import Chunk
from rag_system.retrieval import ChromaIndexRepository


class FakeChroma:
    collections: dict[tuple[str, str], set[str]] = {}
    add_calls = 0
    delete_calls = 0
    delete_error: Exception | None = None

    def __init__(
        self,
        *,
        collection_name,
        embedding_function,
        collection_metadata,
        persist_directory="",
    ) -> None:
        del embedding_function, collection_metadata
        self.key = (persist_directory, collection_name)
        self.collections.setdefault(self.key, set())

    def get(self, *, include):
        if include:
            raise AssertionError("manifest lookup should not load embeddings")
        return {"ids": sorted(self.collections[self.key])}

    def add_texts(self, *, texts, ids, metadatas):
        if not (len(texts) == len(ids) == len(metadatas)):
            raise AssertionError("invalid batch")
        type(self).add_calls += 1
        self.collections[self.key].update(ids)

    def delete_collection(self):
        type(self).delete_calls += 1
        if self.delete_error is not None:
            raise self.delete_error
        self.collections.pop(self.key, None)


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="guide.txt",
        text=f"content for {chunk_id}",
        chunk_index=0,
        start_char=0,
        end_char=20,
    )


class ChromaLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeChroma.collections.clear()
        FakeChroma.add_calls = 0
        FakeChroma.delete_calls = 0
        FakeChroma.delete_error = None

    @staticmethod
    def _repository(directory: str, *, persistent: bool) -> ChromaIndexRepository:
        settings = replace(
            Settings(),
            persist_data=persistent,
            storage_root=Path(directory),
        ).validate()
        repository = ChromaIndexRepository(settings)
        repository._embedding_function = object()
        return repository

    @staticmethod
    def _module():
        module = types.ModuleType("langchain_chroma")
        module.Chroma = FakeChroma
        return module

    def test_persistent_collection_is_reopened_without_reembedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(directory, persistent=True)
            chunks = (make_chunk("chunk-1"), make_chunk("chunk-2"))
            with patch.dict("sys.modules", {"langchain_chroma": self._module()}):
                first = repository.build("idx-stable", chunks)
                first.close()
                second = repository.build("idx-stable", chunks)

            self.assertEqual(FakeChroma.add_calls, 1)
            self.assertEqual(FakeChroma.delete_calls, 0)
            second.delete()
            self.assertEqual(FakeChroma.delete_calls, 1)

    def test_partial_persistent_collection_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(directory, persistent=True)
            persistence_path = str(Path(directory).resolve() / "vector")
            FakeChroma.collections[(persistence_path, "rag_idx_stable")] = {"orphan"}
            with patch.dict("sys.modules", {"langchain_chroma": self._module()}):
                repository.build("idx-stable", (make_chunk("chunk-1"),))

            self.assertEqual(FakeChroma.delete_calls, 1)
            self.assertEqual(FakeChroma.add_calls, 1)

    def test_ephemeral_close_deletes_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(directory, persistent=False)
            with patch.dict("sys.modules", {"langchain_chroma": self._module()}):
                index = repository.build("idx-ephemeral", (make_chunk("chunk-1"),))
            index.close()

            self.assertEqual(FakeChroma.delete_calls, 1)

    def test_explicit_delete_propagates_storage_failure_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(directory, persistent=True)
            with patch.dict("sys.modules", {"langchain_chroma": self._module()}):
                index = repository.build("idx-durable", (make_chunk("chunk-1"),))

            FakeChroma.delete_error = RuntimeError("storage unavailable")
            with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
                index.delete()

            FakeChroma.delete_error = None
            index.delete()
            self.assertEqual(FakeChroma.delete_calls, 2)


if __name__ == "__main__":
    unittest.main()
