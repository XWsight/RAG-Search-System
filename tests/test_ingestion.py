from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag_system.config import Settings
from rag_system.ingestion import AdaptiveTextSplitter, DocumentIngestor
from rag_system.security import DocumentValidationError


class IngestionTests(unittest.TestCase):
    def test_multi_document_ingestion_is_stable_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.md")
            duplicate = Path(directory, "copy.txt")
            second = Path(directory, "second.txt")
            first.write_text("# RAG\n" + "检索增强生成。" * 100, encoding="utf-8")
            duplicate.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
            second.write_text("向量数据库保存向量。" * 100, encoding="utf-8")

            settings = replace(Settings(), chunk_size=200, chunk_overlap=40).validate()
            ingestor = DocumentIngestor(settings)
            result = ingestor.ingest([first, duplicate, second])
            repeated = ingestor.ingest([first, duplicate, second])

            self.assertEqual(len(result.documents), 2)
            self.assertEqual(result.duplicate_count, 1)
            self.assertEqual(result.index_id, repeated.index_id)
            self.assertEqual([chunk.chunk_id for chunk in result.chunks], [chunk.chunk_id for chunk in repeated.chunks])
            self.assertEqual(result.chunks[0].heading, "RAG")

    def test_chunks_have_overlap_and_valid_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "long.txt")
            text = "第一部分。" * 90
            path.write_text(text, encoding="utf-8")
            settings = replace(Settings(), chunk_size=180, chunk_overlap=30).validate()
            result = DocumentIngestor(settings).ingest([path])

            self.assertGreater(len(result.chunks), 1)
            for chunk in result.chunks:
                self.assertEqual(chunk.text, text[chunk.start_char : chunk.end_char])
            self.assertLess(result.chunks[1].start_char, result.chunks[0].end_char)

    def test_limits_are_enforced_before_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "large.txt")
            path.write_text("x" * 20, encoding="utf-8")
            settings = replace(Settings(), max_file_bytes=10, max_total_bytes=10)
            with self.assertRaises(DocumentValidationError):
                DocumentIngestor(settings).ingest([path])

    def test_namespace_is_part_of_the_stable_index_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "tenant-guide.txt")
            path.write_text("tenant isolated retrieval " * 20, encoding="utf-8")
            ingestor = DocumentIngestor(Settings())

            tenant_a = ingestor.ingest([path], namespace="tenant-a:kb-1")
            tenant_a_again = ingestor.ingest([path], namespace="tenant-a:kb-1")
            tenant_b = ingestor.ingest([path], namespace="tenant-b:kb-1")

            self.assertEqual(tenant_a.index_id, tenant_a_again.index_id)
            self.assertNotEqual(tenant_a.index_id, tenant_b.index_id)

    def test_namespace_rejects_control_characters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "guide.txt")
            path.write_text("valid content " * 20, encoding="utf-8")
            with self.assertRaises(ValueError):
                DocumentIngestor(Settings()).ingest([path], namespace="tenant\nother")

    def test_splitter_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTextSplitter(chunk_size=50, chunk_overlap=0)
        with self.assertRaises(ValueError):
            AdaptiveTextSplitter(chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
