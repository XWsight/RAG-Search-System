from __future__ import annotations

import unittest

from rag_system.domain import Chunk
from scripts.benchmark_sparse import SparseBaselineRetriever


class SparseBaselineTests(unittest.TestCase):
    def test_baseline_uses_real_bm25_scores_and_common_hits(self) -> None:
        chunks = (
            Chunk("rag", "doc-1", "rag.md", "RAG retrieves external evidence", 0, 0, 31),
            Chunk("other", "doc-2", "other.md", "unrelated weather report", 0, 0, 24),
        )
        hits = SparseBaselineRetriever(chunks).search("RAG evidence", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "rag")
        self.assertGreater(hits[0].score, 0)
        self.assertLess(hits[0].score, 1)
        self.assertEqual(hits[0].reasons, ("sparse",))


if __name__ == "__main__":
    unittest.main()
