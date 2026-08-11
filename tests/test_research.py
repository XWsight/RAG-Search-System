import unittest

from rag_system.domain import Chunk, SearchHit
from rag_system.research import fuse_query_hits, normalize_query_plan


def hit(identifier: str, document: str, score: float) -> SearchHit:
    chunk = Chunk(identifier, document, f"{document}.md", identifier, 0, 0, len(identifier))
    return SearchHit(chunk, score, reasons=("dense",))


class ResearchRetrievalTests(unittest.TestCase):
    def test_plan_keeps_original_deduplicates_and_bounds_queries(self) -> None:
        plan = normalize_query_plan(
            "原始问题",
            [" 原始问题 ", "子问题一", "子问题一", "x" * 100],
            max_queries=3,
            max_characters=10,
        )
        self.assertEqual(plan, ("原始问题", "子问题一", "xxxxxxxxxx"))

    def test_multi_query_agreement_improves_rank_and_diversifies_sources(self) -> None:
        shared = hit("shared", "doc-a", 0.6)
        rankings = {
            "q1": [hit("only-a", "doc-a", 0.9), shared, hit("b", "doc-b", 0.4)],
            "q2": [shared, hit("c", "doc-c", 0.8), hit("extra-a", "doc-a", 0.7)],
        }
        fused = fuse_query_hits(rankings, top_k=3)
        self.assertEqual(fused[0].chunk.chunk_id, "shared")
        self.assertIn("multi_query", fused[0].reasons)
        self.assertLessEqual(
            sum(item.chunk.document_id == "doc-a" for item in fused),
            2,
        )

    def test_empty_and_invalid_boundaries(self) -> None:
        self.assertEqual(fuse_query_hits({}, top_k=2), ())
        with self.assertRaises(ValueError):
            fuse_query_hits({}, top_k=0)
        with self.assertRaises(ValueError):
            normalize_query_plan("", [], max_queries=2)


if __name__ == "__main__":
    unittest.main()
