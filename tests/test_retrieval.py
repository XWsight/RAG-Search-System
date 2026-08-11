from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from rag_system.config import Settings
from rag_system.domain import Chunk, IndexRef, Route, SearchHit
from rag_system.retrieval import ChromaIndexRepository, HybridRetriever, RoutingPolicy


def make_chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, document_id, f"{document_id}.txt", text, 0, 0, len(text))


class FakeVectorIndex:
    def __init__(self, chunks: list[Chunk], scores: dict[str, float]) -> None:
        self.index_ref = IndexRef("idx_test", 2, len(chunks), 0.0)
        self.chunks = chunks
        self.scores = scores
        self.closed = False

    def search(self, query: str, *, top_k: int):
        ordered = sorted(self.chunks, key=lambda chunk: -self.scores.get(chunk.chunk_id, 0.0))
        return tuple(
            SearchHit(chunk, self.scores.get(chunk.chunk_id, 0.0), dense_rank=rank, reasons=("dense",))
            for rank, chunk in enumerate(ordered[:top_k], start=1)
        )

    def close(self) -> None:
        self.closed = True


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rag = make_chunk("rag", "guide", "RAG 使用检索到的外部资料生成答案。")
        self.vector = make_chunk("vector", "database", "ChromaDB 是保存向量的向量数据库。")
        self.weather = make_chunk("weather", "misc", "今天阳光明媚，适合出门。")
        settings = replace(
            Settings(), dense_candidates=12, sparse_candidates=12, fused_candidates=6, final_evidence_count=3
        ).validate()
        self.settings = settings
        self.retriever = HybridRetriever(
            FakeVectorIndex([self.rag, self.vector, self.weather], {"rag": 0.75, "vector": 0.55, "weather": 0.2}),
            [self.rag, self.vector, self.weather],
            settings,
        )

    def test_hybrid_retrieval_rewards_dense_and_exact_keyword_agreement(self) -> None:
        hits = self.retriever.search("什么是 RAG", top_k=3)
        self.assertEqual(hits[0].chunk.chunk_id, "rag")
        self.assertEqual(set(hits[0].reasons), {"dense", "sparse"})
        self.assertIsNotNone(hits[0].lexical_score)
        self.assertGreater(hits[0].lexical_score or 0.0, 0.0)
        self.assertGreater(hits[0].score, hits[-1].score)

    def test_routing_respects_confidence_and_privacy_switch(self) -> None:
        policy = RoutingPolicy(
            replace(
                self.settings,
                local_confidence_threshold=0.6,
                hybrid_confidence_ratio=0.95,
            )
        )
        high_hit = SearchHit(
            self.rag,
            0.85,
            reasons=("dense", "sparse"),
            lexical_score=0.3,
        )
        partial_hit = SearchHit(
            self.rag,
            0.46,
            reasons=("dense", "sparse"),
            lexical_score=0.3,
        )
        generic_agreement = SearchHit(
            self.weather,
            0.55,
            reasons=("dense", "sparse"),
            lexical_score=0.03,
        )
        low_hit = SearchHit(self.weather, 0.1, reasons=("dense",))
        self.assertEqual(policy.decide([high_hit], allow_web=False).route, Route.LOCAL)
        self.assertEqual(policy.decide([low_hit], allow_web=False).route, Route.REFUSED)
        self.assertEqual(policy.decide([partial_hit], allow_web=True).route, Route.HYBRID)
        self.assertEqual(policy.decide([generic_agreement], allow_web=True).route, Route.WEB)
        self.assertEqual(policy.decide([low_hit], allow_web=True).route, Route.WEB)

    def test_routing_assessment_exposes_bounded_content_free_signals(self) -> None:
        policy = RoutingPolicy(self.settings)
        hits = (
            SearchHit(
                self.rag,
                0.8,
                reasons=("dense", "sparse"),
                lexical_score=0.45,
            ),
            SearchHit(self.vector, 0.6, reasons=("dense",)),
        )

        assessment = policy.assess(hits, allow_web=False)
        payload = assessment.signal.to_dict()

        self.assertEqual(assessment.decision.route, Route.LOCAL)
        self.assertEqual(payload["top_score"], 0.8)
        self.assertEqual(payload["second_score"], 0.6)
        self.assertEqual(payload["margin"], 0.2)
        self.assertTrue(payload["ranker_agreement"])
        self.assertEqual(payload["lexical_support"], 1.0)
        self.assertAlmostEqual(payload["confidence"], assessment.decision.confidence)
        self.assertNotIn("text", payload)

        empty = policy.assess((), allow_web=False)
        self.assertEqual(empty.signal.confidence, 0.0)
        self.assertEqual(empty.decision.route, Route.REFUSED)

    def test_empty_query_returns_no_hits(self) -> None:
        self.assertEqual(self.retriever.search("  ", top_k=3), ())

    def test_persistent_repository_healthcheck_validates_storage_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(Settings(), persist_data=True, storage_root=root).validate()
            repository = ChromaIndexRepository(settings)

            self.assertTrue(repository.healthcheck())
            vector_directory = root / "vector"
            vector_directory.rmdir()

            self.assertFalse(repository.healthcheck())
            self.assertFalse(vector_directory.exists())

            vector_directory.write_text("not a directory", encoding="utf-8")

            self.assertFalse(repository.healthcheck())


if __name__ == "__main__":
    unittest.main()
