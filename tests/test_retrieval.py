from __future__ import annotations

import unittest
from dataclasses import replace

from rag_system.config import Settings
from rag_system.domain import Chunk, IndexRef, Route, SearchHit
from rag_system.retrieval import HybridRetriever, RoutingPolicy


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
        self.assertGreater(hits[0].score, hits[-1].score)

    def test_routing_respects_confidence_and_privacy_switch(self) -> None:
        policy = RoutingPolicy(replace(self.settings, local_confidence_threshold=0.6))
        high_hit = SearchHit(self.rag, 0.85, reasons=("dense", "sparse"))
        low_hit = SearchHit(self.weather, 0.1, reasons=("dense",))
        self.assertEqual(policy.decide([high_hit], allow_web=False).route, Route.LOCAL)
        self.assertEqual(policy.decide([low_hit], allow_web=False).route, Route.REFUSED)
        self.assertEqual(policy.decide([low_hit], allow_web=True).route, Route.WEB)

    def test_empty_query_returns_no_hits(self) -> None:
        self.assertEqual(self.retriever.search("  ", top_k=3), ())


if __name__ == "__main__":
    unittest.main()
