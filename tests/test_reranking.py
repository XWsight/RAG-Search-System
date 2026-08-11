import math
import unittest

from rag_system.domain import Chunk, SearchHit
from rag_system.reranking import CrossEncoderReranker, RerankerError


def hit(identifier: str, score: float) -> SearchHit:
    text = f"text {identifier}"
    return SearchHit(Chunk(identifier, "doc", "source", text, 0, 0, len(text)), score)


class FakeScores:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def predict(self, pairs, **kwargs):
        self.calls += 1
        self.pairs = pairs
        self.kwargs = kwargs
        return self.values


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_cross_scores_can_change_order_and_model_is_loaded_once(self) -> None:
        model = FakeScores([-3.0, 4.0])
        factory_calls = []
        reranker = CrossEncoderReranker(
            "test-model",
            weight=0.9,
            model_factory=lambda name: factory_calls.append(name) or model,
        )
        original = [hit("a", 0.9), hit("b", 0.2)]
        first = reranker.rerank("query", original, top_k=2)
        second = reranker.rerank("query", original, top_k=1)
        self.assertEqual(first[0].chunk.chunk_id, "b")
        self.assertIn("rerank", first[0].reasons)
        self.assertEqual(len(second), 1)
        self.assertEqual(factory_calls, ["test-model"])

    def test_invalid_scores_fail_with_safe_error(self) -> None:
        for values in ([1.0], [math.nan, 0.0], ["bad", 0.0]):
            reranker = CrossEncoderReranker(
                "model", model_factory=lambda _, values=values: FakeScores(values)
            )
            with self.assertRaises(RerankerError):
                reranker.rerank("q", [hit("a", 0.5), hit("b", 0.5)], top_k=2)

    def test_boundaries_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            CrossEncoderReranker("")
        with self.assertRaises(ValueError):
            CrossEncoderReranker("m", weight=2.0)
        reranker = CrossEncoderReranker("m", model_factory=lambda _: FakeScores([]))
        self.assertEqual(reranker.rerank("", [], top_k=1), ())
        with self.assertRaises(ValueError):
            reranker.rerank("q", [], top_k=0)


if __name__ == "__main__":
    unittest.main()
