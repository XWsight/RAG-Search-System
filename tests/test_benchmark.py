import json
import tempfile
import unittest
from pathlib import Path

from rag_system.benchmark import (
    RetrievalBenchmarkCase,
    load_retrieval_benchmark,
    run_retrieval_benchmark,
)
from rag_system.config import Settings
from rag_system.domain import Chunk, Route, SearchHit
from rag_system.evaluation import DatasetValidationError
from rag_system.retrieval import RoutingPolicy


def hit(source: str, score: float = 0.9) -> SearchHit:
    chunk = Chunk(source, source, source, "内容", 0, 0, 2)
    return SearchHit(chunk, score, reasons=("dense", "sparse"))


class FakeRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query: str, *, top_k: int):
        return tuple(self.mapping.get(query, ()))[:top_k]


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_loads_strict_ground_truth_without_predictions(self) -> None:
        payload = {
            "case_id": "one",
            "question": "问题",
            "relevance": {"rag.md": 3},
            "expected_route": "local",
            "allow_web": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "cases.jsonl")
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cases = load_retrieval_benchmark(path)
            self.assertEqual(cases[0].relevance, (("rag.md", 3),))

            payload["prediction"] = "forbidden"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                load_retrieval_benchmark(path)

    def test_real_retriever_outputs_feed_metrics_and_routes(self) -> None:
        cases = (
            RetrievalBenchmarkCase("a", "rag", (("rag.md", 3),), Route.LOCAL),
            RetrievalBenchmarkCase("b", "missing", (), Route.REFUSED),
        )
        retriever = FakeRetriever({"rag": [hit("rag.md"), hit("rag.md", 0.8)]})
        run = run_retrieval_benchmark(
            cases,
            retriever,
            RoutingPolicy(Settings()),
            top_k=3,
        )
        self.assertEqual(run.report.metrics.recall_at_k, 1.0)
        self.assertEqual(run.report.metrics.route_accuracy, 1.0)
        self.assertEqual(run.predictions[0].retrieved_sources, ("rag.md",))
        self.assertIn("逐题结果", run.to_markdown())
        self.assertEqual(json.loads(run.to_json())["report"]["case_count"], 2)

    def test_empty_cases_and_invalid_top_k_are_rejected(self) -> None:
        routing = RoutingPolicy(Settings())
        with self.assertRaises(ValueError):
            run_retrieval_benchmark([], FakeRetriever({}), routing)
        with self.assertRaises(ValueError):
            run_retrieval_benchmark(
                [RetrievalBenchmarkCase("a", "q", (), Route.REFUSED)],
                FakeRetriever({}),
                routing,
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
