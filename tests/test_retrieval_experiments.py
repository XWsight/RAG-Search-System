from __future__ import annotations

import json
import unittest

from rag_system.benchmark import RetrievalBenchmarkCase
from rag_system.config import Settings
from rag_system.domain import Chunk, Route, SearchHit
from rag_system.retrieval_experiments import (
    RetrievalNonDeterminismError,
    run_retrieval_ablation,
)
from rag_system.routing import RoutingPolicy


def hit(source: str) -> SearchHit:
    chunk = Chunk(source, source, source, "检索资料", 0, 0, 4)
    return SearchHit(
        chunk,
        0.9,
        reasons=("dense", "sparse"),
        lexical_score=0.5,
    )


class StaticRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    def search(self, query: str, *, top_k: int):
        self.calls += 1
        return tuple(self.mapping.get(query, ()))[:top_k]


class AlternatingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, *, top_k: int):
        self.calls += 1
        return (hit("a.md"),) if self.calls % 2 else ()


class RetrievalExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = (
            RetrievalBenchmarkCase("a", "A", (("a.md", 3),), Route.LOCAL),
            RetrievalBenchmarkCase("b", "B", (("b.md", 3),), Route.LOCAL),
        )
        self.routing = RoutingPolicy(Settings())

    def test_report_compares_metrics_latency_and_case_level_changes(self) -> None:
        baseline = StaticRetriever({"A": (hit("a.md"),)})
        candidate = StaticRetriever({"A": (hit("a.md"),), "B": (hit("b.md"),)})
        report = run_retrieval_ablation(
            self.cases,
            {"baseline": baseline, "candidate": candidate},
            self.routing,
            baseline="baseline",
            repetitions=3,
            suite_digest="suite-1",
            split="validation",
            configuration={"embedding_model": "test", "top_k": 5},
            variant_configurations={
                "baseline": {"profile": "sparse"},
                "candidate": {"profile": "fusion"},
            },
            index_build_ms=12.5,
        )

        self.assertEqual(baseline.calls, 6)
        self.assertEqual(candidate.calls, 6)
        self.assertEqual(report.comparisons[0].gained_case_ids, ("b",))
        self.assertEqual(report.comparisons[0].lost_case_ids, ())
        self.assertGreater(report.comparisons[0].recall_delta, 0)
        payload = json.loads(report.to_json())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["suite_digest"], "suite-1")
        self.assertEqual(payload["variants"][0]["repetitions"], 3)
        self.assertEqual(payload["variants"][0]["configuration"]["profile"], "sparse")
        self.assertEqual(payload["variants"][0]["benchmark"]["latency"]["case_count"], 6)
        self.assertIn("Delta against baseline", report.to_markdown())
        self.assertIn("candidate", report.to_markdown())

    def test_repeated_runs_fail_if_predictions_are_not_stable(self) -> None:
        with self.assertRaises(RetrievalNonDeterminismError):
            run_retrieval_ablation(
                self.cases[:1],
                {
                    "unstable": AlternatingRetriever(),
                    "stable": StaticRetriever({"A": (hit("a.md"),)}),
                },
                self.routing,
                baseline="stable",
                repetitions=2,
            )

    def test_invalid_experiment_contract_is_rejected(self) -> None:
        stable = StaticRetriever({})
        with self.assertRaisesRegex(ValueError, "at least two"):
            run_retrieval_ablation(
                self.cases,
                {"one": stable},
                self.routing,
                baseline="one",
            )
        with self.assertRaisesRegex(ValueError, "baseline"):
            run_retrieval_ablation(
                self.cases,
                {"one": stable, "two": stable},
                self.routing,
                baseline="missing",
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            run_retrieval_ablation(
                self.cases,
                {"one": stable, "two": stable},
                self.routing,
                baseline="one",
                configuration={"weight": float("nan")},
            )
        with self.assertRaisesRegex(ValueError, "configuration names"):
            run_retrieval_ablation(
                self.cases,
                {"one": stable, "two": stable},
                self.routing,
                baseline="one",
                variant_configurations={"missing": {}},
            )
        with self.assertRaisesRegex(ValueError, "sensitive"):
            run_retrieval_ablation(
                self.cases,
                {"one": stable, "two": stable},
                self.routing,
                baseline="one",
                configuration={"api_key": "never-report-this"},
            )


if __name__ == "__main__":
    unittest.main()
