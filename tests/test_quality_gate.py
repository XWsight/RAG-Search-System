from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rag_system.benchmark import RetrievalBenchmarkCase, run_retrieval_benchmark
from rag_system.config import Settings
from rag_system.domain import Chunk, Route, SearchHit
from rag_system.evaluation import DatasetValidationError
from rag_system.quality_gate import (
    evaluate_quality_gate,
    load_quality_gate,
    quality_gate_from_mapping,
)
from rag_system.retrieval import RoutingPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS = tuple(
    PROJECT_ROOT / "evals" / "corpus" / name
    for name in ("rag.md", "retrieval.md", "safety.md", "storage.md")
)


class FakeRetriever:
    def search(self, query: str, *, top_k: int):
        del query, top_k
        chunk = Chunk("chunk", "doc", "rag.md", "RAG evidence", 0, 0, 12)
        return (SearchHit(chunk, 0.9, reasons=("dense", "sparse")),)


def _run():
    return run_retrieval_benchmark(
        (RetrievalBenchmarkCase("rag", "RAG?", (("rag.md", 3),), Route.LOCAL),),
        FakeRetriever(),
        RoutingPolicy(Settings()),
        clock=iter((1.0, 1.01)).__next__,
    )


def _gate_payload(run) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_digest": run.report.dataset_digest,
        "top_k": run.report.top_k,
        "minimum_metrics": {
            "recall_at_k": 1.0,
            "route_accuracy": 1.0,
        },
        "maximum_latency_ms": {"p95_ms": 20.0},
    }


class QualityGateTests(unittest.TestCase):
    def test_matching_run_passes_and_renders_machine_readable_result(self) -> None:
        run = _run()
        result = evaluate_quality_gate(run, quality_gate_from_mapping(_gate_payload(run)))

        self.assertTrue(result.passed)
        self.assertEqual(result.violations, ())
        self.assertTrue(json.loads(result.to_json())["passed"])
        self.assertIn("通过", result.to_markdown())

    def test_metric_latency_and_compatibility_regressions_are_explained(self) -> None:
        run = _run()
        payload = _gate_payload(run)
        payload["dataset_digest"] = "0000000000000000"
        payload["top_k"] = 9
        payload["minimum_metrics"] = {"mrr_at_k": 1.1}
        with self.assertRaises(DatasetValidationError):
            quality_gate_from_mapping(payload)

        payload["minimum_metrics"] = {"route_accuracy": 1.0}
        payload["maximum_latency_ms"] = {"p95_ms": 5.0}
        result = evaluate_quality_gate(run, quality_gate_from_mapping(payload))

        self.assertFalse(result.passed)
        self.assertEqual(
            {item.field for item in result.violations},
            {"dataset_digest", "top_k", "p95_ms"},
        )
        self.assertIn("maximum milliseconds", result.to_json())

        citation_payload = _gate_payload(run)
        citation_payload["minimum_metrics"] = {"citation_validity": 0.0}
        citation_result = evaluate_quality_gate(
            run,
            quality_gate_from_mapping(citation_payload),
        )
        self.assertFalse(citation_result.passed)
        self.assertEqual(citation_result.violations[0].actual, "N/A")

    def test_schema_rejects_unknown_values_wrong_types_and_duplicate_keys(self) -> None:
        run = _run()
        valid = _gate_payload(run)
        for field, value in (
            ("schema_version", 1.0),
            ("dataset_digest", "not-a-digest"),
            ("top_k", True),
            ("minimum_metrics", {}),
            ("maximum_latency_ms", {"p100_ms": 1}),
        ):
            payload = dict(valid)
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(DatasetValidationError):
                quality_gate_from_mapping(payload)

        payload = dict(valid)
        payload["unknown"] = True
        with self.assertRaises(DatasetValidationError):
            quality_gate_from_mapping(payload)
        with self.assertRaises(DatasetValidationError):
            quality_gate_from_mapping({1: "invalid"})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "duplicate.json")
            path.write_text(
                '{"schema_version":1,"schema_version":1,"dataset_digest":"0000000000000000",'
                '"top_k":5,"minimum_metrics":{"recall_at_k":1},"maximum_latency_ms":{}}',
                encoding="utf-8",
            )
            with self.assertRaises(DatasetValidationError):
                load_quality_gate(path)

    def test_sparse_command_enforces_the_repository_gate_and_writes_diagnostics(self) -> None:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_sparse.py"),
            str(PROJECT_ROOT / "evals" / "retrieval_cases.jsonl"),
            *(str(path) for path in CORPUS),
            "--top-k",
            "5",
            "--quality-gate",
            str(PROJECT_ROOT / "evals" / "gates" / "bm25-smoke.json"),
        ]
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("RAG_")
        }
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("失败诊断", completed.stdout)

        with tempfile.TemporaryDirectory() as directory:
            gate_path = Path(directory, "failing-gate.json")
            report_path = Path(directory, "run.json")
            gate = json.loads(
                (PROJECT_ROOT / "evals" / "gates" / "bm25-smoke.json").read_text(
                    encoding="utf-8"
                )
            )
            gate["minimum_metrics"]["route_accuracy"] = 0.8
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            failed = subprocess.run(
                [*command[:-1], str(gate_path), "--json-output", str(report_path)],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(failed.returncode, 3, failed.stderr)
            self.assertIn("route_accuracy", failed.stderr)
            self.assertIn("失败诊断", failed.stderr)
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
