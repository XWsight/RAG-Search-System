import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag_system.benchmark import run_retrieval_benchmark
from rag_system.benchmark_suite import load_retrieval_suite, validate_suite_contract
from rag_system.config import Settings
from rag_system.domain import Chunk, SearchHit
from rag_system.evaluation import DatasetValidationError
from rag_system.evaluation_suite import EvaluationSuiteError
from rag_system.retrieval import RoutingPolicy
from rag_system.retrieval_analysis import build_retrieval_suite_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "suite_id": "temporary_suite",
        "description": "A sufficiently descriptive temporary retrieval benchmark suite.",
        "corpus_root": "corpus",
        "requirements": {
            "minimum_cases": 2,
            "minimum_families": 1,
            "minimum_categories": 1,
            "minimum_cases_per_split": {
                "development": 2,
                "validation": 1,
                "test": 1,
            },
            "minimum_cases_per_route": {"local": 2, "refused": 1, "web": 1},
            "minimum_cases_per_difficulty": {"easy": 2, "medium": 1, "hard": 1},
        },
        "families": [
            {
                "family_id": "local_family",
                "category": "retrieval",
                "split": "development",
                "difficulty": "easy",
                "expected_route": "local",
                "allow_web": False,
                "relevance": {"source.md": 3},
                "questions": ["第一个本地检索问题是什么？", "第二个本地检索问题是什么？"],
            },
            {
                "family_id": "validation_refusal",
                "category": "boundary",
                "split": "validation",
                "difficulty": "medium",
                "expected_route": "refused",
                "allow_web": False,
                "relevance": {},
                "questions": ["这个验证问题应当拒答吗？", "另一个验证拒答问题呢？"],
            },
            {
                "family_id": "test_web",
                "category": "freshness",
                "split": "test",
                "difficulty": "hard",
                "expected_route": "web",
                "allow_web": True,
                "relevance": {},
                "questions": ["联网查找第一个实时问题。", "联网查找第二个实时问题。"],
            },
        ],
    }


class RetrievalBenchmarkSuiteTests(unittest.TestCase):
    def test_repository_suite_has_declared_coverage(self) -> None:
        suite = load_retrieval_suite(PROJECT_ROOT / "evals" / "retrieval_suite.json")

        self.assertEqual(len(suite.cases), 200)
        self.assertEqual(len(suite.families), 50)
        self.assertEqual(len(suite.documents), 10)
        self.assertEqual(
            suite.summary()["cases_by_split"],
            {"development": 80, "test": 60, "validation": 60},
        )
        self.assertEqual(
            suite.summary()["cases_by_route"], {"local": 160, "refused": 20, "web": 20}
        )
        self.assertEqual(len(suite.cases_for_split("validation")), 60)
        self.assertEqual(len(suite.documents_for_split("validation")), 3)
        self.assertIn("Semantic families", suite.to_markdown())
        validate_suite_contract(
            suite, PROJECT_ROOT / "evals" / "gates" / "retrieval-suite.json"
        )

        with self.assertRaisesRegex(DatasetValidationError, "must be one of"):
            suite.cases_for_split("unknown")

    def test_frozen_contract_detects_manifest_or_corpus_drift(self) -> None:
        suite = load_retrieval_suite(PROJECT_ROOT / "evals" / "retrieval_suite.json")
        contract = {
            key: value
            for key, value in suite.summary().items()
            if key not in {"suite_id"}
        }
        contract["suite_digest"] = "0" * 16
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "suite_digest"):
                validate_suite_contract(suite, path)

    def test_expands_family_without_putting_predictions_in_ground_truth(self) -> None:
        with self._suite(_manifest()) as path:
            suite = load_retrieval_suite(path)

        self.assertEqual(len(suite.cases), 6)
        first = suite.cases[0]
        self.assertEqual(first.case_id, "local_family__01")
        self.assertEqual(first.relevance, (("source.md", 3),))
        self.assertEqual(first.to_dict().keys(), {
            "case_id", "question", "relevance", "expected_route", "allow_web"
        })

    def test_duplicate_normalized_questions_are_rejected(self) -> None:
        payload = _manifest()
        payload["families"][1]["questions"][0] = "第一个 本地检索问题是什么"
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "duplicate normalized question"):
                load_retrieval_suite(path)

    def test_source_cannot_cross_development_and_test(self) -> None:
        payload = _manifest()
        payload["families"][2].update(
            expected_route="local",
            allow_web=False,
            relevance={"source.md": 3},
        )
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "cannot cross dataset splits"):
                load_retrieval_suite(path)

    def test_route_and_relevance_contract_is_strict(self) -> None:
        payload = _manifest()
        payload["families"][0]["expected_route"] = "refused"
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "non-local route"):
                load_retrieval_suite(path)

        payload = _manifest()
        payload["families"][2]["allow_web"] = False
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "allow_web"):
                load_retrieval_suite(path)

    def test_missing_and_escaping_sources_are_rejected(self) -> None:
        payload = _manifest()
        payload["families"][0]["relevance"] = {"missing.md": 3}
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "cannot resolve corpus source"):
                load_retrieval_suite(path)

        payload = _manifest()
        payload["families"][0]["relevance"] = {"../outside.md": 3}
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "safe relative path"):
                load_retrieval_suite(path)

    def test_minimum_coverage_is_enforced(self) -> None:
        payload = _manifest()
        payload["requirements"]["minimum_cases"] = 7
        with self._suite(payload) as path:
            with self.assertRaisesRegex(DatasetValidationError, "requires at least 7"):
                load_retrieval_suite(path)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "suite.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "invalid JSON object"):
                load_retrieval_suite(path)

    def test_validation_command_writes_auditable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_output = Path(directory) / "suite.json"
            markdown_output = Path(directory) / "suite.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "validate_retrieval_suite.py"),
                    str(PROJECT_ROOT / "evals" / "retrieval_suite.json"),
                    "--json-output",
                    str(json_output),
                    "--markdown-output",
                    str(markdown_output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(json_output.read_text(encoding="utf-8"))["case_count"], 200)
            self.assertIn("Coverage matrix", markdown_output.read_text(encoding="utf-8"))

    def test_sparse_command_runs_a_source_isolated_suite_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_output = Path(directory) / "run.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "benchmark_sparse.py"),
                    str(PROJECT_ROOT / "evals" / "retrieval_suite.json"),
                    "--split",
                    "validation",
                    "--json-output",
                    str(json_output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["report"]["case_count"], 60)
            self.assertEqual(payload["evaluated_split"], "validation")
            self.assertTrue(payload["slices"])
            self.assertTrue(payload["route_confusion"])

    def test_suite_run_reports_quality_slices_and_rejects_misalignment(self) -> None:
        suite = load_retrieval_suite(PROJECT_ROOT / "evals" / "retrieval_suite.json")
        source_by_question = {
            case.question: tuple(source for source, _grade in case.relevance)
            for case in suite.cases
        }

        class PerfectRetriever:
            def search(self, query: str, *, top_k: int):
                return tuple(
                    SearchHit(
                        Chunk(source, source, source, "相关资料", 0, 0, 4),
                        0.95,
                        dense_rank=rank,
                        sparse_rank=rank,
                        reasons=("dense", "sparse"),
                        lexical_score=1.0,
                    )
                    for rank, source in enumerate(source_by_question[query], start=1)
                )[:top_k]

        benchmark = run_retrieval_benchmark(
            suite.cases,
            PerfectRetriever(),
            RoutingPolicy(Settings()),
        )
        report = build_retrieval_suite_report(suite, benchmark)

        self.assertEqual(len(report.slices), 21)
        self.assertEqual(report.evaluated_split, "all")
        self.assertTrue(all(item.passed_case_count == item.case_count for item in report.slices))
        self.assertTrue(
            all(item.routing_signals.top_score.p50 >= 0.0 for item in report.slices)
        )
        self.assertEqual(
            report.slices[0].to_dict()["routing_signals"]["ranker_agreement_rate"],
            0.8,
        )
        self.assertIn("路由混淆矩阵", report.to_markdown())
        self.assertIn("margin p50", report.to_markdown())
        self.assertIn("N/A", report.to_markdown())
        self.assertEqual(json.loads(report.to_json())["report"]["case_count"], 200)

        with self.assertRaisesRegex(EvaluationSuiteError, "prediction order"):
            build_retrieval_suite_report(
                suite,
                replace(
                    benchmark,
                    predictions=tuple(reversed(benchmark.predictions)),
                ),
            )
        with self.assertRaisesRegex(EvaluationSuiteError, "counts do not match"):
            build_retrieval_suite_report(
                suite,
                replace(benchmark, report=replace(benchmark.report, case_count=199)),
            )

    class _SuiteContext:
        def __init__(self, payload: dict) -> None:
            self.payload = payload
            self.temporary: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.temporary = tempfile.TemporaryDirectory()
            root = Path(self.temporary.name)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "source.md").write_text("retrieval source", encoding="utf-8")
            path = root / "suite.json"
            path.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")
            return path

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            if self.temporary is not None:
                self.temporary.cleanup()

    def _suite(self, payload: dict) -> _SuiteContext:
        return self._SuiteContext(payload)


if __name__ == "__main__":
    unittest.main()
