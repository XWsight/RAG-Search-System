import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag_system.answer_analysis import build_answer_suite_report
from rag_system.answer_benchmark import load_answer_benchmark, run_answer_benchmark
from rag_system.answer_suite import load_answer_suite, validate_answer_suite_contract
from rag_system.domain import AnswerClaim, GeneratedAnswer
from rag_system.evaluation_suite import EvaluationSuiteError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _answerable_case() -> dict:
    return {
        "case_id": "development_answer",
        "category": "grounding",
        "split": "development",
        "difficulty": "easy",
        "risk_tags": ["citation"],
        "question": "系统根据资料应当给出什么结论？",
        "evidence": [
            {"citation_id": "L1", "text": "系统必须引用资料，并且只回答有证据的内容。"}
        ],
        "facts": [
            {
                "fact_id": "cite_sources",
                "term_groups": [["引用", "标注"], ["资料", "证据"]],
                "supporting_citation_ids": ["L1"],
            }
        ],
        "should_refuse": False,
    }


def _refusal_case(case_id: str, split: str, difficulty: str) -> dict:
    return {
        "case_id": case_id,
        "category": "refusal",
        "split": split,
        "difficulty": difficulty,
        "risk_tags": ["missing_evidence"],
        "question": f"资料没有包含的第{case_id}项结论是什么？",
        "evidence": [
            {"citation_id": "L2", "text": f"资料只描述系统的运行状态，不包含{case_id}。"}
        ],
        "facts": [],
        "should_refuse": True,
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "suite_id": "temporary_answer_suite",
        "description": "A sufficiently descriptive temporary structured answer benchmark suite.",
        "requirements": {
            "minimum_cases": 3,
            "minimum_categories": 2,
            "minimum_facts": 1,
            "minimum_answerable_cases": 1,
            "minimum_refusal_cases": 2,
            "minimum_risk_tags": 2,
            "minimum_cases_per_split": {
                "development": 1,
                "validation": 1,
                "test": 1,
            },
            "minimum_cases_per_difficulty": {"easy": 1, "medium": 1, "hard": 1},
        },
        "cases": [
            _answerable_case(),
            _refusal_case("validation_refusal", "validation", "medium"),
            _refusal_case("test_refusal", "test", "hard"),
        ],
    }


class AnswerBenchmarkSuiteTests(unittest.TestCase):
    def test_repository_suite_has_declared_coverage(self) -> None:
        suite = load_answer_suite(PROJECT_ROOT / "evals" / "answer_suite.json")

        self.assertEqual(len(suite.cases), 50)
        self.assertEqual(len(suite.benchmark_cases), 50)
        self.assertEqual(suite.summary()["fact_count"], 70)
        self.assertEqual(suite.summary()["answerable_case_count"], 35)
        self.assertEqual(suite.summary()["refusal_case_count"], 15)
        self.assertEqual(
            suite.summary()["cases_by_split"],
            {"development": 20, "test": 15, "validation": 15},
        )
        self.assertEqual(len(suite.cases_for_split("validation")), 15)
        self.assertIn("Coverage matrix", suite.to_markdown())
        validate_answer_suite_contract(
            suite, PROJECT_ROOT / "evals" / "gates" / "answer-suite.json"
        )

    def test_duplicate_normalized_questions_are_rejected(self) -> None:
        payload = _manifest()
        payload["cases"][1]["question"] = "系统 根据资料，应当给出什么结论？！"
        with self._suite(payload) as path:
            with self.assertRaisesRegex(EvaluationSuiteError, "duplicate normalized"):
                load_answer_suite(path)

    def test_duplicate_normalized_evidence_is_rejected(self) -> None:
        payload = _manifest()
        payload["cases"][1]["evidence"][0]["text"] = (
            "系统必须引用资料，并且只回答有证据的内容！"
        )
        with self._suite(payload) as path:
            with self.assertRaisesRegex(EvaluationSuiteError, "duplicate normalized answer evidence"):
                load_answer_suite(path)

    def test_minimum_coverage_is_enforced(self) -> None:
        payload = _manifest()
        payload["requirements"]["minimum_facts"] = 2
        with self._suite(payload) as path:
            with self.assertRaisesRegex(EvaluationSuiteError, "facts=1<2"):
                load_answer_suite(path)

    def test_split_and_difficulty_coverage_are_enforced(self) -> None:
        payload = _manifest()
        payload["requirements"]["minimum_cases_per_split"]["test"] = 2
        with self._suite(payload) as path:
            with self.assertRaisesRegex(EvaluationSuiteError, "test=1<2"):
                load_answer_suite(path)

    def test_risk_tags_must_be_unique_identifiers(self) -> None:
        payload = _manifest()
        payload["cases"][0]["risk_tags"] = ["citation", "citation"]
        with self._suite(payload) as path:
            with self.assertRaisesRegex(EvaluationSuiteError, "must be unique"):
                load_answer_suite(path)

    def test_ambiguous_reference_claim_is_rejected(self) -> None:
        payload = _manifest()
        payload["cases"][0]["facts"].append(
            {
                "fact_id": "same_claim",
                "term_groups": [["引用"], ["资料"]],
                "supporting_citation_ids": ["L1"],
            }
        )
        with self._suite(payload) as path:
            with self.assertRaisesRegex(EvaluationSuiteError, "ambiguous or unsatisfiable"):
                load_answer_suite(path)

    def test_answer_ground_truth_errors_are_wrapped(self) -> None:
        payload = _manifest()
        payload["cases"][1]["should_refuse"] = False
        with self._suite(payload) as path:
            with self.assertRaises(EvaluationSuiteError):
                load_answer_suite(path)

    def test_frozen_contract_detects_suite_drift(self) -> None:
        with self._suite(_manifest()) as path:
            suite = load_answer_suite(path)
            contract = {
                key: value
                for key, value in suite.summary().items()
                if key != "suite_id"
            }
            contract["suite_digest"] = "0" * 16
            contract_path = path.parent / "contract.json"
            contract_path.write_text(
                json.dumps(contract, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(EvaluationSuiteError, "suite_digest"):
                validate_answer_suite_contract(suite, contract_path)

    def test_validation_command_writes_auditable_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_output = Path(directory) / "suite.json"
            markdown_output = Path(directory) / "suite.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "validate_answer_suite.py"),
                    str(PROJECT_ROOT / "evals" / "answer_suite.json"),
                    "--contract",
                    str(PROJECT_ROOT / "evals" / "gates" / "answer-suite.json"),
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
            self.assertEqual(json.loads(json_output.read_text(encoding="utf-8"))["case_count"], 50)
            self.assertIn("Coverage matrix", markdown_output.read_text(encoding="utf-8"))

    def test_full_run_reports_every_governed_quality_slice(self) -> None:
        suite = load_answer_suite(PROJECT_ROOT / "evals" / "answer_suite.json")
        benchmark = run_answer_benchmark(suite.benchmark_cases, self._perfect_generator(suite))

        report = build_answer_suite_report(suite, benchmark)

        self.assertEqual(report.evaluated_split, "all")
        self.assertEqual(len(report.slices), 34)
        self.assertTrue(all(item.passed_case_count == item.case_count for item in report.slices))
        self.assertEqual(
            [item.value for item in report.slices if item.dimension == "split"],
            ["development", "test", "validation"],
        )
        self.assertEqual(len([item for item in report.slices if item.dimension == "category"]), 13)
        self.assertEqual(len([item for item in report.slices if item.dimension == "risk_tag"]), 15)
        self.assertEqual(json.loads(report.to_json())["benchmark"]["case_count"], 50)
        self.assertIn("## 质量切片", report.to_markdown())
        self.assertIn("N/A", report.to_markdown())
        self.assertIn("无。", report.to_markdown())

    def test_slice_exposes_localized_failure_instead_of_only_average(self) -> None:
        suite = load_answer_suite(PROJECT_ROOT / "evals" / "answer_suite.json")
        perfect = self._perfect_generator(suite)

        def generate(question, evidence):
            if question == "上传文档在解析前要限制什么？":
                raise RuntimeError("provider detail must not enter the report")
            return perfect(question, evidence)

        benchmark = run_answer_benchmark(suite.benchmark_cases, generate)
        report = build_answer_suite_report(suite, benchmark)
        ingestion = next(
            item
            for item in report.slices
            if item.dimension == "category" and item.value == "ingestion"
        )

        self.assertEqual(ingestion.case_count, 4)
        self.assertEqual(ingestion.passed_case_count, 3)
        self.assertEqual(ingestion.failure_case_ids, ("loader_limits",))
        self.assertEqual(ingestion.metrics.contract_success_rate, 0.75)
        self.assertNotIn("provider detail", report.to_json())

    def test_split_run_only_reports_selected_suite_cases(self) -> None:
        suite = load_answer_suite(PROJECT_ROOT / "evals" / "answer_suite.json")
        cases = suite.cases_for_split("validation")
        benchmark = run_answer_benchmark(cases, self._perfect_generator(suite))

        report = build_answer_suite_report(suite, benchmark, split="validation")

        self.assertEqual(report.evaluated_split, "validation")
        self.assertEqual(report.benchmark.case_count, 15)
        self.assertEqual(
            [(item.dimension, item.value) for item in report.slices if item.dimension == "split"],
            [("split", "validation")],
        )

    def test_slice_report_rejects_results_from_another_dataset(self) -> None:
        suite = load_answer_suite(PROJECT_ROOT / "evals" / "answer_suite.json")
        legacy = load_answer_benchmark(PROJECT_ROOT / "evals" / "answer_cases.jsonl")
        benchmark = run_answer_benchmark(legacy, self._perfect_generator_for_cases(legacy))

        with self.assertRaisesRegex(EvaluationSuiteError, "digest does not match"):
            build_answer_suite_report(suite, benchmark)

    def test_slice_report_rejects_reordered_or_inconsistent_results(self) -> None:
        suite = load_answer_suite(PROJECT_ROOT / "evals" / "answer_suite.json")
        benchmark = run_answer_benchmark(suite.benchmark_cases, self._perfect_generator(suite))

        with self.assertRaisesRegex(EvaluationSuiteError, "result order"):
            build_answer_suite_report(
                suite,
                replace(benchmark, results=tuple(reversed(benchmark.results))),
            )
        with self.assertRaisesRegex(EvaluationSuiteError, "counts do not match"):
            build_answer_suite_report(suite, replace(benchmark, case_count=49))

    class _SuiteContext:
        def __init__(self, payload: dict) -> None:
            self.payload = payload
            self.temporary: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            self.temporary = tempfile.TemporaryDirectory()
            path = Path(self.temporary.name) / "suite.json"
            path.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")
            return path

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            if self.temporary is not None:
                self.temporary.cleanup()

    def _suite(self, payload: dict) -> _SuiteContext:
        return self._SuiteContext(payload)

    def _perfect_generator(self, suite):
        return self._perfect_generator_for_cases(suite.benchmark_cases)

    def _perfect_generator_for_cases(self, cases):
        by_question = {case.question: case for case in cases}

        def generate(question, evidence):
            del evidence
            case = by_question[question]
            if case.should_refuse:
                return GeneratedAnswer((), insufficient=True)
            return GeneratedAnswer(
                tuple(
                    AnswerClaim(
                        " ".join(group[0] for group in fact.term_groups),
                        (fact.supporting_citation_ids[0],),
                    )
                    for fact in case.facts
                )
            )

        return generate


if __name__ == "__main__":
    unittest.main()
