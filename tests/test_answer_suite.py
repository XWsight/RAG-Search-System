import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rag_system.answer_suite import load_answer_suite, validate_answer_suite_contract
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


if __name__ == "__main__":
    unittest.main()
