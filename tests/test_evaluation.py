from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rag_system.evaluation import (
    DatasetValidationError,
    evaluate_cases,
    evaluation_case_from_mapping,
    load_evaluation_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _case(**overrides: object):
    payload: dict[str, object] = {
        "case_id": "case-1",
        "question": "测试问题",
        "relevance": {"a": 3},
        "retrieved_ids": ["a"],
        "expected_route": "local",
        "predicted_route": "local",
        "allowed_citation_ids": ["L1"],
        "answer": "这是有证据的事实 [L1]。",
        "citation_required": True,
    }
    payload.update(overrides)
    return evaluation_case_from_mapping(payload)


class EvaluationMetricTests(unittest.TestCase):
    def test_metrics_are_computed_at_requested_cutoff(self) -> None:
        first = _case(
            case_id="first",
            relevance={"a": 3, "b": 1},
            retrieved_ids=["b", "a", "x"],
            answer="事实一 [L1]。事实二 [L9]。",
        )
        second = _case(
            case_id="second",
            relevance={"c": 3},
            retrieved_ids=["x", "c"],
            predicted_route="web",
            answer="事实三 [L1]。事实四没有引用。",
        )

        report = evaluate_cases((first, second), top_k=2)
        ideal_first = 7 / math.log2(2) + 1 / math.log2(3)
        actual_first = 1 / math.log2(2) + 7 / math.log2(3)
        expected_ndcg = (actual_first / ideal_first + (7 / math.log2(3)) / 7) / 2

        self.assertEqual(report.metrics.recall_at_k, 1.0)
        self.assertEqual(report.metrics.mrr_at_k, 0.75)
        self.assertAlmostEqual(report.metrics.ndcg_at_k, expected_ndcg, places=10)
        self.assertEqual(report.metrics.route_accuracy, 0.5)
        self.assertAlmostEqual(report.metrics.citation_validity, 2 / 3)
        self.assertEqual(report.metrics.citation_coverage, 0.75)

    def test_report_is_deterministic_and_renders_both_formats(self) -> None:
        cases = (_case(),)
        first = evaluate_cases(cases, top_k=3)
        second = evaluate_cases(cases, top_k=3)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertIn('"recall_at_k": 1.0', first.to_json())
        self.assertIn("Recall@3", first.to_markdown())
        self.assertIn("引用覆盖率", first.to_markdown())

    def test_version_dots_do_not_create_false_uncited_sentences(self) -> None:
        case = _case(answer="使用 BAAI/bge-small-zh-v1.5 生成向量 [L1]。")
        report = evaluate_cases((case,), top_k=1)
        self.assertEqual(report.metrics.citation_coverage, 1.0)

    def test_unanswerable_cases_are_excluded_from_retrieval_metrics(self) -> None:
        case = _case(
            relevance={},
            retrieved_ids=[],
            expected_route="refused",
            predicted_route="refused",
            allowed_citation_ids=[],
            answer="资料不足。",
            citation_required=False,
        )
        report = evaluate_cases((case,), top_k=5)
        self.assertEqual(report.retrieval_case_count, 0)
        self.assertEqual(report.citation_case_count, 0)
        self.assertEqual(report.metrics.recall_at_k, 0.0)
        self.assertEqual(report.metrics.route_accuracy, 1.0)
        self.assertIn("| 引用有效率 | N/A |", report.to_markdown())
        self.assertIn("| 引用覆盖率 | N/A |", report.to_markdown())


class DatasetValidationTests(unittest.TestCase):
    def test_sample_dataset_loads(self) -> None:
        cases = load_evaluation_dataset(PROJECT_ROOT / "evals" / "sample_dataset.jsonl")
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case.case_id for case in cases}), 6)

    def test_unknown_fields_and_duplicate_ranked_ids_are_rejected(self) -> None:
        payload = _case().to_dict()
        payload["unexpected"] = True
        with self.assertRaises(DatasetValidationError):
            evaluation_case_from_mapping(payload)

        duplicate = _case().to_dict()
        duplicate["retrieved_ids"] = ["a", "a"]
        with self.assertRaises(DatasetValidationError):
            evaluation_case_from_mapping(duplicate)

    def test_invalid_relevance_route_and_citation_schema_are_rejected(self) -> None:
        with self.assertRaises(DatasetValidationError):
            _case(relevance={"a": 4})
        with self.assertRaises(DatasetValidationError):
            _case(expected_route="unknown")
        with self.assertRaises(DatasetValidationError):
            _case(allowed_citation_ids=["source-1"])
        with self.assertRaises(DatasetValidationError):
            _case(citation_required=True, answer="", allowed_citation_ids=["L1"])

    def test_loader_reports_blank_lines_and_duplicate_case_ids(self) -> None:
        valid_line = json.dumps(_case().to_dict(), ensure_ascii=False)
        with tempfile.TemporaryDirectory() as directory:
            blank_path = Path(directory, "blank.jsonl")
            blank_path.write_text(valid_line + "\n\n", encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                load_evaluation_dataset(blank_path)

            duplicate_path = Path(directory, "duplicate.jsonl")
            duplicate_path.write_text(valid_line + "\n" + valid_line, encoding="utf-8")
            with self.assertRaises(DatasetValidationError):
                load_evaluation_dataset(duplicate_path)


class EvaluationCommandTests(unittest.TestCase):
    def test_command_writes_json_and_markdown_without_external_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory, "report.json")
            markdown_path = Path(directory, "report.md")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                    str(PROJECT_ROOT / "evals" / "sample_dataset.jsonl"),
                    "--top-k",
                    "3",
                    "--json-output",
                    str(json_path),
                    "--markdown-output",
                    str(markdown_path),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["top_k"], 3)
            self.assertEqual(payload["case_count"], 6)
            self.assertIn("nDCG@3", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
