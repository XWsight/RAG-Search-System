import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_system.domain import GeneratedAnswer
from scripts import benchmark_answers


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Settings:
    def validate(self):
        return self


class _Model:
    available = True

    def __init__(self, settings) -> None:
        del settings
        self.closed = False

    def answer(self, question, evidence):
        del question, evidence
        return GeneratedAnswer((), insufficient=True)

    def close(self) -> None:
        self.closed = True


class AnswerBenchmarkCommandTests(unittest.TestCase):
    def test_governed_suite_writes_slice_aware_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dotenv = root / ".env"
            output = root / "report.json"
            dotenv.write_text("TEST_ONLY=1\n", encoding="utf-8")

            with (
                patch.object(benchmark_answers, "Settings", _Settings),
                patch.object(benchmark_answers, "ZhipuChatModel", _Model),
                patch.object(benchmark_answers, "_load_evaluation_environment"),
            ):
                exit_code = benchmark_answers.main(
                    [
                        str(PROJECT_ROOT / "evals" / "answer_suite.json"),
                        "--split",
                        "validation",
                        "--dotenv",
                        str(dotenv),
                        "--json-output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["evaluated_split"], "validation")
            self.assertEqual(payload["benchmark"]["case_count"], 15)
            self.assertTrue(payload["slices"])
            self.assertEqual(
                {
                    item["value"]
                    for item in payload["slices"]
                    if item["dimension"] == "split"
                },
                {"validation"},
            )

    def test_legacy_jsonl_keeps_the_existing_report_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dotenv = root / ".env"
            output = root / "report.json"
            dotenv.write_text("TEST_ONLY=1\n", encoding="utf-8")

            with (
                patch.object(benchmark_answers, "Settings", _Settings),
                patch.object(benchmark_answers, "ZhipuChatModel", _Model),
                patch.object(benchmark_answers, "_load_evaluation_environment"),
            ):
                exit_code = benchmark_answers.main(
                    [
                        str(PROJECT_ROOT / "evals" / "answer_cases.jsonl"),
                        "--dotenv",
                        str(dotenv),
                        "--json-output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["case_count"], 4)
            self.assertNotIn("slices", payload)

    def test_split_is_rejected_for_legacy_jsonl_before_provider_setup(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--split requires"):
            benchmark_answers.main(
                [
                    str(PROJECT_ROOT / "evals" / "answer_cases.jsonl"),
                    "--split",
                    "development",
                    "--dotenv",
                    "missing.env",
                ]
            )


if __name__ == "__main__":
    unittest.main()
