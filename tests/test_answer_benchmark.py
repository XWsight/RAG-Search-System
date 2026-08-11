from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_system.answer_benchmark import (
    AnswerDatasetError,
    load_answer_benchmark,
    run_answer_benchmark,
)
from rag_system.answer_quality_gate import (
    answer_quality_gate_from_mapping,
    evaluate_answer_quality_gate,
    load_answer_quality_gate,
)
from rag_system.domain import AnswerClaim, GeneratedAnswer


DATASET = Path(__file__).resolve().parents[1] / "evals" / "answer_cases.jsonl"
GATE = Path(__file__).resolve().parents[1] / "evals" / "gates" / "answer-live.json"


class AnswerBenchmarkTests(unittest.TestCase):
    def test_repository_ground_truth_is_strict_and_prediction_free(self) -> None:
        cases = load_answer_benchmark(DATASET)
        self.assertEqual(len(cases), 4)
        self.assertEqual(sum(len(case.facts) for case in cases), 8)
        self.assertTrue(cases[-1].should_refuse)
        self.assertEqual(cases[-1].facts, ())

    def test_perfect_structured_predictions_score_each_dimension_separately(self) -> None:
        cases = load_answer_benchmark(DATASET)

        def generate(question, evidence):
            del evidence
            if question == "什么是 RAG？":
                return GeneratedAnswer(
                    (
                        AnswerClaim("RAG 是检索增强生成。", ("L1",)),
                        AnswerClaim("系统先检索外部资料。", ("L1",)),
                        AnswerClaim("大模型基于资料生成回答。", ("L1",)),
                    )
                )
            if question == "关键词检索和向量检索分别擅长什么？":
                return GeneratedAnswer(
                    (
                        AnswerClaim("BM25 擅长精确词汇。", ("L1",)),
                        AnswerClaim("向量检索擅长语义相近的内容。", ("L2",)),
                    )
                )
            if question == "文档上传有什么安全限制？":
                return GeneratedAnswer(
                    (
                        AnswerClaim("上传应限制单文件大小。", ("L1",)),
                        AnswerClaim("上传应限制文件总量。", ("L1",)),
                        AnswerClaim("上传应限制允许格式。", ("L1",)),
                    )
                )
            return GeneratedAnswer((), insufficient=True)

        report = run_answer_benchmark(cases, generate)
        self.assertEqual(
            set(report.metrics.to_dict().values()),
            {1.0},
        )
        self.assertTrue(all(result.passed for result in report.results))
        self.assertIn("事实召回率", report.to_markdown())
        self.assertIn("无。", report.to_markdown())

    def test_combined_or_wrongly_attributed_claims_do_not_get_credit(self) -> None:
        cases = load_answer_benchmark(DATASET)

        def generate(question, evidence):
            del evidence
            if "分别" in question:
                return GeneratedAnswer(
                    (
                        AnswerClaim(
                            "BM25 擅长精确词汇，向量检索擅长语义相近内容。",
                            ("L1", "L2"),
                        ),
                    )
                )
            if "哪一天" in question:
                return GeneratedAnswer((AnswerClaim("论文发布于某日。", ("L1",)),))
            return GeneratedAnswer((AnswerClaim("无关结论。", ("L1",)),))

        report = run_answer_benchmark(cases, generate)
        self.assertEqual(report.metrics.contract_success_rate, 1.0)
        self.assertLess(report.metrics.refusal_accuracy, 1.0)
        self.assertEqual(report.metrics.fact_recall, 0.0)
        self.assertEqual(report.metrics.atomic_claim_rate, 0.0)
        self.assertEqual(report.metrics.attribution_precision, 0.0)
        self.assertTrue(all(not result.passed for result in report.results))

    def test_generator_protocol_failures_are_bounded_to_safe_error_codes(self) -> None:
        cases = load_answer_benchmark(DATASET)

        def fail(question, evidence):
            del question, evidence
            raise RuntimeError("secret upstream detail")

        report = run_answer_benchmark(cases, fail)
        self.assertEqual(report.metrics.contract_success_rate, 0.0)
        self.assertEqual({item.error_code for item in report.results}, {"RuntimeError"})
        self.assertNotIn("secret upstream detail", report.to_json())

    def test_dataset_rejects_duplicate_keys_and_ambiguous_refusal_labels(self) -> None:
        invalid = (
            '{"case_id":"x","case_id":"y"}\n',
            (
                '{"case_id":"x","question":"q","evidence":[{"citation_id":"L1",'
                '"text":"e"}],"facts":[],"should_refuse":false}\n'
            ),
        )
        for content in invalid:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "cases.jsonl"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(AnswerDatasetError):
                    load_answer_benchmark(path)

    def test_manual_quality_gate_is_dataset_bound_and_metric_complete(self) -> None:
        cases = load_answer_benchmark(DATASET)

        def perfect(question, evidence):
            case = next(item for item in cases if item.question == question)
            if case.should_refuse:
                return GeneratedAnswer((), insufficient=True)
            claims = tuple(
                AnswerClaim(
                    " ".join(group[0] for group in fact.term_groups),
                    (fact.supporting_citation_ids[0],),
                )
                for fact in case.facts
            )
            return GeneratedAnswer(claims)

        report = run_answer_benchmark(cases, perfect)
        gate = load_answer_quality_gate(GATE)
        self.assertTrue(evaluate_answer_quality_gate(report, gate).passed)

        def fail(question, evidence):
            del question, evidence
            raise RuntimeError("unavailable")

        failed = run_answer_benchmark(cases, fail)
        result = evaluate_answer_quality_gate(failed, gate)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.violations), 5)

        payload = {
            "schema_version": 1,
            "dataset_digest": report.dataset_digest,
            "minimum_metrics": {"fact_recall": 0.5},
        }
        with self.assertRaises(AnswerDatasetError):
            answer_quality_gate_from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
