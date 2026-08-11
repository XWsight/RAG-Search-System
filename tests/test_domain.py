from __future__ import annotations

import unittest

from rag_system.domain import AnswerResult, Route, RouteDecision


class DomainInvariantTests(unittest.TestCase):
    def test_answer_result_copies_and_freezes_diagnostics(self) -> None:
        diagnostics = {"evidence_count": 2}
        result = AnswerResult(
            "answer",
            RouteDecision(Route.LOCAL, 0.9, "reason"),
            diagnostics=diagnostics,
        )

        diagnostics["evidence_count"] = 99
        self.assertEqual(result.diagnostics["evidence_count"], 2)
        with self.assertRaises(TypeError):
            result.diagnostics["evidence_count"] = 3

    def test_answer_result_rejects_non_mapping_diagnostics(self) -> None:
        with self.assertRaises(TypeError):
            AnswerResult(
                "answer",
                RouteDecision(Route.LOCAL, 0.9, "reason"),
                diagnostics=[],
            )


if __name__ == "__main__":
    unittest.main()
