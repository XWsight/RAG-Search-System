from __future__ import annotations

import unittest

from rag_system.domain import AnswerClaim, GeneratedAnswer
from rag_system.grounding import (
    GroundingContractError,
    render_grounded_answer,
    validate_grounded_answer,
)


class GroundingContractTests(unittest.TestCase):
    def test_valid_claims_have_stable_rendering_and_audit(self) -> None:
        answer = GeneratedAnswer(
            (
                AnswerClaim("RAG 先检索资料。", ("L1",)),
                AnswerClaim("证据不足时应拒答。", ("L2", "W1")),
            )
        )
        audit = validate_grounded_answer(answer, ("L1", "L2", "W1"))
        self.assertEqual(audit.claim_count, 2)
        self.assertEqual(audit.citation_count, 3)
        self.assertEqual(audit.used_citation_ids, ("L1", "L2", "W1"))
        self.assertEqual(
            render_grounded_answer(answer),
            "RAG 先检索资料。 [L1]\n\n证据不足时应拒答。 [L2] [W1]",
        )

    def test_insufficient_answer_is_an_explicit_empty_state(self) -> None:
        answer = GeneratedAnswer((), insufficient=True)
        audit = validate_grounded_answer(answer, ("L1",))
        self.assertEqual(audit.claim_count, 0)
        self.assertEqual(render_grounded_answer(answer), "现有资料不足以回答这个问题。")

    def test_sufficient_and_insufficient_states_cannot_be_ambiguous(self) -> None:
        invalid = (
            GeneratedAnswer(()),
            GeneratedAnswer((AnswerClaim("结论。", ("L1",)),), insufficient=True),
        )
        for answer in invalid:
            with self.subTest(answer=answer), self.assertRaises(GroundingContractError):
                validate_grounded_answer(answer, ("L1",))

    def test_every_claim_requires_available_unique_evidence(self) -> None:
        invalid = (
            AnswerClaim("没有引用。", ()),
            AnswerClaim("越界引用。", ("W9",)),
            AnswerClaim("重复引用。", ("L1", "L1")),
            AnswerClaim("内嵌引用 [L1]。", ("L1",)),
        )
        for claim in invalid:
            with self.subTest(claim=claim), self.assertRaises(GroundingContractError):
                validate_grounded_answer(GeneratedAnswer((claim,)), ("L1",))

    def test_duplicate_or_unbounded_claims_are_rejected(self) -> None:
        duplicate = GeneratedAnswer(
            (
                AnswerClaim("同一结论。", ("L1",)),
                AnswerClaim("同一结论。", ("L1",)),
            )
        )
        too_many = GeneratedAnswer(
            tuple(AnswerClaim(f"结论 {index}", ("L1",)) for index in range(25))
        )
        overlong = GeneratedAnswer((AnswerClaim("字" * 2_001, ("L1",)),))
        for answer in (duplicate, too_many, overlong):
            with self.subTest(answer=answer), self.assertRaises(GroundingContractError):
                validate_grounded_answer(answer, ("L1",))

    def test_claim_text_is_rejected_instead_of_silently_rewritten(self) -> None:
        for text in ("控制\x00字符", " 前导空白"):
            with self.subTest(text=text), self.assertRaises(GroundingContractError):
                validate_grounded_answer(
                    GeneratedAnswer((AnswerClaim(text, ("L1",)),)),
                    ("L1",),
                )

    def test_evidence_registry_is_strict_and_unambiguous(self) -> None:
        answer = GeneratedAnswer((AnswerClaim("结论。", ("L1",)),))
        for allowed in (("L1", "L1"), ("source-1",), "L1"):
            with self.subTest(allowed=allowed), self.assertRaises(GroundingContractError):
                validate_grounded_answer(answer, allowed)


if __name__ == "__main__":
    unittest.main()
