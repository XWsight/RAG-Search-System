from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

from rag_system.answer_protocol import GroundedAnswerProtocol
from rag_system.domain import AnswerClaim, GeneratedAnswer
from rag_system.provider_errors import ProviderProtocolError


class GroundedAnswerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = GroundedAnswerProtocol(max_context_characters=1_000)

    def test_prepare_builds_immutable_bounded_vendor_neutral_messages(self) -> None:
        prepared = self.protocol.prepare(
            "  什么是 RAG？  ",
            (("L1", "甲" * 700), ("L2", "乙" * 700)),
        )

        self.assertEqual(tuple(item.role for item in prepared.messages), ("system", "user"))
        self.assertIn("不可信", prepared.messages[0].content)
        payload = json.loads(prepared.messages[1].content)
        self.assertEqual(payload["question"], "什么是 RAG？")
        self.assertEqual(sum(len(item["text"]) for item in payload["evidence"]), 1_000)
        self.assertTrue(payload["evidence"][-1]["text"].endswith("..."))
        self.assertEqual(prepared.citation_ids, ("L1", "L2"))
        with self.assertRaises(FrozenInstanceError):
            prepared.messages[0].role = "user"

    def test_prepare_rejects_ambiguous_or_unbounded_evidence(self) -> None:
        invalid = (
            (("source-1", "text"),),
            (("L1", "one"), ("L1", "two")),
            tuple((f"L{index}", "text") for index in range(1, 26)),
        )
        for evidence in invalid:
            with self.subTest(size=len(evidence)), self.assertRaises(ValueError):
                self.protocol.prepare("question", evidence)

    def test_decode_returns_domain_answer_after_exact_contract_validation(self) -> None:
        content = (
            '{"claims":[{"text":"RAG 是检索增强生成。",'
            '"citation_ids":["L1"]}],"insufficient":false}'
        )

        answer = self.protocol.decode(content, ("L1",))

        self.assertEqual(
            answer,
            GeneratedAnswer((AnswerClaim("RAG 是检索增强生成。", ("L1",)),)),
        )

    def test_decode_failures_have_stable_repairable_codes(self) -> None:
        invalid = (
            ("not-json", "answer_invalid_json"),
            ('{"claims":[],"claims":[],"insufficient":true}', "answer_invalid_json"),
            ('{"claims":[],"insufficient":NaN}', "answer_invalid_json"),
            ('{"claims":[],"insufficient":false}', "answer_grounding_contract"),
            (
                '{"claims":[{"text":"结论。","citation_ids":["W1"]}],'
                '"insufficient":false}',
                "answer_grounding_contract",
            ),
        )
        for content, code in invalid:
            with self.subTest(code=code), self.assertRaises(ProviderProtocolError) as caught:
                self.protocol.decode(content, ("L1",))
            self.assertEqual(caught.exception.code, code)
            self.assertTrue(caught.exception.repairable)

    def test_repair_message_never_contains_untrusted_previous_output(self) -> None:
        message = self.protocol.repair_message()
        self.assertEqual(message.role, "system")
        self.assertIn("原始 question/evidence", message.content)


if __name__ == "__main__":
    unittest.main()
