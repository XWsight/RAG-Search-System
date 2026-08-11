from __future__ import annotations

import unittest

from rag_system.text import (
    lexical_relevance,
    lexical_tokens,
    normalize_text,
    stable_digest,
    truncate_text,
)
from rag_system.ranking import audit_citations, reciprocal_rank_fusion


class TextUtilitiesTests(unittest.TestCase):
    def test_normalize_text_handles_unicode_and_whitespace(self) -> None:
        self.assertEqual(normalize_text("  ＲＡＧ\n\t系统  "), "RAG 系统")

    def test_tokenizer_preserves_identifiers_and_chinese_bigrams(self) -> None:
        tokens = lexical_tokens("ChromaDB 1.5 向量数据库")
        self.assertIn("chromadb", tokens)
        self.assertIn("1.5", tokens)
        self.assertIn("向量", tokens)
        self.assertIn("数据", tokens)

    def test_lexical_relevance_rewards_exact_terms(self) -> None:
        relevant = lexical_relevance("什么是 ChromaDB", "ChromaDB 是一个向量数据库")
        irrelevant = lexical_relevance("什么是 ChromaDB", "今天的天气非常晴朗")
        self.assertGreater(relevant, irrelevant)
        self.assertGreaterEqual(relevant, 0.4)
        self.assertEqual(irrelevant, 0.0)

    def test_stable_digest_is_ordered_and_repeatable(self) -> None:
        first = stable_digest(["a", "bc"])
        second = stable_digest(["a", "bc"])
        reordered = stable_digest(["bc", "a"])
        self.assertEqual(first, second)
        self.assertNotEqual(first, reordered)

    def test_truncate_text_obeys_limit(self) -> None:
        self.assertEqual(truncate_text("abcdef", 5), "ab...")
        self.assertEqual(truncate_text("abc", 5), "abc")
        self.assertEqual(len(truncate_text("abcdef", 2)), 2)


class RankingUtilitiesTests(unittest.TestCase):
    def test_rrf_rewards_items_found_by_multiple_rankers(self) -> None:
        fused = reciprocal_rank_fusion(
            {
                "dense": ["semantic", "shared", "dense-only"],
                "lexical": ["shared", "exact", "semantic"],
            }
        )
        self.assertEqual(fused[0].item_id, "shared")
        self.assertEqual(set(fused[0].contributing_rankers), {"dense", "lexical"})

    def test_rrf_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"dense": ["x"]}, rank_constant=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"dense": ["x"]}, weights={"dense": -1})

    def test_citation_audit_detects_unknown_and_missing_citations(self) -> None:
        audit = audit_citations(
            "RAG 使用外部资料 [L1]。这是另一条没有证据的结论。引用错误 [W9]。",
            ["L1", "W1"],
        )
        self.assertEqual(audit.cited_ids, ("L1", "W9"))
        self.assertEqual(audit.invalid_ids, ("W9",))
        self.assertEqual(len(audit.uncited_sentences), 1)
        self.assertLess(audit.completeness, 1.0)


if __name__ == "__main__":
    unittest.main()
