from __future__ import annotations

import math
import unittest

from rag_system.sparse import BM25Index, SparseDocument, stable_document_id


class StableDocumentIdTests(unittest.TestCase):
    def test_id_is_repeatable_and_normalizes_superficial_whitespace(self) -> None:
        first = stable_document_id("  ChromaDB\n向量数据库  ")
        second = stable_document_id("ChromaDB 向量数据库")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("doc-"))

    def test_namespace_separates_identical_text(self) -> None:
        text = "RAG 使用外部知识库。"
        self.assertNotEqual(
            stable_document_id(text, namespace="manual"),
            stable_document_id(text, namespace="website"),
        )


class BM25IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index(
            [
                SparseDocument("rag", "RAG 会检索外部知识，再让模型生成回答。"),
                SparseDocument("vector", "ChromaDB 是保存文本向量的向量数据库。"),
                SparseDocument("python", "Python 列表可以调用 append 添加元素。"),
            ]
        )

    def test_exact_identifier_and_chinese_terms_rank_relevant_document_first(self) -> None:
        results = self.index.search("ChromaDB 向量数据库", top_k=2)
        self.assertEqual(results[0].document_id, "vector")
        self.assertGreater(results[0].score, 0)
        self.assertEqual(len({hit.document_id for hit in results}), len(results))

    def test_top_k_limits_results_and_ties_use_stable_document_id_order(self) -> None:
        index = BM25Index(
            [
                SparseDocument("b", "共同词"),
                SparseDocument("a", "共同词"),
                SparseDocument("c", "共同词"),
            ]
        )
        results = index.search("共同词", top_k=2)
        self.assertEqual([hit.document_id for hit in results], ["a", "b"])

    def test_empty_or_tokenless_query_returns_no_results(self) -> None:
        self.assertEqual(self.index.search("   "), ())
        self.assertEqual(self.index.search("!!!"), ())
        self.assertEqual(BM25Index().search("RAG"), ())

    def test_duplicate_id_and_text_is_collapsed(self) -> None:
        document = SparseDocument("same", "相同文档内容")
        index = BM25Index([document, document])
        self.assertEqual(index.document_count, 1)
        self.assertEqual(len(index.search("文档内容")), 1)

    def test_generated_ids_deduplicate_identical_normalized_texts(self) -> None:
        index = BM25Index.from_texts(["RAG  系统", " RAG 系统 "])
        self.assertEqual(index.document_count, 1)
        self.assertEqual(len(index.document_ids), 1)

    def test_duplicate_id_with_different_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BM25Index(
                [
                    SparseDocument("conflict", "第一份内容"),
                    SparseDocument("conflict", "第二份内容"),
                ]
            )

    def test_id_and_text_lengths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            BM25Index.from_texts(["一", "二"], document_ids=["one"])

    def test_parameter_validation(self) -> None:
        for k1 in (0, -1, math.inf, math.nan):
            with self.subTest(k1=k1), self.assertRaises(ValueError):
                BM25Index(k1=k1)
        for b in (-0.01, 1.01, math.inf, math.nan):
            with self.subTest(b=b), self.assertRaises(ValueError):
                BM25Index(b=b)

        with self.assertRaises(TypeError):
            BM25Index(k1=True)
        with self.assertRaises(TypeError):
            self.index.search("RAG", top_k=True)
        with self.assertRaises(ValueError):
            self.index.search("RAG", top_k=0)

    def test_document_validation(self) -> None:
        with self.assertRaises(ValueError):
            SparseDocument("", "内容")
        with self.assertRaises(ValueError):
            SparseDocument("empty", "   ")
        with self.assertRaises(ValueError):
            SparseDocument("punctuation", "!!!")


if __name__ == "__main__":
    unittest.main()
