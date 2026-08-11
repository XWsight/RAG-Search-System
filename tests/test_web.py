import unittest

from rag_system.domain import WebSearchResult
from rag_system.web import canonicalize_url, rank_web_results


def result(identifier: str, title: str, content: str, url: str) -> WebSearchResult:
    return WebSearchResult(identifier, title, content, url)


class WebResultQualityTests(unittest.TestCase):
    def test_url_canonicalization_removes_fragments_tracking_and_default_ports(self) -> None:
        self.assertEqual(
            canonicalize_url("HTTPS://Example.COM:443/a/?utm_source=x&b=2&a=1#part"),
            "https://example.com/a?a=1&b=2",
        )
        self.assertEqual(canonicalize_url("file:///tmp/a"), "")

    def test_duplicate_urls_and_duplicate_content_are_removed(self) -> None:
        ranked = rank_web_results(
            "混合检索",
            [
                result("1", "混合检索", "融合关键词与向量检索", "https://a.test/x?utm_source=one"),
                result("2", "混合检索", "不同摘要", "https://a.test/x?utm_source=two"),
                result("3", "混合检索", "融合关键词与向量检索", "https://b.test/y"),
            ],
        )
        self.assertEqual([item.result.result_id for item in ranked], ["1"])

    def test_ranking_is_relevant_bounded_and_domain_diverse(self) -> None:
        ranked = rank_web_results(
            "什么是 RAG 检索",
            [
                result("a1", "RAG 检索", "RAG 会先检索证据。" * 30, "https://a.test/1"),
                result("a2", "RAG", "检索增强生成。" * 30, "https://a.test/2"),
                result("a3", "RAG", "检索增强生成。另一个页面。" * 30, "https://a.test/3"),
                result("b1", "无关", "天气预报", "https://b.test/1"),
                result("c1", "RAG 资料", "向量检索与证据。" * 20, "https://c.test/1"),
            ],
            limit=3,
            per_domain=1,
        )
        self.assertEqual(len(ranked), 3)
        self.assertEqual(len({item.domain for item in ranked}), 3)
        self.assertEqual(ranked[0].result.result_id, "a1")

    def test_invalid_limits_and_blank_query(self) -> None:
        with self.assertRaises(ValueError):
            rank_web_results("q", [], limit=0)
        self.assertEqual(rank_web_results("   ", []), ())


if __name__ == "__main__":
    unittest.main()
