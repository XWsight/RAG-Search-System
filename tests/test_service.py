from __future__ import annotations

import unittest
from contextlib import contextmanager

from rag_system.config import Settings
from rag_system.domain import (
    AnswerRequest,
    Chunk,
    Route,
    SearchHit,
    WebSearchResult,
)
from rag_system.providers import ProviderUnavailableError
from rag_system.service import RagService


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query: str, *, top_k: int):
        self.queries.append(query)
        return self.hits[:top_k]


class FakeIndexManager:
    def __init__(self, retriever):
        self.retriever = retriever

    @contextmanager
    def lease(self, index_id: str):
        if index_id != "idx":
            raise KeyError(index_id)
        yield self.retriever


class FakeChat:
    def __init__(self, answer="基于资料的回答 [L1]。", *, available=True, error=None):
        self.response = answer
        self.available = available
        self.error = error
        self.calls = 0

    def answer(self, question, evidence):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class FakeWeb:
    def __init__(self, results=(), *, available=True):
        self.results = results
        self.available = available
        self.calls = 0

    def search(self, query, *, count):
        self.calls += 1
        return self.results


class FakePlanner:
    def __init__(self, queries=(), *, available=True, error=None):
        self.queries = queries
        self.available = available
        self.error = error
        self.calls = 0

    def plan_queries(self, question, *, max_queries):
        self.calls += 1
        if self.error:
            raise self.error
        return self.queries[:max_queries]


def make_hit(score: float, reasons=("dense", "sparse")) -> SearchHit:
    text = "RAG 使用检索到的资料生成答案。"
    chunk = Chunk("chunk", "doc", "guide.md", text, 0, 0, len(text), "原理")
    return SearchHit(chunk, score, reasons=reasons)


class RagServiceTests(unittest.TestCase):
    def build_service(self, hit, chat=None, web=None, planner=None):
        chat = chat or FakeChat()
        web = web or FakeWeb()
        service = RagService(
            Settings(),
            FakeIndexManager(FakeRetriever([hit] if hit else [])),
            chat,
            web,
            query_planner=planner,
        )
        return service, chat, web

    def test_local_answer_uses_chat_and_never_calls_web(self) -> None:
        service, chat, web = self.build_service(make_hit(0.9))
        result = service.answer("idx", AnswerRequest("什么是 RAG", "s", True, True))
        self.assertEqual(result.decision.route, Route.LOCAL)
        self.assertEqual(chat.calls, 1)
        self.assertEqual(web.calls, 0)
        self.assertEqual(result.citations[0].citation_id, "L1")

    def test_privacy_switches_prevent_all_network_calls(self) -> None:
        service, chat, web = self.build_service(make_hit(0.9))
        result = service.answer("idx", AnswerRequest("什么是 RAG", "s", False, False))
        self.assertEqual(result.decision.route, Route.RETRIEVAL_ONLY)
        self.assertEqual(chat.calls, 0)
        self.assertEqual(web.calls, 0)

    def test_low_confidence_can_refuse_without_web(self) -> None:
        service, chat, web = self.build_service(make_hit(0.05, ("dense",)))
        result = service.answer("idx", AnswerRequest("未知问题", "s", True, False))
        self.assertEqual(result.decision.route, Route.REFUSED)
        self.assertEqual(chat.calls, 0)
        self.assertEqual(web.calls, 0)

    def test_web_route_builds_valid_web_citations(self) -> None:
        web_result = WebSearchResult("r", "可靠来源", "联网找到的资料", "https://example.com")
        service, chat, web = self.build_service(
            make_hit(0.05, ("dense",)),
            chat=FakeChat("联网回答 [W1]。"),
            web=FakeWeb([web_result]),
        )
        result = service.answer("idx", AnswerRequest("联网问题", "s", True, True))
        self.assertEqual(result.decision.route, Route.WEB)
        self.assertEqual(web.calls, 1)
        self.assertEqual(chat.calls, 1)
        self.assertEqual(result.citations[0].citation_id, "W1")

    def test_invalid_model_citations_are_removed(self) -> None:
        service, _, _ = self.build_service(make_hit(0.9), chat=FakeChat("正确 [L1]，错误 [W99]。"))
        result = service.answer("idx", AnswerRequest("问题", "s", True, False))
        self.assertIn("[L1]", result.answer)
        self.assertNotIn("[W99]", result.answer)
        self.assertEqual(result.diagnostics["invalid_citations_removed"], 1)
        self.assertEqual(result.diagnostics["invalid_citations"], 0)

    def test_provider_failure_preserves_retrieved_evidence(self) -> None:
        chat = FakeChat(error=ProviderUnavailableError("offline"))
        service, _, _ = self.build_service(make_hit(0.9), chat=chat)
        result = service.answer("idx", AnswerRequest("问题", "s", True, False))
        self.assertEqual(result.decision.route, Route.ERROR)
        self.assertEqual(len(result.citations), 1)
        self.assertNotIn("offline", result.answer)

    def test_recent_questions_contextualize_retrieval_without_cross_session_leakage(self) -> None:
        service, _, _ = self.build_service(make_hit(0.9))
        retriever = service.index_manager.retriever

        first = service.answer("idx", AnswerRequest("介绍混合检索", "session-a", False, False))
        second = service.answer("idx", AnswerRequest("它有什么优点", "session-a", False, False))
        service.answer("idx", AnswerRequest("完全不同的问题", "session-b", False, False))

        self.assertEqual(first.diagnostics["history_turns"], 0)
        self.assertEqual(second.diagnostics["history_turns"], 1)
        self.assertIn("介绍混合检索", retriever.queries[1])
        self.assertIn("它有什么优点", retriever.queries[1])
        self.assertNotIn("云端生成未开启", retriever.queries[1])
        self.assertEqual(retriever.queries[2], "完全不同的问题")

    def test_clear_session_removes_retrieval_history(self) -> None:
        service, _, _ = self.build_service(make_hit(0.9))
        service.answer("idx", AnswerRequest("第一问", "session-a", False, False))
        self.assertTrue(service.clear_session("session-a"))
        service.answer("idx", AnswerRequest("第二问", "session-a", False, False))
        self.assertEqual(service.index_manager.retriever.queries[-1], "第二问")

    def test_research_mode_uses_bounded_query_plan(self) -> None:
        planner = FakePlanner(("子问题一", "子问题二"))
        service, _, _ = self.build_service(make_hit(0.9), planner=planner)
        result = service.answer(
            "idx",
            AnswerRequest("复杂问题", "session", True, False, True),
        )
        self.assertEqual(planner.calls, 1)
        self.assertEqual(len(service.index_manager.retriever.queries), 3)
        self.assertEqual(result.diagnostics["planned_query_count"], 3)
        self.assertEqual(result.diagnostics["web_query_count"], 0)

    def test_research_mode_never_plans_without_cloud_permission(self) -> None:
        planner = FakePlanner(("不应调用",))
        service, _, _ = self.build_service(make_hit(0.9), planner=planner)
        service.answer("idx", AnswerRequest("问题", "session", False, False, True))
        self.assertEqual(planner.calls, 0)

    def test_research_mode_has_bounded_multi_query_web_search(self) -> None:
        planner = FakePlanner(("查询一", "查询二", "查询三", "查询四"))
        web_result = WebSearchResult("r", "来源", "研究资料", "https://example.com")
        service, _, web = self.build_service(
            make_hit(0.9),
            web=FakeWeb([web_result]),
            planner=planner,
        )
        result = service.answer(
            "idx",
            AnswerRequest("复杂问题", "session", True, True, True),
        )
        self.assertEqual(web.calls, service.settings.research_max_web_queries)
        self.assertEqual(result.diagnostics["web_query_count"], 3)
        self.assertEqual(result.decision.route, Route.HYBRID)


if __name__ == "__main__":
    unittest.main()
