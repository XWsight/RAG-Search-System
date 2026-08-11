from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import requests

from rag_system.config import SecretValue, Settings
from rag_system.providers import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    ProviderUnavailableError,
    ZhipuChatModel,
    ZhipuWebSearch,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        data: object = None,
        *,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._data = data
        self._json_error = json_error

    @property
    def text(self) -> str:
        raise AssertionError("providers must never read an upstream response body")

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        return self._data


class FakeSession:
    def __init__(self, *actions: FakeResponse | Exception) -> None:
        self._actions = list(actions)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self._actions:
            raise AssertionError("unexpected HTTP request")
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    def close(self) -> None:
        self.closed = True


def configured_settings(*, retry_attempts: int = 2) -> Settings:
    return replace(
        Settings(),
        api_key=SecretValue("test-private-key"),
        retry_attempts=retry_attempts,
        connect_timeout_seconds=1.5,
        read_timeout_seconds=4.5,
    ).validate()


class ZhipuChatModelTests(unittest.TestCase):
    def test_close_releases_the_http_session(self) -> None:
        session = FakeSession()
        model = ZhipuChatModel(configured_settings(), session=session)
        model.close()
        model.close()
        self.assertTrue(session.closed)

    def test_valid_response_uses_split_timeout_and_grounded_messages(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                {"choices": [{"message": {"content": "RAG 是检索增强生成。[S1]"}}]},
            )
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        answer = model.answer("什么是 RAG？", [("S1", "RAG 会先检索资料。")])

        self.assertEqual(answer, "RAG 是检索增强生成。[S1]")
        self.assertTrue(model.available)
        self.assertEqual(session.calls[0]["timeout"], (1.5, 4.5))
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer test-private-key")
        messages = session.calls[0]["json"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("不可信", messages[0]["content"])
        self.assertNotIn("test-private-key", str(session.calls[0]["json"]))

    def test_authentication_error_is_not_retried_or_leaked(self) -> None:
        session = FakeSession(FakeResponse(401, {"error": "sensitive body"}))
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        with self.assertRaises(ProviderAuthenticationError) as caught:
            model.answer("问题", [])

        self.assertEqual(len(session.calls), 1)
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertNotIn("test-private-key", str(caught.exception))

    def test_retryable_429_can_recover_within_bound(self) -> None:
        delays: list[float] = []
        session = FakeSession(
            FakeResponse(429, {"error": "busy"}),
            FakeResponse(200, {"choices": [{"message": {"content": "恢复成功"}}]}),
        )
        model = ZhipuChatModel(configured_settings(retry_attempts=1), session=session, sleeper=delays.append)

        self.assertEqual(model.answer("问题", []), "恢复成功")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(delays, [0.25])

    def test_timeout_exhausts_bounded_retries_with_sanitized_error(self) -> None:
        session = FakeSession(
            requests.Timeout("socket details and test-private-key"),
            requests.Timeout("socket details and test-private-key"),
            requests.Timeout("socket details and test-private-key"),
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        with self.assertRaises(ProviderUnavailableError) as caught:
            model.answer("问题", [])

        self.assertEqual(len(session.calls), 3)
        self.assertNotIn("socket", str(caught.exception))
        self.assertNotIn("test-private-key", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_timeout_can_recover_and_retry_after_is_bounded(self) -> None:
        delays: list[float] = []
        rate_limited = FakeResponse(429, {"error": "busy"})
        rate_limited.headers["Retry-After"] = "30"
        session = FakeSession(
            requests.Timeout("temporary"),
            rate_limited,
            FakeResponse(200, {"choices": [{"message": {"content": "recovered"}}]}),
        )
        model = ZhipuChatModel(
            configured_settings(retry_attempts=2),
            session=session,
            sleeper=delays.append,
        )

        self.assertEqual(model.answer("question", []), "recovered")
        self.assertEqual(delays, [0.25, 2.0])

    def test_invalid_json_and_missing_fields_are_protocol_errors(self) -> None:
        invalid_json = ZhipuChatModel(
            configured_settings(),
            session=FakeSession(FakeResponse(200, json_error=ValueError("bad json"))),
            sleeper=lambda _: None,
        )
        missing_content = ZhipuChatModel(
            configured_settings(),
            session=FakeSession(FakeResponse(200, {"choices": [{"message": {}}]})),
            sleeper=lambda _: None,
        )

        with self.assertRaises(ProviderProtocolError):
            invalid_json.answer("问题", [])
        with self.assertRaises(ProviderProtocolError):
            missing_content.answer("问题", [])

    def test_missing_key_is_unavailable_without_network(self) -> None:
        session = FakeSession()
        settings = replace(configured_settings(), api_key=SecretValue(""))
        model = ZhipuChatModel(settings, session=session, sleeper=lambda _: None)

        self.assertFalse(model.available)
        with self.assertRaises(ProviderUnavailableError):
            model.answer("问题", [])
        self.assertEqual(session.calls, [])

    def test_query_plan_uses_json_mode_and_validates_bounded_queries(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"queries":["RAG 评测","RAG 评测","混合检索与引用"]}'
                            }
                        }
                    ]
                },
            )
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)
        queries = model.plan_queries("怎样评估 RAG？", max_queries=2)
        self.assertEqual(queries, ("RAG 评测", "混合检索与引用"))
        self.assertEqual(
            session.calls[0]["json"]["response_format"],
            {"type": "json_object"},
        )

    def test_query_plan_rejects_invalid_json_schema(self) -> None:
        for content in ("not-json", '{"items":["q"]}', '{"queries":[]}'):
            model = ZhipuChatModel(
                configured_settings(),
                session=FakeSession(
                    FakeResponse(200, {"choices": [{"message": {"content": content}}]})
                ),
                sleeper=lambda _: None,
            )
            with self.assertRaises(ProviderProtocolError):
                model.plan_queries("问题", max_queries=2)


class ZhipuWebSearchTests(unittest.TestCase):
    def test_valid_results_are_bounded_and_urls_are_validated(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "search_result": [
                        {
                            "refer": "ref-1",
                            "title": "标题" * 200,
                            "content": "摘要" * 3_000,
                            "link": "https://example.com/article",
                        },
                        {
                            "title": "危险链接",
                            "content": "仍可保留文字来源",
                            "link": "https://example.com/\njavascript:alert(1)",
                        },
                    ]
                },
            )
        )
        search = ZhipuWebSearch(configured_settings(), session=session, sleeper=lambda _: None)

        results = search.search("RAG 最新进展", count=2)

        self.assertEqual(len(results), 2)
        self.assertLessEqual(len(results[0].title), 300)
        self.assertLessEqual(len(results[0].content), 4_000)
        self.assertEqual(results[0].url, "https://example.com/article")
        self.assertEqual(results[1].url, "")
        self.assertEqual(session.calls[0]["timeout"], (1.5, 4.5))

    def test_missing_search_result_and_non_retryable_status_fail_safely(self) -> None:
        missing_field = ZhipuWebSearch(
            configured_settings(),
            session=FakeSession(FakeResponse(200, {"unexpected": []})),
            sleeper=lambda _: None,
        )
        rejected_session = FakeSession(FakeResponse(400, {"error": "private upstream body"}))
        rejected = ZhipuWebSearch(
            configured_settings(), session=rejected_session, sleeper=lambda _: None
        )

        with self.assertRaises(ProviderProtocolError):
            missing_field.search("RAG", count=3)
        with self.assertRaises(ProviderError) as caught:
            rejected.search("RAG", count=3)

        self.assertEqual(len(rejected_session.calls), 1)
        self.assertNotIn("private upstream body", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
