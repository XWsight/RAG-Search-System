from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Any

import requests

from rag_system.config import SecretValue, Settings
from rag_system.domain import AnswerClaim, GeneratedAnswer
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
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"claims":[{"text":"RAG 是检索增强生成。",'
                                    '"citation_ids":["L1"]}],"insufficient":false}'
                                )
                            }
                        }
                    ]
                },
            )
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        answer = model.answer("什么是 RAG？", [("L1", "RAG 会先检索资料。")])

        self.assertEqual(
            answer,
            GeneratedAnswer((AnswerClaim("RAG 是检索增强生成。", ("L1",)),)),
        )
        self.assertTrue(model.available)
        self.assertEqual(session.calls[0]["timeout"], (1.5, 4.5))
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer test-private-key")
        messages = session.calls[0]["json"]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("不可信", messages[0]["content"])
        self.assertIn("直接回答 question", messages[0]["content"])
        self.assertEqual(
            session.calls[0]["json"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            session.calls[0]["json"]["thinking"],
            {"type": "disabled"},
        )
        self.assertIs(session.calls[0]["json"]["do_sample"], False)
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 4_096)
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
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"claims":[],"insufficient":true}'
                            }
                        }
                    ]
                },
            ),
        )
        model = ZhipuChatModel(configured_settings(retry_attempts=1), session=session, sleeper=delays.append)

        self.assertEqual(model.answer("问题", []), GeneratedAnswer((), insufficient=True))
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
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"claims":[],"insufficient":true}'
                            }
                        }
                    ]
                },
            ),
        )
        model = ZhipuChatModel(
            configured_settings(retry_attempts=2),
            session=session,
            sleeper=delays.append,
        )

        self.assertEqual(model.answer("question", []), GeneratedAnswer((), insufficient=True))
        self.assertEqual(delays, [0.25, 2.0])

    def test_invalid_json_and_missing_fields_are_protocol_errors(self) -> None:
        invalid_json = ZhipuChatModel(
            configured_settings(),
            session=FakeSession(FakeResponse(200, json_error=ValueError("bad json"))),
            sleeper=lambda _: None,
        )
        missing_content = ZhipuChatModel(
            configured_settings(),
            session=FakeSession(
                FakeResponse(200, {"choices": [{"message": {}}]}),
                FakeResponse(200, {"choices": [{"message": {}}]}),
            ),
            sleeper=lambda _: None,
        )

        with self.assertRaises(ProviderProtocolError):
            invalid_json.answer("问题", [])
        with self.assertRaises(ProviderProtocolError):
            missing_content.answer("问题", [])

    def test_grounded_answer_contract_rejects_unsupported_or_uncited_claims(self) -> None:
        invalid_answers = (
            '{"claims":[{"text":"无引用结论","citation_ids":[]}],"insufficient":false}',
            '{"claims":[{"text":"越界结论","citation_ids":["W9"]}],"insufficient":false}',
            '{"claims":[{"text":"重复键","text":"冲突","citation_ids":["L1"]}],'
            '"insufficient":false}',
            '{"claims":[],"insufficient":false}',
        )
        for content in invalid_answers:
            with self.subTest(content=content):
                model = ZhipuChatModel(
                    configured_settings(),
                    session=FakeSession(
                        FakeResponse(200, {"choices": [{"message": {"content": content}}]}),
                        FakeResponse(200, {"choices": [{"message": {"content": content}}]}),
                    ),
                    sleeper=lambda _: None,
                )
                with self.assertRaises(ProviderProtocolError):
                    model.answer("问题", [("L1", "证据")])

    def test_grounded_answer_rejects_overlong_claim_instead_of_truncating_it(self) -> None:
        content = json.dumps(
            {
                "claims": [{"text": "字" * 2_001, "citation_ids": ["L1"]}],
                "insufficient": False,
            },
            ensure_ascii=False,
        )
        model = ZhipuChatModel(
            configured_settings(),
            session=FakeSession(
                FakeResponse(200, {"choices": [{"message": {"content": content}}]}),
                FakeResponse(200, {"choices": [{"message": {"content": content}}]}),
            ),
            sleeper=lambda _: None,
        )

        with self.assertRaises(ProviderProtocolError) as caught:
            model.answer("问题", [("L1", "证据")])

        self.assertEqual(caught.exception.code, "answer_grounding_contract")

    def test_grounded_answer_retries_one_protocol_failure_without_echoing_it(self) -> None:
        session = FakeSession(
            FakeResponse(200, {"choices": [{"message": {"content": "untrusted-invalid"}}]}),
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"claims":[{"text":"有效结论。",'
                                    '"citation_ids":["L1"]}],"insufficient":false}'
                                )
                            }
                        }
                    ]
                },
            ),
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)
        answer = model.answer("问题", [("L1", "证据")])
        self.assertEqual(answer.claims[0].text, "有效结论。")
        self.assertEqual(len(session.calls), 2)
        retry_messages = session.calls[1]["json"]["messages"]
        self.assertNotIn("untrusted-invalid", str(retry_messages))
        self.assertIn("上一次输出未通过", retry_messages[-1]["content"])

    def test_truncated_grounded_answer_is_retried_once_and_can_recover(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": ""},
                        }
                    ]
                },
            ),
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    '{"claims":[{"text":"有效结论。",'
                                    '"citation_ids":["L1"]}],"insufficient":false}'
                                )
                            },
                        }
                    ]
                },
            ),
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        answer = model.answer("问题", [("L1", "证据")])

        self.assertEqual(answer.claims[0].text, "有效结论。")
        self.assertEqual(len(session.calls), 2)

    def test_repeated_truncation_fails_with_stable_error_code(self) -> None:
        truncated = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": ""},
                }
            ]
        }
        session = FakeSession(FakeResponse(200, truncated), FakeResponse(200, truncated))
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        with self.assertRaises(ProviderProtocolError) as caught:
            model.answer("问题", [("L1", "证据")])

        self.assertEqual(caught.exception.code, "answer_output_truncated")
        self.assertEqual(len(session.calls), 2)

    def test_non_repairable_finish_reason_is_not_retried(self) -> None:
        session = FakeSession(
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "sensitive",
                            "message": {"content": ""},
                        }
                    ]
                },
            )
        )
        model = ZhipuChatModel(configured_settings(), session=session, sleeper=lambda _: None)

        with self.assertRaises(ProviderProtocolError) as caught:
            model.answer("问题", [("L1", "证据")])

        self.assertEqual(caught.exception.code, "answer_incomplete")
        self.assertEqual(len(session.calls), 1)

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
        self.assertEqual(
            session.calls[0]["json"]["thinking"],
            {"type": "disabled"},
        )
        self.assertIs(session.calls[0]["json"]["do_sample"], False)

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
