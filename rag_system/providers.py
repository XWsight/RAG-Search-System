"""Reliable, privacy-conscious adapters for the Zhipu HTTP APIs."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import requests

from rag_system.answer_protocol import AnswerProtocol, GroundedAnswerProtocol
from rag_system.config import Settings
from rag_system.domain import GeneratedAnswer, WebSearchResult
from rag_system.grounding import MAX_GROUNDED_ANSWER_CHARACTERS
from rag_system.json_contract import JsonContractError, decode_json_object
from rag_system.provider_errors import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from rag_system.security import safe_external_url


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_SEARCH_QUERY_CHARACTERS = 70
_MAX_RESULT_ID_CHARACTERS = 96
_MAX_TITLE_CHARACTERS = 300
_MAX_CONTENT_CHARACTERS = 4_000
_MAX_URL_CHARACTERS = 2_048


class _ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def json(self) -> object: ...


class _SessionLike(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
        timeout: tuple[float, float],
    ) -> _ResponseLike: ...

    def close(self) -> None: ...


def _clean_text(value: object, *, max_characters: int) -> str:
    """Normalize an untrusted response field and cap its memory/display cost."""

    if not isinstance(value, str) or max_characters < 1:
        return ""
    normalized = _CONTROL_CHARACTERS.sub("", value).strip()
    if len(normalized) <= max_characters:
        return normalized
    if max_characters <= 3:
        return normalized[:max_characters]
    return f"{normalized[: max_characters - 3]}..."


def _clean_url(value: object) -> str:
    """Reject rather than mutate suspicious or overlong source URLs."""

    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if len(candidate) > _MAX_URL_CHARACTERS or any(character.isspace() for character in candidate):
        return ""
    return safe_external_url(candidate)


class _ZhipuHTTPClient:
    """Shared HTTP behavior with strict retry and response boundaries."""

    def __init__(
        self,
        settings: Settings,
        *,
        session: _SessionLike | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings.validate()
        self._provided_session = session
        self._thread_local = threading.local()
        self._sessions: list[_SessionLike] = []
        self._session_lock = threading.RLock()
        self._closed = False
        self._sleeper = sleeper

    @property
    def available(self) -> bool:
        return bool(self._settings.api_key)

    def close(self) -> None:
        with self._session_lock:
            if self._closed:
                return
            self._closed = True
            unique_sessions: list[_SessionLike] = []
            seen_sessions: set[int] = set()
            for session in (self._provided_session, *self._sessions):
                if session is not None and id(session) not in seen_sessions:
                    seen_sessions.add(id(session))
                    unique_sessions.append(session)
            sessions = tuple(unique_sessions)
            self._sessions.clear()
        for session in sessions:
            session.close()

    def _session_for_request(self) -> _SessionLike:
        with self._session_lock:
            if self._closed:
                raise ProviderUnavailableError("云端服务连接池已关闭。")
            if self._provided_session is not None:
                return self._provided_session
            session = getattr(self._thread_local, "session", None)
            if session is None:
                session = requests.Session()
                self._thread_local.session = session
                self._sessions.append(session)
            return session

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.available:
            raise ProviderUnavailableError("未配置智谱 API Key，云端服务不可用。")

        headers = {
            "Authorization": f"Bearer {self._settings.api_key.reveal()}",
            "Content-Type": "application/json",
        }
        timeout = (
            self._settings.connect_timeout_seconds,
            self._settings.read_timeout_seconds,
        )

        response: _ResponseLike | None = None
        session = self._session_for_request()
        for attempt in range(self._settings.retry_attempts + 1):
            try:
                response = session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt < self._settings.retry_attempts:
                    self._sleeper(self._retry_delay(attempt, None))
                    continue
                raise ProviderUnavailableError("云端服务连接失败，请稍后重试。") from None
            except requests.RequestException:
                raise ProviderUnavailableError("无法连接云端服务，请稍后重试。") from None

            status_code = response.status_code
            if status_code not in _RETRYABLE_STATUS_CODES:
                break
            if attempt >= self._settings.retry_attempts:
                if status_code == 429:
                    raise ProviderRateLimitError("云端服务当前请求过多，请稍后重试。")
                raise ProviderUnavailableError("云端服务暂时不可用，请稍后重试。")

            self._sleeper(self._retry_delay(attempt, response))

        if response is None:  # Defensive guard for type checkers and future edits.
            raise ProviderUnavailableError("云端服务暂时不可用，请稍后重试。")
        if response.status_code in {401, 403}:
            raise ProviderAuthenticationError("智谱 API Key 无效或无权访问该服务。")
        if not 200 <= response.status_code < 300:
            raise ProviderError(f"云端服务拒绝了请求（HTTP {response.status_code}）。")

        try:
            data = response.json()
        except (TypeError, ValueError):
            raise ProviderProtocolError("云端服务返回了无法解析的数据。") from None
        if not isinstance(data, dict):
            raise ProviderProtocolError("云端服务返回的数据结构不正确。")
        return data

    @staticmethod
    def _retry_delay(attempt: int, response: _ResponseLike | None) -> float:
        delay = min(0.25 * (2**attempt), 2.0)
        if response is None or response.status_code != 429:
            return delay
        raw_retry_after = response.headers.get("Retry-After", "")
        try:
            retry_after = float(raw_retry_after)
        except (TypeError, ValueError):
            return delay
        if retry_after < 0:
            return delay
        return max(delay, min(retry_after, 2.0))


class ZhipuChatModel(_ZhipuHTTPClient):
    """Grounded chat completion adapter implementing ``ChatModel``."""

    def __init__(
        self,
        settings: Settings,
        *,
        answer_protocol: AnswerProtocol | None = None,
        session: _SessionLike | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(settings, session=session, sleeper=sleeper)
        self._answer_protocol = answer_protocol or GroundedAnswerProtocol(
            max_context_characters=self._settings.max_context_characters
        )

    def answer(
        self,
        question: str,
        evidence: Sequence[tuple[str, str]],
    ) -> GeneratedAnswer:
        normalized_question = question.strip() if isinstance(question, str) else ""
        if not normalized_question:
            raise ValueError("question cannot be empty")
        if len(normalized_question) > self._settings.max_question_characters:
            raise ValueError("question exceeds the configured character limit")

        prepared = self._answer_protocol.prepare(normalized_question, evidence)
        messages = [message.to_payload() for message in prepared.messages]
        request_payload = {
            "model": self._settings.chat_model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "do_sample": False,
            "max_tokens": self._settings.answer_max_tokens,
            "response_format": {"type": "json_object"},
        }

        for contract_attempt in range(2):
            try:
                data = self._post_json(self._settings.chat_url, request_payload)
                content = self._message_content(data, operation="answer")
                return self._answer_protocol.decode(content, prepared.citation_ids)
            except ProviderProtocolError as error:
                if contract_attempt == 1 or not error.repairable:
                    raise
                messages.append(self._answer_protocol.repair_message().to_payload())
        raise ProviderProtocolError(
            "云端模型回答未通过证据契约校验。",
            code="answer_grounding_contract",
        )

    def plan_queries(self, question: str, *, max_queries: int) -> tuple[str, ...]:
        """Create a bounded JSON query plan for complex or multi-hop questions."""

        normalized_question = question.strip() if isinstance(question, str) else ""
        if not normalized_question:
            raise ValueError("question cannot be empty")
        if len(normalized_question) > self._settings.max_question_characters:
            raise ValueError("question exceeds the configured character limit")
        if not 1 <= max_queries <= 6:
            raise ValueError("max_queries must be between 1 and 6")

        data = self._post_json(
            self._settings.chat_url,
            {
                "model": self._settings.chat_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "把问题拆成用于资料检索的短查询。只返回 JSON 对象，格式为 "
                            '{"queries":["查询1","查询2"]}。不得超过给定数量；每项不超过70字符；'
                            "不回答问题，不生成工具或指令。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": normalized_question, "max_queries": max_queries},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "thinking": {"type": "disabled"},
                "do_sample": False,
                "max_tokens": self._settings.query_plan_max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        content = self._message_content(data, operation="query_plan")
        try:
            payload = decode_json_object(content)
        except JsonContractError:
            raise ProviderProtocolError(
                "云端模型返回了无效的查询计划。",
                code="query_plan_invalid_json",
            ) from None
        if set(payload) != {"queries"}:
            raise ProviderProtocolError("云端模型返回的查询计划结构不正确。")
        queries = payload["queries"]
        if not isinstance(queries, list) or not queries:
            raise ProviderProtocolError("云端模型没有返回可用的查询计划。")

        resolved: list[str] = []
        seen: set[str] = set()
        for raw_query in queries:
            query = _clean_text(raw_query, max_characters=_MAX_SEARCH_QUERY_CHARACTERS)
            identity = query.casefold()
            if not query or identity in seen:
                continue
            resolved.append(query)
            seen.add(identity)
            if len(resolved) >= max_queries:
                break
        if not resolved:
            raise ProviderProtocolError("云端模型没有返回可用的查询计划。")
        return tuple(resolved)

    @staticmethod
    def _message_content(data: Mapping[str, Any], *, operation: str) -> str:
        try:
            choices = data["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            finish_reason = choice.get("finish_reason")
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderProtocolError(
                "云端模型响应缺少回答字段。",
                code=f"{operation}_missing_content",
                repairable=operation == "answer",
            ) from error
        if finish_reason == "length":
            raise ProviderProtocolError(
                "云端模型输出达到长度限制。",
                code=f"{operation}_output_truncated",
                repairable=operation == "answer",
            )
        if finish_reason not in {None, "stop"}:
            raise ProviderProtocolError(
                "云端模型未正常完成输出。",
                code=f"{operation}_incomplete",
            )
        resolved = _clean_text(content, max_characters=MAX_GROUNDED_ANSWER_CHARACTERS)
        if not resolved:
            raise ProviderProtocolError(
                "云端模型返回了空回答。",
                code=f"{operation}_empty_content",
                repairable=operation == "answer",
            )
        return resolved


class ZhipuWebSearch(_ZhipuHTTPClient):
    """Validated adapter implementing ``WebSearchProvider``."""

    def search(self, query: str, *, count: int) -> Sequence[WebSearchResult]:
        normalized_query = query.strip() if isinstance(query, str) else ""
        if not normalized_query:
            raise ValueError("query cannot be empty")
        if len(normalized_query) > _MAX_SEARCH_QUERY_CHARACTERS:
            raise ValueError("query cannot exceed 70 characters")
        if not 1 <= count <= 50:
            raise ValueError("count must be between 1 and 50")

        data = self._post_json(
            self._settings.search_url,
            {
                "search_query": normalized_query,
                "search_engine": "search_std",
                "search_intent": False,
                "count": count,
                "content_size": "medium",
            },
        )
        if "search_result" not in data or not isinstance(data["search_result"], list):
            raise ProviderProtocolError("云端搜索响应缺少结果字段。")

        results: list[WebSearchResult] = []
        for index, item in enumerate(data["search_result"][:count], start=1):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title"), max_characters=_MAX_TITLE_CHARACTERS)
            content = _clean_text(item.get("content"), max_characters=_MAX_CONTENT_CHARACTERS)
            url = _clean_url(item.get("link"))
            if not any((title, content, url)):
                continue
            result_id = _clean_text(
                item.get("refer"), max_characters=_MAX_RESULT_ID_CHARACTERS
            ) or f"web-{index}"
            results.append(
                WebSearchResult(
                    result_id=result_id,
                    title=title or "未命名来源",
                    content=content,
                    url=url,
                )
            )
        return tuple(results)


__all__ = [
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderProtocolError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
    "ZhipuChatModel",
    "ZhipuWebSearch",
]
