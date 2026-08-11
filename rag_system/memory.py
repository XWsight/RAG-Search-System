"""Thread-safe, bounded, in-memory conversation history.

The store is intentionally process-local and non-persistent.  Each session is
isolated by its caller-provided ID, inactive sessions expire, and an LRU bound
prevents unbounded memory growth.  History rendering is deterministic and does
not call an external model.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _truncate_prefix(value: str, limit: int) -> str:
    """Return a deterministic prefix whose length never exceeds ``limit``."""

    if len(value) <= limit:
        return value
    if limit <= 1:
        return "…"[:limit]
    return f"{value[: limit - 1]}…"


def _fit_pair(question: str, answer: str, limit: int) -> tuple[str, str]:
    """Fit one complete turn inside the aggregate character budget."""

    if len(question) + len(answer) <= limit:
        return question, answer

    # Preserve both sides when possible.  A short question gives its unused
    # share to the answer; otherwise the question receives one third.
    question_budget = min(len(question), max(1, limit // 3))
    answer_budget = limit - question_budget
    if len(answer) < answer_budget:
        answer_budget = len(answer)
        question_budget = limit - answer_budget
    if answer_budget < 1:
        answer_budget = 1
        question_budget = max(1, limit - 1)

    return (
        _truncate_prefix(question, question_budget),
        _truncate_prefix(answer, answer_budget),
    )


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One immutable question/answer pair in a session."""

    question: str
    answer: str
    sequence: int
    created_at: float

    @property
    def character_count(self) -> int:
        return len(self.question) + len(self.answer)


@dataclass(slots=True)
class _SessionBuffer:
    turns: list[ConversationTurn] = field(default_factory=list)
    character_count: int = 0
    last_accessed_at: float = 0.0
    next_sequence: int = 1


class ConversationMemory:
    """A privacy-preserving session memory with TTL and LRU eviction.

    TTL uses a monotonic clock.  Reading a live session counts as activity and
    refreshes both its TTL and its position in the LRU order.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 32,
        ttl_seconds: float = 3_600.0,
        max_rounds: int = 8,
        max_characters: int = 6_000,
        summary_max_characters: int = 3_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if max_characters < 2:
            raise ValueError("max_characters must be at least 2")
        if summary_max_characters < 1:
            raise ValueError("summary_max_characters must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._max_sessions = max_sessions
        self._ttl_seconds = float(ttl_seconds)
        self._max_rounds = max_rounds
        self._max_characters = max_characters
        self._summary_max_characters = summary_max_characters
        self._clock = clock
        self._sessions: OrderedDict[str, _SessionBuffer] = OrderedDict()
        self._lock = RLock()

    def add_turn(self, session_id: str, question: str, answer: str) -> ConversationTurn:
        """Append a turn and return the exact bounded value that was stored."""

        session_key = _require_text(session_id, "session_id")
        clean_question = _require_text(question, "question")
        clean_answer = _require_text(answer, "answer")
        clean_question, clean_answer = _fit_pair(
            clean_question,
            clean_answer,
            self._max_characters,
        )

        with self._lock:
            now = float(self._clock())
            self._purge_expired_locked(now)
            buffer = self._sessions.get(session_key)
            if buffer is None:
                self._evict_for_capacity_locked()
                buffer = _SessionBuffer(last_accessed_at=now)
                self._sessions[session_key] = buffer

            turn = ConversationTurn(
                question=clean_question,
                answer=clean_answer,
                sequence=buffer.next_sequence,
                created_at=now,
            )
            buffer.next_sequence += 1
            buffer.turns.append(turn)
            buffer.character_count += turn.character_count
            buffer.last_accessed_at = now

            while (
                len(buffer.turns) > self._max_rounds
                or buffer.character_count > self._max_characters
            ):
                removed = buffer.turns.pop(0)
                buffer.character_count -= removed.character_count

            self._sessions.move_to_end(session_key)
            return turn

    def history(self, session_id: str) -> tuple[ConversationTurn, ...]:
        """Return an immutable snapshot for one session, or an empty tuple."""

        session_key = _require_text(session_id, "session_id")
        with self._lock:
            now = float(self._clock())
            self._purge_expired_locked(now)
            buffer = self._sessions.get(session_key)
            if buffer is None:
                return ()
            self._touch_locked(session_key, buffer, now)
            return tuple(buffer.turns)

    def summarize(self, session_id: str, *, max_characters: int | None = None) -> str:
        """Render recent history for safe concatenation into a new query.

        The newest rounds are preferred when the rendering budget is smaller
        than the stored history.  No provider call or persistence is involved.
        """

        limit = self._summary_max_characters if max_characters is None else max_characters
        if limit < 1:
            raise ValueError("max_characters must be positive")

        turns = self.history(session_id)
        if not turns:
            return ""

        blocks = [
            f"第 {turn.sequence} 轮\n问题：{turn.question}\n回答：{turn.answer}"
            for turn in turns
        ]
        header = "历史对话（仅作为上下文）：\n"
        if len(header) >= limit:
            return _truncate_prefix(header, limit)

        available = limit - len(header)
        selected: list[str] = []
        used = 0
        for block in reversed(blocks):
            separator_length = 2 if selected else 0
            if used + separator_length + len(block) <= available:
                selected.append(block)
                used += separator_length + len(block)
                continue
            if not selected:
                selected.append(_truncate_prefix(block, available))
            break

        body = "\n\n".join(reversed(selected))
        return f"{header}{body}"[:limit]

    def contextualize_query(
        self,
        session_id: str,
        question: str,
        *,
        max_history_characters: int | None = None,
    ) -> str:
        """Combine deterministic history with a current standalone question."""

        clean_question = _require_text(question, "question")
        summary = self.summarize(session_id, max_characters=max_history_characters)
        if not summary:
            return clean_question
        return f"{summary}\n\n当前问题：{clean_question}"

    def clear(self, session_id: str) -> bool:
        """Delete exactly one session and report whether it existed."""

        session_key = _require_text(session_id, "session_id")
        with self._lock:
            self._purge_expired_locked(float(self._clock()))
            return self._sessions.pop(session_key, None) is not None

    def clear_all(self) -> int:
        """Delete every in-process session and return the number removed."""

        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count

    def has_session(self, session_id: str) -> bool:
        session_key = _require_text(session_id, "session_id")
        with self._lock:
            self._purge_expired_locked(float(self._clock()))
            return session_key in self._sessions

    def stats(self) -> dict[str, int]:
        """Return aggregate counters only; session identifiers are omitted."""

        with self._lock:
            self._purge_expired_locked(float(self._clock()))
            return {
                "sessions": len(self._sessions),
                "turns": sum(len(buffer.turns) for buffer in self._sessions.values()),
                "characters": sum(
                    buffer.character_count for buffer in self._sessions.values()
                ),
            }

    def _touch_locked(
        self,
        session_id: str,
        buffer: _SessionBuffer,
        now: float,
    ) -> None:
        buffer.last_accessed_at = now
        self._sessions.move_to_end(session_id)

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, buffer in self._sessions.items()
            if now - buffer.last_accessed_at >= self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _evict_for_capacity_locked(self) -> None:
        while len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)


__all__ = ["ConversationMemory", "ConversationTurn"]
