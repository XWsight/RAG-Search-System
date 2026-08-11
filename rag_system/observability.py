"""Structured, privacy-preserving operational events."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final


ALLOWED_EVENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "cache_hit",
        "chunk_count",
        "citation_count",
        "component",
        "document_count",
        "duration_ms",
        "error_type",
        "http_status",
        "latency_ms",
        "model",
        "operation",
        "outcome",
        "provider",
        "queue_depth",
        "rate_limited",
        "result_count",
        "retry_after_seconds",
        "retry_count",
        "route",
        "source_kind",
        "tenant_hash",
        "token_count",
    }
)
_LEVEL_NAMES: Final[dict[int, str]] = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_REPEATED_WHITESPACE = re.compile(r"\s+")
_INVALID_IDENTIFIER_CHARACTERS = re.compile(r"[^A-Za-z0-9_.:-]+")
_MAX_EVENT_CHARACTERS = 96
_MAX_IDENTIFIER_CHARACTERS = 128
_MAX_FIELD_CHARACTERS = 256


def _new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _clean_string(value: str, *, max_characters: int) -> str:
    cleaned = _CONTROL_CHARACTERS.sub(" ", value)
    cleaned = _REPEATED_WHITESPACE.sub(" ", cleaned).strip()
    return cleaned[:max_characters]


def _clean_identifier(value: str) -> str:
    cleaned = _clean_string(value, max_characters=_MAX_IDENTIFIER_CHARACTERS)
    return _INVALID_IDENTIFIER_CHARACTERS.sub("_", cleaned).strip("_")


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Correlation identifiers shared by all events for one request path."""

    trace_id: str
    request_id: str

    def __post_init__(self) -> None:
        for name, value in (("trace_id", self.trace_id), ("request_id", self.request_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")
            if len(value) > 512:
                raise ValueError(f"{name} is too long")

    @classmethod
    def new(
        cls,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
    ) -> "TraceContext":
        return cls(
            trace_id=trace_id or _new_identifier("trace"),
            request_id=request_id or _new_identifier("request"),
        )

    def child_request(self) -> "TraceContext":
        return TraceContext(trace_id=self.trace_id, request_id=_new_identifier("request"))


class JsonEventLogger:
    """Emit one allowlisted JSON object per log record.

    Question text, document text, source excerpts, headers, and arbitrary error
    messages are deliberately absent from ``ALLOWED_EVENT_FIELDS``.
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        known_secrets: Sequence[str] = (),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._logger = logger
        self._known_secrets = tuple(
            sorted({value for value in known_secrets if value}, key=len, reverse=True)
        )
        self._clock = clock

    def emit(
        self,
        event: str,
        *,
        context: TraceContext,
        fields: Mapping[str, object] | None = None,
        level: int = logging.INFO,
    ) -> dict[str, object]:
        if level not in _LEVEL_NAMES:
            raise ValueError("level must be a standard logging level")
        if not isinstance(event, str):
            raise TypeError("event must be a string")
        safe_event = _clean_string(self._redact(event), max_characters=_MAX_EVENT_CHARACTERS)
        if not safe_event:
            raise ValueError("event cannot be empty")

        timestamp = float(self._clock())
        if not math.isfinite(timestamp):
            raise ValueError("clock returned a non-finite timestamp")

        supplied = dict(fields or {})
        record: dict[str, object] = {
            "event": safe_event,
            "event_version": 1,
            "level": _LEVEL_NAMES[level],
            "request_id": self._safe_context_id(context.request_id, "request"),
            "timestamp": self._format_timestamp(timestamp),
            "trace_id": self._safe_context_id(context.trace_id, "trace"),
        }
        dropped = 0
        for key, value in supplied.items():
            if key not in ALLOWED_EVENT_FIELDS:
                dropped += 1
                continue
            safe_value = self._safe_field_value(value)
            if safe_value is None and value is not None:
                dropped += 1
                continue
            record[key] = safe_value
        if dropped:
            record["dropped_field_count"] = dropped

        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._logger.log(level, payload)
        return record

    def _safe_field_value(self, value: object) -> str | int | float | bool | None:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            return _clean_string(self._redact(value), max_characters=_MAX_FIELD_CHARACTERS)
        return None

    def _safe_context_id(self, value: str, prefix: str) -> str:
        cleaned = _clean_identifier(self._redact(value))
        return cleaned or _new_identifier(prefix)

    def _redact(self, value: str) -> str:
        redacted = value
        for secret in self._known_secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        try:
            rendered = datetime.fromtimestamp(timestamp, tz=UTC).isoformat(
                timespec="milliseconds"
            )
        except (OSError, OverflowError, ValueError):
            raise ValueError("clock returned an unsupported timestamp") from None
        return rendered.replace("+00:00", "Z")


__all__ = ["ALLOWED_EVENT_FIELDS", "JsonEventLogger", "TraceContext"]
