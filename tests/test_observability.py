from __future__ import annotations

import io
import json
import logging
import math
import unittest

from rag_system.observability import JsonEventLogger, TraceContext


def captured_logger() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logger = logging.Logger("structured-test", level=logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream


class StructuredEventTests(unittest.TestCase):
    def test_emits_one_deterministic_json_event_with_allowlisted_fields(self) -> None:
        logger, stream = captured_logger()
        event_logger = JsonEventLogger(logger, clock=lambda: 0.0)
        context = TraceContext(trace_id="trace-1", request_id="request-1")

        record = event_logger.emit(
            "request.completed",
            context=context,
            fields={
                "route": "local",
                "latency_ms": 12.5,
                "document_count": 2,
                "question": "不得进入日志的问题正文",
                "document_text": "不得进入日志的文档正文",
            },
        )

        rendered = stream.getvalue()
        parsed = json.loads(rendered)
        self.assertEqual(rendered.count("\n"), 1)
        self.assertEqual(parsed, record)
        self.assertEqual(parsed["timestamp"], "1970-01-01T00:00:00.000Z")
        self.assertEqual(parsed["trace_id"], "trace-1")
        self.assertEqual(parsed["request_id"], "request-1")
        self.assertEqual(parsed["route"], "local")
        self.assertEqual(parsed["dropped_field_count"], 2)
        self.assertNotIn("问题正文", rendered)
        self.assertNotIn("文档正文", rendered)

    def test_controls_and_known_secrets_are_removed_from_all_string_fields(self) -> None:
        logger, stream = captured_logger()
        event_logger = JsonEventLogger(
            logger,
            known_secrets=("private-key", "secondary"),
            clock=lambda: 1.25,
        )
        context = TraceContext(
            trace_id="trace\nprivate-key",
            request_id="request\tsecondary",
        )

        event_logger.emit(
            "provider\nresponse private-key",
            context=context,
            fields={
                "component": "chat\rprivate-key",
                "outcome": "ok\x00secondary",
            },
            level=logging.WARNING,
        )

        rendered = stream.getvalue()
        parsed = json.loads(rendered)
        self.assertNotIn("private-key", rendered)
        self.assertNotIn("secondary", rendered)
        self.assertNotIn("\x00", rendered)
        self.assertEqual(parsed["level"], "WARNING")
        self.assertIn("[REDACTED]", parsed["component"])

    def test_non_scalar_and_non_finite_fields_are_dropped(self) -> None:
        logger, stream = captured_logger()
        event_logger = JsonEventLogger(logger, clock=lambda: 10.0)

        event_logger.emit(
            "request.metrics",
            context=TraceContext.new(),
            fields={
                "latency_ms": math.nan,
                "component": ["not", "scalar"],
                "cache_hit": False,
            },
        )

        parsed = json.loads(stream.getvalue())
        self.assertNotIn("latency_ms", parsed)
        self.assertNotIn("component", parsed)
        self.assertFalse(parsed["cache_hit"])
        self.assertEqual(parsed["dropped_field_count"], 2)

    def test_secret_is_redacted_before_field_length_is_bounded(self) -> None:
        logger, stream = captured_logger()
        secret = "secret-that-crosses-the-limit"
        event_logger = JsonEventLogger(logger, known_secrets=(secret,), clock=lambda: 2.0)

        event_logger.emit(
            "bounded.field",
            context=TraceContext.new(),
            fields={"outcome": "x" * 240 + secret},
        )

        rendered = stream.getvalue()
        self.assertNotIn(secret, rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_child_request_preserves_trace_and_rotates_request_id(self) -> None:
        parent = TraceContext.new()
        child = parent.child_request()

        self.assertEqual(parent.trace_id, child.trace_id)
        self.assertNotEqual(parent.request_id, child.request_id)
        self.assertTrue(parent.trace_id.startswith("trace_"))
        self.assertTrue(parent.request_id.startswith("request_"))


if __name__ == "__main__":
    unittest.main()
