from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from rag_system.config import SecretValue, Settings


class ConfigurationTests(unittest.TestCase):
    def test_secret_never_appears_in_string_representation(self) -> None:
        secret = SecretValue("very-private-value")
        self.assertNotIn("very-private-value", repr(secret))
        self.assertNotIn("very-private-value", str(secret))
        self.assertEqual(secret.reveal(), "very-private-value")

    def test_defaults_are_valid_and_privacy_preserving(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings().validate()
        self.assertFalse(settings.allow_cloud_default)
        self.assertFalse(settings.allow_web_default)
        self.assertFalse(settings.persist_data)
        self.assertGreater(settings.chunk_size, settings.chunk_overlap)
        self.assertGreaterEqual(settings.max_jobs, settings.job_workers)

    def test_invalid_chunking_and_url_are_rejected(self) -> None:
        settings = Settings()
        with self.assertRaises(ValueError):
            replace(settings, chunk_overlap=settings.chunk_size).validate()
        with self.assertRaises(ValueError):
            replace(settings, chat_url="http://example.com/chat").validate()

    def test_non_finite_timeouts_and_character_bounds_are_rejected(self) -> None:
        settings = Settings()
        for timeout in (float("nan"), float("inf"), 0.0, 301.0):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                replace(settings, read_timeout_seconds=timeout).validate()
        with self.assertRaises(ValueError):
            replace(settings, max_question_characters=0).validate()
        with self.assertRaises(ValueError):
            replace(settings, max_context_characters=999).validate()
        with self.assertRaises(ValueError):
            replace(settings, max_concurrent_answers=0).validate()


if __name__ == "__main__":
    unittest.main()
