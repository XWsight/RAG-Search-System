from __future__ import annotations

import unittest

from rag_system.provider_errors import ProviderProtocolError


class ProviderErrorContractTests(unittest.TestCase):
    def test_protocol_error_has_bounded_machine_readable_policy(self) -> None:
        error = ProviderProtocolError(
            "safe message",
            code="answer_schema",
            repairable=True,
        )
        self.assertEqual(error.code, "answer_schema")
        self.assertTrue(error.repairable)

        for code in ("UpperCase", "contains-dash", "x" * 65):
            with self.subTest(code=code), self.assertRaises(ValueError):
                ProviderProtocolError("safe", code=code)
        with self.assertRaises(TypeError):
            ProviderProtocolError("safe", repairable=1)
        for message in ("", " leading", "line\nbreak", "x" * 257):
            with self.subTest(message=message), self.assertRaises(ValueError):
                ProviderProtocolError(message)


if __name__ == "__main__":
    unittest.main()
