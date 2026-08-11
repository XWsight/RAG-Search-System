from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.scan_secrets import scan


class SecretScanTests(unittest.TestCase):
    def test_detects_credentials_and_allows_explicit_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaked = root / "settings.txt"
            placeholder = root / "example.txt"
            leaked.write_text("API_KEY=" + "A" * 32, encoding="utf-8")
            placeholder.write_text("ZHIPU_API_KEY=your_zhipu_api_key", encoding="utf-8")

            findings = scan((leaked, placeholder), root)

            self.assertEqual(len(findings), 1)
            self.assertIn("settings.txt:1", findings[0])

    def test_forbids_tracked_secret_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / ".env"
            environment.write_text("", encoding="utf-8")
            self.assertIn("forbidden tracked credential file", scan((environment,), root)[0])


if __name__ == "__main__":
    unittest.main()
