from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_system.security import (
    DocumentValidationError,
    markdown_text,
    read_text_document,
    redact_secrets,
    safe_external_url,
    safe_source_name,
)


class SecurityBoundaryTests(unittest.TestCase):
    def test_read_text_document_supports_utf8_and_gb18030(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            utf8_path = Path(directory, "知识.md")
            utf8_path.write_bytes("向量数据库".encode())
            name, text, encoding = read_text_document(utf8_path, max_bytes=1024)
            self.assertEqual(name, "知识.md")
            self.assertEqual(text, "向量数据库")
            self.assertIn(encoding, {"utf-8-sig", "utf-8"})

            gb_path = Path(directory, "legacy.txt")
            gb_path.write_bytes("中文资料".encode("gb18030"))
            _, gb_text, gb_encoding = read_text_document(gb_path, max_bytes=1024)
            self.assertEqual(gb_text, "中文资料")
            self.assertEqual(gb_encoding, "gb18030")

    def test_read_text_document_rejects_size_extension_and_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory, "large.txt")
            oversized.write_text("12345", encoding="utf-8")
            with self.assertRaises(DocumentValidationError):
                read_text_document(oversized, max_bytes=4)

            unsupported = Path(directory, "payload.exe")
            unsupported.write_text("text", encoding="utf-8")
            with self.assertRaises(DocumentValidationError):
                read_text_document(unsupported, max_bytes=1024)

            empty = Path(directory, "empty.md")
            empty.write_bytes(b"")
            with self.assertRaises(DocumentValidationError):
                read_text_document(empty, max_bytes=1024)

    def test_external_url_rejects_credentials_and_non_http_schemes(self) -> None:
        self.assertEqual(safe_external_url("https://example.com/source"), "https://example.com/source")
        self.assertEqual(safe_external_url("javascript:alert(1)"), "")
        self.assertEqual(safe_external_url("https://example.com/a b"), "")
        self.assertEqual(
            safe_external_url("https://example.com/a_(b)?q=[x]"),
            "https://example.com/a_%28b%29?q=%5Bx%5D",
        )
        self.assertEqual(safe_external_url("file:///etc/passwd"), "")
        self.assertEqual(safe_external_url("https://user:pass@example.com"), "")

    def test_markdown_and_secret_sanitizers(self) -> None:
        self.assertEqual(markdown_text("<script>x</script>"), "&lt;script&gt;x&lt;/script&gt;")
        self.assertEqual(redact_secrets("token=secret", ("secret",)), "token=[REDACTED]")
        self.assertEqual(safe_source_name("folder/name\x00.md"), "name.md")


if __name__ == "__main__":
    unittest.main()
