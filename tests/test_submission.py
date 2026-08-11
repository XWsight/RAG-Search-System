from __future__ import annotations

import io
import unittest

from rag_system.application import (
    PlatformUnavailableError,
    PlatformValidationError,
    UploadDocument,
)
from rag_system.submission import UploadBatchPreparer


class TrackingStream(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.request_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.request_sizes.append(size)
        return super().read(size)


class UploadBatchPreparerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preparer = UploadBatchPreparer(
            max_file_bytes=8,
            max_total_bytes=12,
            max_documents=2,
            document_id_factory=lambda: "1234567890abcdef1234567890abcdef",
        )

    def test_materializes_binary_streams_with_a_bounded_read(self) -> None:
        stream = TrackingStream(b"document")

        prepared = self.preparer.prepare((UploadDocument("guide.txt", stream),))

        self.assertEqual(prepared, (UploadDocument("guide.txt", b"document"),))
        self.assertTrue(stream.request_sizes)
        self.assertLessEqual(max(stream.request_sizes), 9)

    def test_rejects_empty_oversized_and_non_binary_submissions(self) -> None:
        with self.assertRaises(PlatformValidationError):
            self.preparer.prepare(())
        with self.assertRaises(PlatformValidationError):
            self.preparer.prepare((UploadDocument("large.txt", b"123456789"),))
        with self.assertRaises(PlatformValidationError):
            self.preparer.prepare(
                (
                    UploadDocument("a.txt", b"1234567"),
                    UploadDocument("b.txt", b"123456"),
                )
            )
        with self.assertRaises(PlatformValidationError):
            self.preparer.prepare((UploadDocument("bad.txt", io.StringIO("text")),))

    def test_digest_is_order_independent_but_content_sensitive(self) -> None:
        first = UploadDocument("a.txt", b"alpha")
        second = UploadDocument("b.txt", b"beta")

        forward = self.preparer.request_digest("docs", (first, second))
        reverse = self.preparer.request_digest("docs", (second, first))
        changed = self.preparer.request_digest(
            "docs",
            (first, UploadDocument("b.txt", b"changed")),
        )

        self.assertEqual(forward, reverse)
        self.assertNotEqual(forward, changed)

    def test_document_identifier_is_normalized_and_validated(self) -> None:
        self.assertEqual(
            self.preparer.new_document_id(),
            "doc_1234567890abcdef1234567890abcdef",
        )
        invalid = UploadBatchPreparer(
            max_file_bytes=1,
            max_total_bytes=1,
            max_documents=1,
            document_id_factory=lambda: "short",
        )
        with self.assertRaises(PlatformUnavailableError):
            invalid.new_document_id()


if __name__ == "__main__":
    unittest.main()
