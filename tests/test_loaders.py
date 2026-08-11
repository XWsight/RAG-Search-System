from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from html import escape
from pathlib import Path
from unittest.mock import patch

from rag_system.domain import SourceDocument
from rag_system.loaders import (
    DocumentLoadError,
    LoaderLimits,
    MissingDocumentDependencyError,
    SecureDocumentLoader,
)


def write_docx(
    path: Path,
    paragraphs: list[str],
    *,
    extra_members: dict[str, bytes] | None = None,
) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    ).encode()
    content_types = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        b'<Override PartName="/word/document.xml" '
        b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        b"</Types>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document)
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)


class SecureDocumentLoaderTests(unittest.TestCase):
    def test_utf8_and_gb18030_text_become_stable_source_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            utf8 = root / "知识.txt"
            gb18030 = root / "legacy.md"
            utf8.write_bytes("检索增强生成".encode())
            gb18030.write_bytes("中文旧文档".encode("gb18030"))

            loader = SecureDocumentLoader()
            documents = loader.load([utf8, gb18030])
            repeated = loader.load_one(utf8)

            self.assertEqual(len(documents), 2)
            self.assertTrue(all(isinstance(document, SourceDocument) for document in documents))
            self.assertEqual(documents[0].text, "检索增强生成")
            self.assertEqual(documents[1].text, "中文旧文档")
            self.assertEqual(documents[0].document_id, repeated.document_id)
            self.assertEqual(len(documents[0].content_hash), 64)
            self.assertEqual(documents[1].encoding, "gb18030")

    def test_html_keeps_visible_text_and_discards_script_and_style(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "page.html")
            path.write_text(
                "<html><head><style>.secret{display:none}</style></head>"
                "<body><h1>标题</h1><script>steal()</script>"
                "<p>公开内容 &amp; 来源</p><noscript>隐藏内容</noscript></body></html>",
                encoding="utf-8",
            )

            document = SecureDocumentLoader().load_one(path)

            self.assertIn("标题", document.text)
            self.assertIn("公开内容 & 来源", document.text)
            self.assertNotIn("steal", document.text)
            self.assertNotIn("display:none", document.text)
            self.assertNotIn("隐藏内容", document.text)

    def test_docx_extracts_paragraphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "report.docx")
            write_docx(path, ["第一段", "第二段"])

            document = SecureDocumentLoader().load_one(path)

            self.assertEqual(document.text, "第一段\n\n第二段")
            self.assertEqual(document.encoding, "docx/xml")

    def test_docx_rejects_traversal_and_decompression_bombs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traversal = root / "traversal.docx"
            write_docx(traversal, ["正文"], extra_members={"../escape.txt": b"bad"})

            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader().load_one(traversal)

            bomb = root / "bomb.docx"
            write_docx(bomb, ["A" * 10_000])
            limits = replace(
                LoaderLimits(),
                max_uncompressed_bytes=1_024,
                max_compression_ratio=50.0,
            )
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader(limits).load_one(bomb)

    def test_file_character_paragraph_and_format_limits_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = root / "large.txt"
            oversized.write_text("x" * 20, encoding="utf-8")
            empty = root / "empty.md"
            empty.write_bytes(b"")
            binary = root / "binary.txt"
            binary.write_bytes(b"safe\x00unsafe")
            dangerous = root / "payload.exe"
            dangerous.write_text("text", encoding="utf-8")
            many_paragraphs = root / "paragraphs.md"
            many_paragraphs.write_text("a\n\nb\n\nc", encoding="utf-8")
            too_many_characters = root / "characters.txt"
            too_many_characters.write_text("123456", encoding="utf-8")

            size_limits = replace(
                LoaderLimits(),
                max_file_bytes=10,
                max_total_file_bytes=20,
            )
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader(size_limits).load_one(oversized)
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader().load_one(empty)
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader().load_one(binary)
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader().load_one(dangerous)
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader(replace(LoaderLimits(), max_paragraphs=2)).load_one(
                    many_paragraphs
                )
            with self.assertRaises(DocumentLoadError):
                SecureDocumentLoader(replace(LoaderLimits(), max_characters=5)).load_one(
                    too_many_characters
                )

    def test_allowed_root_prevents_reading_an_outside_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            allowed = base / "allowed"
            allowed.mkdir()
            inside = allowed / "inside.txt"
            outside = base / "outside.txt"
            inside.write_text("inside", encoding="utf-8")
            outside.write_text("outside", encoding="utf-8")
            loader = SecureDocumentLoader(allowed_root=allowed)

            self.assertEqual(loader.load_one(inside).text, "inside")
            with self.assertRaises(DocumentLoadError):
                loader.load_one(outside)

    def test_pdf_dependency_and_page_limit_fail_safely_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "paper.pdf")
            path.write_bytes(b"%PDF-1.4\nminimal")

            with patch.dict(sys.modules, {"pypdf": None}):
                with self.assertRaises(MissingDocumentDependencyError) as caught:
                    SecureDocumentLoader().load_one(path)
            self.assertIn("pypdf", str(caught.exception))

            class Page:
                def extract_text(self) -> str:
                    return "page"

            class Reader:
                is_encrypted = False
                pages = [Page(), Page(), Page()]

            limits = replace(LoaderLimits(), max_pages=2)
            loader = SecureDocumentLoader(limits, pdf_reader_factory=lambda _: Reader())
            with self.assertRaises(DocumentLoadError):
                loader.load_one(path)

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(LoaderLimits(), max_uncompressed_bytes=0).validate()
        with self.assertRaises(ValueError):
            replace(LoaderLimits(), max_compression_ratio=0.5).validate()


if __name__ == "__main__":
    unittest.main()
