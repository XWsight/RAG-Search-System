"""Validated multi-document loading and deterministic text chunking."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_system.config import Settings
from rag_system.domain import Chunk, SourceDocument
from rag_system.loaders import LoaderLimits, SecureDocumentLoader
from rag_system.security import DocumentValidationError
from rag_system.text import stable_digest


_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_BREAK_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "；", "; ", "，", ", ", " ")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    index_id: str
    documents: tuple[SourceDocument, ...]
    chunks: tuple[Chunk, ...]
    duplicate_count: int = 0


class AdaptiveTextSplitter:
    """Character splitter that prefers semantic boundaries and tracks offsets."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: SourceDocument) -> tuple[Chunk, ...]:
        text = document.text
        headings = self._headings(text)
        chunks: list[Chunk] = []
        cursor = 0

        while cursor < len(text):
            proposed_end = min(len(text), cursor + self.chunk_size)
            end = self._preferred_end(text, cursor, proposed_end)
            if end <= cursor:
                end = proposed_end

            leading = len(text[cursor:end]) - len(text[cursor:end].lstrip())
            trailing = len(text[cursor:end].rstrip())
            start_char = cursor + leading
            end_char = cursor + trailing

            if start_char < end_char:
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{stable_digest([document.document_id, str(start_char), str(end_char)])}",
                        document_id=document.document_id,
                        source_name=document.name,
                        text=text[start_char:end_char],
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        heading=self._heading_at(headings, start_char),
                    )
                )

            if end >= len(text):
                break
            next_cursor = max(cursor + 1, end - self.chunk_overlap)
            cursor = next_cursor

        return tuple(chunks)

    def _preferred_end(self, text: str, start: int, proposed_end: int) -> int:
        if proposed_end >= len(text):
            return len(text)

        minimum = min(proposed_end, start + max(80, int(self.chunk_size * 0.55)))
        window = text[minimum:proposed_end]
        best_end = -1
        for separator in _BREAK_SEPARATORS:
            position = window.rfind(separator)
            if position >= 0:
                candidate = minimum + position + len(separator)
                best_end = max(best_end, candidate)
        return best_end if best_end > start else proposed_end

    @staticmethod
    def _headings(text: str) -> tuple[tuple[int, str], ...]:
        return tuple((match.start(), match.group(2).strip()) for match in _HEADING_PATTERN.finditer(text))

    @staticmethod
    def _heading_at(headings: Sequence[tuple[int, str]], position: int) -> str:
        active = ""
        for offset, heading in headings:
            if offset > position:
                break
            active = heading
        return active


class DocumentIngestor:
    """Apply resource limits, deduplicate files, and create a stable index ID."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.validate()
        self.splitter = AdaptiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.loader = SecureDocumentLoader(
            LoaderLimits(
                max_documents=settings.max_documents,
                max_file_bytes=settings.max_file_bytes,
                max_total_file_bytes=settings.max_total_bytes,
                max_uncompressed_bytes=settings.max_archive_uncompressed_bytes,
                max_pages=settings.max_pdf_pages,
                max_characters=settings.max_document_characters,
            )
        )

    def ingest(
        self,
        paths: Sequence[str | Path] | None = None,
        *,
        namespace: str = "",
    ) -> IngestionResult:
        if not isinstance(namespace, str):
            raise TypeError("namespace must be a string")
        namespace = namespace.strip()
        if len(namespace) > 256 or any(ord(character) < 32 for character in namespace):
            raise ValueError("namespace must be at most 256 printable characters")
        resolved_paths = tuple(Path(path) for path in (paths or (self.settings.default_document,)))
        if not resolved_paths:
            raise DocumentValidationError("请至少选择一个文档。")
        if len(resolved_paths) > self.settings.max_documents:
            raise DocumentValidationError(f"一次最多上传 {self.settings.max_documents} 个文档。")

        documents: list[SourceDocument] = []
        seen_hashes: set[str] = set()
        duplicate_count = 0

        for document in self.loader.load(resolved_paths):
            if document.content_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(document.content_hash)
            documents.append(document)

        if not documents:
            raise DocumentValidationError("没有可建立索引的唯一文档。")

        chunks = tuple(chunk for document in documents for chunk in self.splitter.split(document))
        if not chunks:
            raise DocumentValidationError("文档没有产生可检索的文本片段。")
        if len(chunks) > self.settings.max_chunks:
            raise DocumentValidationError(f"文档切分后超过 {self.settings.max_chunks} 个片段。")

        identity_parts = [
            identity
            for document in documents
            for identity in (document.document_id, document.content_hash)
        ]
        index_identity = [
            "schema:v2",
            f"namespace:{namespace}",
            *identity_parts,
            self.settings.embedding_model,
            str(self.settings.chunk_size),
            str(self.settings.chunk_overlap),
        ]
        index_id = f"idx_{stable_digest(index_identity)}"
        return IngestionResult(
            index_id=index_id,
            documents=tuple(documents),
            chunks=chunks,
            duplicate_count=duplicate_count,
        )
