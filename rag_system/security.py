"""Input boundaries and output sanitizers for uploaded and remote content."""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from urllib.parse import quote, urlparse


ALLOWED_TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class DocumentValidationError(ValueError):
    """Raised when an uploaded document violates an ingestion boundary."""


def safe_source_name(path: str | os.PathLike[str]) -> str:
    """Return a display-only basename with control characters removed."""

    name = Path(path).name
    cleaned = _CONTROL_CHARACTERS.sub("", name).strip()
    return cleaned or "untitled"


def read_text_document(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    allowed_extensions: frozenset[str] = ALLOWED_TEXT_EXTENSIONS,
) -> tuple[str, str, str]:
    """Validate and decode one regular text file.

    Returns ``(safe_source_name, text, encoding)``. The size is checked before
    reading so a large upload cannot be loaded into memory accidentally.
    """

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    candidate = Path(path)
    if candidate.is_symlink():
        raise DocumentValidationError("不支持符号链接文件。")
    if not candidate.is_file():
        raise DocumentValidationError("上传内容不是常规文件。")
    if candidate.suffix.lower() not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        raise DocumentValidationError(f"仅支持文本文件：{supported}")

    size = candidate.stat().st_size
    if size == 0:
        raise DocumentValidationError("文档为空。")
    if size > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise DocumentValidationError(f"单个文档不能超过 {limit_mb:g} MB。")

    raw = candidate.read_bytes()
    for encoding in DEFAULT_ENCODINGS:
        try:
            decoded = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        text = _CONTROL_CHARACTERS.sub("", decoded).strip()
        if not text:
            raise DocumentValidationError("文档没有可用文本。")
        return safe_source_name(candidate), text, encoding

    raise DocumentValidationError("无法识别文档编码，请转换为 UTF-8 后重试。")


def safe_external_url(value: object) -> str:
    """Allow only absolute HTTP(S) URLs for source links."""

    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if len(candidate) > 2_048 or any(character.isspace() for character in candidate):
        return ""
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    return quote(candidate, safe="/:?#@!$&'*+,;=%")


def markdown_text(value: object, *, max_characters: int = 2_000) -> str:
    """Escape untrusted content before rendering it in a Markdown component."""

    text = str(value or "")
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if len(text) > max_characters:
        text = f"{text[: max_characters - 3]}..."
    return html.escape(_CONTROL_CHARACTERS.sub("", text), quote=True)


def redact_secrets(text: str, secrets: tuple[str, ...]) -> str:
    """Replace known non-empty secrets in diagnostic text."""

    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted
