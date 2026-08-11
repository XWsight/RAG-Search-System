"""Shared strict primitives for versioned evaluation suite assets."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag_system.evaluation import DatasetValidationError
from rag_system.json_contract import JsonContractError, decode_json_object


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")


class EvaluationSuiteError(DatasetValidationError):  # type: ignore[misc]
    """Raised when a suite manifest or frozen contract is ambiguous."""


def read_json_object(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    try:
        resolved = Path(path).resolve(strict=True)
        payload = decode_json_object(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JsonContractError) as error:
        raise EvaluationSuiteError(f"cannot read {label}: {error}") from error
    return resolved, payload


def exact_fields(
    value: Mapping[Any, Any], expected: frozenset[str], *, location: str
) -> None:
    raw_keys = set(value)
    if not all(isinstance(key, str) for key in raw_keys):
        raise EvaluationSuiteError(f"{location} field names must be strings")
    keys = {key for key in raw_keys if isinstance(key, str)}
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise EvaluationSuiteError(f"{location} missing fields: {', '.join(missing)}")
    if unknown:
        raise EvaluationSuiteError(f"{location} unknown fields: {', '.join(unknown)}")


def identifier(value: object, *, location: str) -> str:
    resolved = bounded_text(value, location=location, minimum=3, maximum=64)
    if _IDENTIFIER.fullmatch(resolved) is None:
        raise EvaluationSuiteError(f"{location} must use lowercase snake_case")
    return resolved


def bounded_text(
    value: object, *, location: str, minimum: int, maximum: int
) -> str:
    if not isinstance(value, str):
        raise EvaluationSuiteError(f"{location} must be a string")
    resolved = value.strip()
    if not minimum <= len(resolved) <= maximum:
        raise EvaluationSuiteError(
            f"{location} length must be between {minimum} and {maximum} characters"
        )
    return resolved


def enum_value(value: object, allowed: tuple[str, ...], *, location: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise EvaluationSuiteError(f"{location} must be one of: {', '.join(allowed)}")
    return value


def positive_int(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationSuiteError(f"{location} must be a positive integer")
    return value


def normalized_text_fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def canonical_bundle_digest(
    manifest: Mapping[str, Any],
    *,
    artifacts: Mapping[str, bytes] | None = None,
    artifact_field: str = "artifact_sha256",
) -> str:
    if not artifact_field:
        raise EvaluationSuiteError("artifact digest field cannot be empty")
    artifact_hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted((artifacts or {}).items())
    }
    canonical = json.dumps(
        {"manifest": manifest, artifact_field: artifact_hashes},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_frozen_contract(
    summary: Mapping[str, Any],
    path: str | Path,
    *,
    fields: frozenset[str],
    label: str,
) -> None:
    _resolved, payload = read_json_object(path, label=label)
    exact_fields(payload, fields, location=label)
    if payload.get("schema_version") != 1:
        raise EvaluationSuiteError(f"{label} schema_version must be 1")
    mismatches = [
        field
        for field in sorted(fields - {"schema_version"})
        if payload[field] != summary[field]
    ]
    if mismatches:
        raise EvaluationSuiteError(f"{label} mismatch: " + ", ".join(mismatches))


__all__ = [
    "EvaluationSuiteError",
    "bounded_text",
    "canonical_bundle_digest",
    "enum_value",
    "exact_fields",
    "identifier",
    "normalized_text_fingerprint",
    "positive_int",
    "read_json_object",
    "validate_frozen_contract",
]
