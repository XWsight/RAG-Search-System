"""Strict JSON decoding primitives for untrusted protocol payloads."""

from __future__ import annotations

import json
from typing import Any


class JsonContractError(ValueError):
    """Raised when JSON cannot be decoded into one unambiguous object."""


def decode_json_object(content: str) -> dict[str, Any]:
    """Decode an exact JSON object while rejecting duplicates and NaN values."""

    if not isinstance(content, str):
        raise JsonContractError("JSON content must be a string")
    try:
        payload = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise JsonContractError("invalid JSON object") from None
    if not isinstance(payload, dict):
        raise JsonContractError("JSON root must be an object")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError("duplicate JSON key")
        resolved[key] = value
    return resolved


def _reject_constant(value: str) -> None:
    del value
    raise ValueError("non-finite JSON number")


__all__ = ["JsonContractError", "decode_json_object"]
