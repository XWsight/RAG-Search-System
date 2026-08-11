"""Stable failure contract shared by application code and provider adapters."""

from __future__ import annotations

import re


_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class ProviderError(RuntimeError):
    """Base class for sanitized failures at an external-provider boundary."""


class ProviderAuthenticationError(ProviderError):
    """The configured credential was rejected by an upstream service."""


class ProviderRateLimitError(ProviderError):
    """An upstream service remained rate limited after bounded retries."""


class ProviderUnavailableError(ProviderError):
    """An upstream service could not be reached or remained unavailable."""


class ProviderProtocolError(ProviderError):
    """An upstream response violated a bounded, documented protocol.

    ``repairable`` is deliberately separate from transport retryability. It
    means that one fresh model response may repair a malformed structured
    answer; it never authorizes an unbounded retry loop.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_protocol_error",
        repairable: bool = False,
    ) -> None:
        if (
            not isinstance(message, str)
            or not 1 <= len(message) <= 256
            or message.strip() != message
            or _CONTROL_CHARACTERS.search(message)
        ):
            raise ValueError("provider error message is invalid")
        if _ERROR_CODE.fullmatch(code) is None:
            raise ValueError("provider error code is invalid")
        if not isinstance(repairable, bool):
            raise TypeError("repairable must be a boolean")
        super().__init__(message)
        self.code = code
        self.repairable = repairable


__all__ = [
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderProtocolError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
]
