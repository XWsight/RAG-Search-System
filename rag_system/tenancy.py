"""Tenant identity, API-key authentication, and resource authorization.

All objects are immutable after construction and are therefore safe for
concurrent reads.  The authenticator retains only SHA-256 digests; raw API
keys are never stored, represented, or included in exceptions.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias


_TENANT_ID_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?")
_SUBJECT_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._:@-]{0,126}[A-Za-z0-9])?")
_ROLE_PATTERN = re.compile(r"[a-z](?:[a-z0-9:_-]{0,62}[a-z0-9])?")
_RESOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._:/-]{0,254}[A-Za-z0-9])?")
_API_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~+/=-]{15,255}")
_BEARER_PATTERN = re.compile(r"Bearer[ \t]+([^\s,]+)", re.IGNORECASE)


class TenancyError(Exception):
    """Base error for authentication and resource authorization."""


class AuthenticationError(TenancyError):
    """A deliberately generic authentication failure."""

    def __init__(self) -> None:
        super().__init__("Authentication failed.")


class AuthorizationError(TenancyError):
    """A denial that does not reveal whether a resource exists."""

    def __init__(self) -> None:
        super().__init__("Resource is unavailable.")


@dataclass(frozen=True, slots=True, order=True)
class TenantId:
    """A validated tenant namespace safe for use in internal identifiers."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("TenantId value must be a string")
        if not 2 <= len(self.value) <= 64 or _TENANT_ID_PATTERN.fullmatch(self.value) is None:
            raise ValueError(
                "TenantId must be 2-64 lowercase letters, numbers, underscores, or hyphens"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated subject scoped to exactly one tenant."""

    subject: str
    tenant_id: TenantId
    roles: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str):
            raise TypeError("Principal subject must be a string")
        if _SUBJECT_PATTERN.fullmatch(self.subject) is None:
            raise ValueError("Principal subject has an invalid format")
        if not isinstance(self.tenant_id, TenantId):
            raise TypeError("Principal tenant_id must be a TenantId")

        frozen_roles = frozenset(self.roles)
        if any(not isinstance(role, str) or _ROLE_PATTERN.fullmatch(role) is None for role in frozen_roles):
            raise ValueError("Principal roles contain an invalid value")
        object.__setattr__(self, "roles", frozen_roles)

    def has_role(self, role: str) -> bool:
        if not isinstance(role, str) or _ROLE_PATTERN.fullmatch(role) is None:
            raise ValueError("Role has an invalid format")
        return role in self.roles


@dataclass(frozen=True, slots=True, repr=False)
class _CredentialRecord:
    digest: bytes = field(repr=False)
    principal: Principal

    def __repr__(self) -> str:
        return f"_CredentialRecord(principal={self.principal!r}, digest=<redacted>)"


HeaderValue: TypeAlias = str | Sequence[str]
Headers: TypeAlias = Mapping[str, HeaderValue] | Iterable[tuple[str, str]]


@dataclass(frozen=True, slots=True, repr=False)
class ApiKeyAuthenticator:
    """Authenticate high-entropy API keys without retaining plaintext values."""

    _records: tuple[_CredentialRecord, ...] = field(repr=False)

    @classmethod
    def from_mapping(cls, credentials: Mapping[str, Principal]) -> ApiKeyAuthenticator:
        """Hash an explicit ``raw key -> principal`` mapping immediately.

        The caller remains responsible for discarding its input mapping.  The
        returned object contains only fixed-size digests and principals.
        """

        if not isinstance(credentials, Mapping):
            raise TypeError("credentials must be a mapping")
        if not credentials:
            raise ValueError("At least one credential is required")

        records: list[_CredentialRecord] = []
        seen_digests: list[bytes] = []
        for raw_key, principal in credentials.items():
            if not isinstance(principal, Principal):
                raise TypeError("Credential values must be Principal instances")
            if not _valid_api_key(raw_key):
                raise ValueError("Credential mapping contains an invalid API key")

            digest = _digest(raw_key)
            if any(hmac.compare_digest(digest, existing) for existing in seen_digests):
                raise ValueError("Credential mapping contains a duplicate API key")
            seen_digests.append(digest)
            records.append(_CredentialRecord(digest=digest, principal=principal))

        return cls(tuple(records))

    def __post_init__(self) -> None:
        if not self._records:
            raise ValueError("At least one credential is required")
        if any(not isinstance(record, _CredentialRecord) for record in self._records):
            raise TypeError("Invalid credential record")
        object.__setattr__(self, "_records", tuple(self._records))

    def __repr__(self) -> str:
        return f"ApiKeyAuthenticator(credentials={len(self._records)})"

    @property
    def credential_count(self) -> int:
        return len(self._records)

    def authenticate(self, api_key: str) -> Principal:
        """Authenticate one already-extracted key.

        Every stored digest is compared using ``hmac.compare_digest`` and the
        loop never returns early.  This avoids key-position timing differences.
        """

        if not _valid_api_key(api_key):
            raise AuthenticationError()
        candidate = _digest(api_key)
        match: Principal | None = None
        for record in self._records:
            if hmac.compare_digest(candidate, record.digest):
                match = record.principal
        if match is None:
            raise AuthenticationError()
        return match

    def authenticate_headers(self, headers: Headers) -> Principal:
        """Accept exactly one Bearer or X-API-Key header credential."""

        authorization_values: list[str] = []
        api_key_values: list[str] = []
        try:
            pairs = _header_pairs(headers)
            for name, value in pairs:
                lowered = name.strip().lower()
                if lowered == "authorization":
                    authorization_values.append(value)
                elif lowered == "x-api-key":
                    api_key_values.append(value)
        except (TypeError, ValueError):
            raise AuthenticationError() from None

        credential_count = len(authorization_values) + len(api_key_values)
        if credential_count != 1:
            raise AuthenticationError()

        if authorization_values:
            value = authorization_values[0].strip()
            match = _BEARER_PATTERN.fullmatch(value)
            if match is None:
                raise AuthenticationError()
            return self.authenticate(match.group(1))

        value = api_key_values[0].strip()
        return self.authenticate(value)


@dataclass(frozen=True, slots=True, repr=False)
class ResourceAuthorizer:
    """Immutable resource ownership map with non-enumerating denials."""

    _owners: Mapping[str, TenantId] = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, owners: Mapping[str, TenantId]) -> ResourceAuthorizer:
        if not isinstance(owners, Mapping):
            raise TypeError("owners must be a mapping")
        normalized: dict[str, TenantId] = {}
        for resource_id, tenant_id in owners.items():
            clean_id = _validate_resource_id(resource_id)
            if not isinstance(tenant_id, TenantId):
                raise TypeError("Resource owners must be TenantId instances")
            if clean_id in normalized:
                raise ValueError("Duplicate resource identifier")
            normalized[clean_id] = tenant_id
        return cls(MappingProxyType(normalized))

    def __post_init__(self) -> None:
        # Copying prevents a caller-owned mutable mapping from changing policy.
        copied = dict(self._owners)
        for resource_id, tenant_id in copied.items():
            _validate_resource_id(resource_id)
            if not isinstance(tenant_id, TenantId):
                raise TypeError("Resource owners must be TenantId instances")
        object.__setattr__(self, "_owners", MappingProxyType(copied))

    def __repr__(self) -> str:
        return f"ResourceAuthorizer(resources={len(self._owners)})"

    def require_access(self, principal: Principal, resource_id: str) -> str:
        """Return the validated ID or raise the same error for all denials."""

        if not isinstance(principal, Principal):
            raise TypeError("principal must be a Principal")
        try:
            clean_id = _validate_resource_id(resource_id)
        except (TypeError, ValueError):
            raise AuthorizationError() from None

        owner = self._owners.get(clean_id)
        if owner is None or not hmac.compare_digest(owner.value, principal.tenant_id.value):
            raise AuthorizationError()
        return clean_id


def require_tenant_access(principal: Principal, owner: TenantId | None) -> None:
    """Validate direct ownership without revealing missing versus foreign data."""

    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    if owner is None or not isinstance(owner, TenantId):
        raise AuthorizationError()
    if not hmac.compare_digest(principal.tenant_id.value, owner.value):
        raise AuthorizationError()


def _digest(api_key: str) -> bytes:
    return hashlib.sha256(api_key.encode("utf-8")).digest()


def _valid_api_key(value: object) -> bool:
    return isinstance(value, str) and _API_KEY_PATTERN.fullmatch(value) is not None


def _validate_resource_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("resource_id must be a string")
    if _RESOURCE_ID_PATTERN.fullmatch(value) is None or ".." in value:
        raise ValueError("resource_id has an invalid format")
    return value


def _header_pairs(headers: Headers) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    if isinstance(headers, Mapping):
        items = headers.items()
        for name, raw_value in items:
            if not isinstance(name, str):
                raise TypeError("Header names must be strings")
            if isinstance(raw_value, str):
                pairs.append((name, raw_value))
                continue
            if not isinstance(raw_value, Sequence):
                raise TypeError("Header values must be strings or sequences")
            for value in raw_value:
                if not isinstance(value, str):
                    raise TypeError("Header values must be strings")
                pairs.append((name, value))
        return tuple(pairs)

    for item in headers:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Headers must contain name/value pairs")
        name, value = item
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("Header names and values must be strings")
        pairs.append((name, value))
    return tuple(pairs)


__all__ = [
    "ApiKeyAuthenticator",
    "AuthenticationError",
    "AuthorizationError",
    "Headers",
    "Principal",
    "ResourceAuthorizer",
    "TenancyError",
    "TenantId",
    "require_tenant_access",
]
