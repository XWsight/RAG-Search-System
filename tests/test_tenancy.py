from __future__ import annotations

import dataclasses
import unittest
from concurrent.futures import ThreadPoolExecutor

from rag_system.tenancy import (
    ApiKeyAuthenticator,
    AuthenticationError,
    AuthorizationError,
    Principal,
    ResourceAuthorizer,
    TenantId,
    require_tenant_access,
)


TENANT_A_KEY = "rsk_live_tenant_a_0123456789abcdef"
TENANT_B_KEY = "rsk_live_tenant_b_fedcba9876543210"


def principal(tenant: str, subject: str) -> Principal:
    return Principal(subject, TenantId(tenant), frozenset({"reader"}))


class TenancyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.principal_a = principal("tenant-a", "user-a")
        self.principal_b = principal("tenant-b", "user-b")
        self.authenticator = ApiKeyAuthenticator.from_mapping(
            {
                TENANT_A_KEY: self.principal_a,
                TENANT_B_KEY: self.principal_b,
            }
        )

    def test_tenant_and_principal_are_strict_and_immutable(self) -> None:
        self.assertEqual(str(TenantId("tenant-01")), "tenant-01")
        for invalid in ("A", "UPPER", "-leading", "trailing-", "white space", "../escape"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                TenantId(invalid)

        roles = {"reader"}
        item = Principal("subject-1", TenantId("tenant-01"), roles)
        roles.add("admin")
        self.assertEqual(item.roles, frozenset({"reader"}))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            item.subject = "changed"  # type: ignore[misc]

    def test_authentication_supports_bearer_and_api_key_headers(self) -> None:
        bearer = self.authenticator.authenticate_headers(
            {"Authorization": f"Bearer {TENANT_A_KEY}"}
        )
        api_key = self.authenticator.authenticate_headers(
            {"X-API-Key": TENANT_B_KEY}
        )
        self.assertEqual(bearer, self.principal_a)
        self.assertEqual(api_key, self.principal_b)

    def test_missing_repeated_and_malformed_credentials_are_rejected(self) -> None:
        cases = (
            {},
            {
                "Authorization": f"Bearer {TENANT_A_KEY}",
                "X-API-Key": TENANT_A_KEY,
            },
            {"Authorization": [f"Bearer {TENANT_A_KEY}", f"Bearer {TENANT_A_KEY}"]},
            [
                ("Authorization", f"Bearer {TENANT_A_KEY}"),
                ("authorization", f"Bearer {TENANT_A_KEY}"),
            ],
            {"Authorization": f"Basic {TENANT_A_KEY}"},
            {"Authorization": f"Bearer {TENANT_A_KEY},Bearer {TENANT_B_KEY}"},
            {"X-API-Key": "too-short"},
        )
        for headers in cases:
            with self.subTest(headers_type=type(headers).__name__), self.assertRaises(
                AuthenticationError
            ):
                self.authenticator.authenticate_headers(headers)

    def test_plaintext_is_not_retained_or_leaked(self) -> None:
        representation = repr(self.authenticator)
        record_representations = repr(self.authenticator._records)
        self.assertNotIn(TENANT_A_KEY, representation)
        self.assertNotIn(TENANT_B_KEY, representation)
        self.assertNotIn(TENANT_A_KEY, record_representations)
        self.assertNotIn(TENANT_B_KEY, record_representations)
        self.assertIn("credentials=2", representation)

        attempted_secret = "rsk_live_unknown_0123456789abcdef"
        with self.assertRaises(AuthenticationError) as raised:
            self.authenticator.authenticate(attempted_secret)
        self.assertNotIn(attempted_secret, str(raised.exception))
        self.assertNotIn(attempted_secret, repr(raised.exception))

        stored_values = tuple(record.digest for record in self.authenticator._records)
        self.assertTrue(all(isinstance(value, bytes) and len(value) == 32 for value in stored_values))

    def test_cross_tenant_and_missing_resource_have_identical_denials(self) -> None:
        authorizer = ResourceAuthorizer.from_mapping(
            {
                "index-a": self.principal_a.tenant_id,
                "index-b": self.principal_b.tenant_id,
            }
        )
        self.assertEqual(authorizer.require_access(self.principal_a, "index-a"), "index-a")

        messages: list[str] = []
        for resource_id in ("index-b", "missing-index", "../invalid"):
            with self.assertRaises(AuthorizationError) as raised:
                authorizer.require_access(self.principal_a, resource_id)
            messages.append(str(raised.exception))
            self.assertNotIn(resource_id, str(raised.exception))
        self.assertEqual(len(set(messages)), 1)

        require_tenant_access(self.principal_a, self.principal_a.tenant_id)
        for owner in (self.principal_b.tenant_id, None):
            with self.assertRaises(AuthorizationError) as raised:
                require_tenant_access(self.principal_a, owner)
            self.assertEqual(str(raised.exception), messages[0])

    def test_resource_mapping_is_copied_and_immutable(self) -> None:
        owners = {"index-a": self.principal_a.tenant_id}
        authorizer = ResourceAuthorizer.from_mapping(owners)
        owners["index-b"] = self.principal_b.tenant_id
        with self.assertRaises(AuthorizationError):
            authorizer.require_access(self.principal_b, "index-b")
        with self.assertRaises(TypeError):
            authorizer._owners["index-b"] = self.principal_b.tenant_id  # type: ignore[index]

    def test_concurrent_authentication_is_stable(self) -> None:
        def authenticate(number: int) -> Principal:
            key = TENANT_A_KEY if number % 2 == 0 else TENANT_B_KEY
            return self.authenticator.authenticate(key)

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(authenticate, range(400)))

        self.assertEqual(results.count(self.principal_a), 200)
        self.assertEqual(results.count(self.principal_b), 200)


if __name__ == "__main__":
    unittest.main()
