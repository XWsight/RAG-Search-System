from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from rag_system.bootstrap import StorageRootLease, parse_api_credentials
from rag_system.config import SecretValue


class BootstrapTests(unittest.TestCase):
    def test_credentials_are_strict_and_raw_keys_are_not_retained_in_repr(self) -> None:
        raw_key = "0123456789abcdef0123456789abcdef"
        encoded = SecretValue(
            '{"'
            + raw_key
            + '":{"subject":"operator-1","tenant_id":"tenant-a",'
            '"roles":["reader","writer","operator"]}}'
        )
        authenticator, principals = parse_api_credentials(encoded)

        self.assertEqual(authenticator.authenticate(raw_key), principals[0])
        self.assertNotIn(raw_key, repr(authenticator))
        self.assertEqual(principals[0].tenant_id.value, "tenant-a")

    def test_duplicate_json_keys_and_unknown_fields_are_rejected(self) -> None:
        key = "0123456789abcdef"
        duplicate = SecretValue(
            f'{{"{key}":{{"subject":"user-1","tenant_id":"tenant-a",'
            f'"roles":["reader"]}},"{key}":{{"subject":"user-2",'
            '"tenant_id":"tenant-b","roles":["reader"]}}}'
        )
        unknown = SecretValue(
            f'{{"{key}":{{"subject":"user-1","tenant_id":"tenant-a",'
            '"roles":["reader"],"extra":true}}}'
        )

        with self.assertRaisesRegex(ValueError, "RAG_API_KEYS_JSON"):
            parse_api_credentials(duplicate)
        with self.assertRaisesRegex(ValueError, "RAG_API_KEYS_JSON"):
            parse_api_credentials(unknown)

    def test_storage_root_lease_rejects_a_second_process_slot_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = StorageRootLease.acquire(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    StorageRootLease.acquire(root)
            finally:
                first.close()

            second = StorageRootLease.acquire(root)
            second.close()


if __name__ == "__main__":
    unittest.main()
