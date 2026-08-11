from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Lock

from rag_system.idempotency import (
    IdempotencyCapacityError,
    IdempotencyConflictError,
    IdempotencySchemaError,
    IdempotencyStore,
    IdempotencyUnavailableError,
    IdempotencyValidationError,
)
from rag_system.tenancy import Principal, TenantId


def make_principal(tenant: str) -> Principal:
    return Principal(f"user-{tenant}", TenantId(tenant), frozenset({"writer"}))


def request_digest(payload: bytes = b'{"name":"manual"}') -> str:
    return hashlib.sha256(payload).hexdigest()


class FakeClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value
        self._lock = Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.value += seconds


class IdempotencyStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name, "idempotency.sqlite3")
        self.clock = FakeClock()
        self.store = IdempotencyStore(
            self.database,
            ttl_seconds=60,
            max_records_per_tenant=3,
            clock=self.clock,
        )
        self.tenant_a = make_principal("tenant-a")
        self.tenant_b = make_principal("tenant-b")

    def test_reserve_replay_bind_and_restart_recovery(self) -> None:
        digest = request_digest()
        first = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0001",
            digest,
        )

        self.assertTrue(first.created)
        self.assertFalse(first.is_bound)
        self.assertIsNone(first.resource_id)
        self.assertIsNone(first.job_id)

        pending_replay = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0001",
            digest,
        )
        self.assertFalse(pending_replay.created)
        self.assertEqual(pending_replay.reservation_id, first.reservation_id)

        bound = self.store.bind_result(
            self.tenant_a,
            first.reservation_id,
            "kb_0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef",
        )
        self.assertTrue(bound.is_bound)
        self.assertFalse(bound.created)

        reopened = IdempotencyStore(
            self.database,
            ttl_seconds=60,
            max_records_per_tenant=3,
            clock=self.clock,
        )
        restored = reopened.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0001",
            digest,
        )
        self.assertFalse(restored.created)
        self.assertEqual(restored.reservation_id, first.reservation_id)
        self.assertEqual(restored.resource_id, bound.resource_id)
        self.assertEqual(restored.job_id, bound.job_id)

    def test_key_reuse_with_different_digest_is_safe_conflict(self) -> None:
        key = "request-key-0002"
        self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            request_digest(b"first"),
        )
        with self.assertRaises(IdempotencyConflictError) as raised:
            self.store.reserve(
                self.tenant_a,
                "knowledge_bases.create",
                key,
                request_digest(b"second"),
            )

        message = str(raised.exception)
        self.assertNotIn(key, message)
        self.assertNotIn(request_digest(b"first"), message)
        self.assertNotIn(request_digest(b"second"), message)

    def test_plaintext_key_and_supplied_digest_are_not_persisted(self) -> None:
        key = "DO-NOT-PERSIST-THIS-KEY-7284"
        digest = request_digest(b"private canonical request")
        self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            digest,
        )

        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                "SELECT key_hash, request_fingerprint FROM idempotency_entries"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(len(row[0]), 64)
        self.assertEqual(len(row[1]), 64)
        self.assertNotEqual(row[0], hashlib.sha256(key.encode()).hexdigest())
        self.assertNotEqual(row[1], digest)

        persisted = b"".join(
            path.read_bytes()
            for path in Path(self.directory.name).iterdir()
            if path.is_file()
        )
        self.assertNotIn(key.encode(), persisted)
        self.assertNotIn(digest.encode(), persisted)

    def test_same_key_is_independent_across_tenants_and_operations(self) -> None:
        digest = request_digest()
        key = "shared-request-key"
        first = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            digest,
        )
        other_tenant = self.store.reserve(
            self.tenant_b,
            "knowledge_bases.create",
            key,
            digest,
        )
        other_operation = self.store.reserve(
            self.tenant_a,
            "answers.create",
            key,
            digest,
        )

        self.assertNotEqual(first.reservation_id, other_tenant.reservation_id)
        self.assertNotEqual(first.reservation_id, other_operation.reservation_id)
        with self.assertRaises(IdempotencyUnavailableError) as foreign:
            self.store.bind_result(
                self.tenant_b,
                first.reservation_id,
                "kb_0123456789abcdef0123456789abcdef",
                "0123456789abcdef0123456789abcdef",
            )
        with self.assertRaises(IdempotencyUnavailableError) as missing:
            self.store.bind_result(
                self.tenant_b,
                "idem_ffffffffffffffffffffffffffffffff",
                "kb_0123456789abcdef0123456789abcdef",
                "0123456789abcdef0123456789abcdef",
            )
        self.assertEqual(str(foreign.exception), str(missing.exception))

    def test_concurrent_same_key_across_store_instances_creates_once(self) -> None:
        stores = [
            IdempotencyStore(
                self.database,
                ttl_seconds=60,
                max_records_per_tenant=100,
                clock=self.clock,
            )
            for _ in range(12)
        ]

        def reserve(index: int) -> tuple[str, bool]:
            result = stores[index % len(stores)].reserve(
                self.tenant_a,
                "knowledge_bases.create",
                "concurrent-request-key",
                request_digest(),
            )
            return result.reservation_id, result.created

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(reserve, range(48)))

        self.assertEqual(len({reservation_id for reservation_id, _ in results}), 1)
        self.assertEqual(sum(created for _, created in results), 1)
        with closing(sqlite3.connect(self.database)) as connection:
            count = connection.execute("SELECT COUNT(*) FROM idempotency_entries").fetchone()[0]
        self.assertEqual(count, 1)

    def test_binding_is_idempotent_and_different_result_conflicts(self) -> None:
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0003",
            request_digest(),
        )
        first = self.store.bind_result(
            self.tenant_a,
            reservation.reservation_id,
            "kb_0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef",
        )
        repeated = self.store.bind_result(
            self.tenant_a,
            reservation.reservation_id,
            "kb_0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef",
        )
        self.assertEqual(first, repeated)

        with self.assertRaises(IdempotencyConflictError):
            self.store.bind_result(
                self.tenant_a,
                reservation.reservation_id,
                "kb_ffffffffffffffffffffffffffffffff",
                "ffffffffffffffffffffffffffffffff",
            )

    def test_recovery_binds_after_crash_and_rotates_only_the_same_resource_job(self) -> None:
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "recovery-request-key",
            request_digest(),
        )
        first = self.store.recover_binding(
            self.tenant_a,
            reservation.reservation_id,
            "kb_0123456789abcdef0123456789abcdef",
            "job_0123456789abcdef0123456789abcdef",
        )
        second = self.store.recover_binding(
            self.tenant_a,
            reservation.reservation_id,
            "kb_0123456789abcdef0123456789abcdef",
            "job_ffffffffffffffffffffffffffffffff",
        )
        self.assertEqual(first.resource_id, second.resource_id)
        self.assertNotEqual(first.job_id, second.job_id)

        with self.assertRaises(IdempotencyConflictError):
            self.store.recover_binding(
                self.tenant_a,
                reservation.reservation_id,
                "kb_ffffffffffffffffffffffffffffffff",
                "job_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_abandon_releases_an_unbound_key_immediately(self) -> None:
        key = "abandon-request-key"
        digest = request_digest()
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            digest,
        )

        self.assertTrue(self.store.abandon(self.tenant_a, reservation.reservation_id))
        with self.assertRaises(IdempotencyUnavailableError):
            self.store.bind_result(
                self.tenant_a,
                reservation.reservation_id,
                "kb_0123456789abcdef0123456789abcdef",
                "0123456789abcdef0123456789abcdef",
            )

        replacement = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            digest,
        )
        self.assertTrue(replacement.created)
        self.assertNotEqual(replacement.reservation_id, reservation.reservation_id)

    def test_abandon_rejects_bound_reservations_without_mutation(self) -> None:
        key = "bound-abandon-key"
        digest = request_digest()
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            digest,
        )
        bound = self.store.bind_result(
            self.tenant_a,
            reservation.reservation_id,
            "kb_0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef",
        )

        with self.assertRaises(IdempotencyConflictError):
            self.store.abandon(self.tenant_a, reservation.reservation_id)
        replay = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            key,
            digest,
        )
        self.assertEqual(replay.resource_id, bound.resource_id)
        self.assertEqual(replay.job_id, bound.job_id)

    def test_abandon_is_tenant_safe_expiry_safe_and_concurrent(self) -> None:
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "concurrent-abandon-key",
            request_digest(),
        )
        stores = [
            IdempotencyStore(
                self.database,
                ttl_seconds=60,
                max_records_per_tenant=100,
                clock=self.clock,
            )
            for _ in range(8)
        ]

        def abandon(index: int) -> str:
            try:
                stores[index % len(stores)].abandon(
                    self.tenant_a,
                    reservation.reservation_id,
                )
            except IdempotencyUnavailableError:
                return "unavailable"
            return "deleted"

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(abandon, range(24)))
        self.assertEqual(outcomes.count("deleted"), 1)
        self.assertEqual(outcomes.count("unavailable"), 23)

        foreign = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "foreign-abandon-key",
            request_digest(),
        )
        denials = []
        for principal, reservation_id in (
            (self.tenant_b, foreign.reservation_id),
            (self.tenant_a, "idem_ffffffffffffffffffffffffffffffff"),
            (self.tenant_a, "invalid"),
        ):
            with self.assertRaises(IdempotencyUnavailableError) as raised:
                self.store.abandon(principal, reservation_id)
            denials.append(str(raised.exception))
        self.assertEqual(len(set(denials)), 1)

        self.clock.advance(60)
        with self.assertRaises(IdempotencyUnavailableError) as expired:
            self.store.abandon(self.tenant_a, foreign.reservation_id)
        self.assertEqual(str(expired.exception), denials[0])

    def test_ttl_expiry_removes_old_reservation_and_releases_capacity(self) -> None:
        old = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0004",
            request_digest(),
        )
        self.clock.advance(60)
        replacement = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0004",
            request_digest(b"new request after expiry"),
        )
        self.assertTrue(replacement.created)
        self.assertNotEqual(replacement.reservation_id, old.reservation_id)
        with self.assertRaises(IdempotencyUnavailableError):
            self.store.bind_result(
                self.tenant_a,
                old.reservation_id,
                "kb_0123456789abcdef0123456789abcdef",
                "0123456789abcdef0123456789abcdef",
            )

        self.clock.advance(60)
        self.assertEqual(self.store.purge_expired(), 1)
        self.assertEqual(self.store.purge_expired(), 0)

    def test_capacity_is_bounded_per_tenant(self) -> None:
        limited = IdempotencyStore(
            Path(self.directory.name, "limited.sqlite3"),
            ttl_seconds=60,
            max_records_per_tenant=2,
            clock=self.clock,
        )
        for index in range(2):
            limited.reserve(
                self.tenant_a,
                "knowledge_bases.create",
                f"capacity-key-{index}",
                request_digest(str(index).encode()),
            )
        with self.assertRaises(IdempotencyCapacityError):
            limited.reserve(
                self.tenant_a,
                "knowledge_bases.create",
                "capacity-key-2",
                request_digest(b"2"),
            )

        other = limited.reserve(
            self.tenant_b,
            "knowledge_bases.create",
            "capacity-key-2",
            request_digest(b"2"),
        )
        self.assertTrue(other.created)

    def test_inputs_and_configuration_are_strictly_bounded(self) -> None:
        digest = request_digest()
        for operation in ("", "UPPER", "contains space", "x" * 65, 1):
            with self.subTest(operation=operation), self.assertRaises(
                IdempotencyValidationError
            ):
                self.store.reserve(self.tenant_a, operation, "request-key-0005", digest)
        for key in ("short", "contains space", "x" * 256, "unicode-密钥", 1):
            with self.subTest(key=key), self.assertRaises(IdempotencyValidationError):
                self.store.reserve(self.tenant_a, "answers.create", key, digest)
        for invalid_digest in ("", "f" * 63, "F" * 64, "g" * 64, 1):
            with self.subTest(digest=invalid_digest), self.assertRaises(
                IdempotencyValidationError
            ):
                self.store.reserve(
                    self.tenant_a,
                    "answers.create",
                    "request-key-0005",
                    invalid_digest,
                )

        for ttl in (0, -1, float("inf"), 30 * 24 * 60 * 60 + 1, True):
            with self.subTest(ttl=ttl), self.assertRaises(IdempotencyValidationError):
                IdempotencyStore(Path(self.directory.name, f"bad-{ttl}.sqlite3"), ttl_seconds=ttl)
        with self.assertRaises(TypeError):
            self.store.reserve("tenant-a", "answers.create", "request-key-0005", digest)

    def test_result_ids_and_lookup_denials_are_safe(self) -> None:
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-0006",
            request_digest(),
        )
        for resource_id, job_id in (
            ("../kb", "valid-job"),
            ("valid-kb", "job/escape"),
            ("x" * 193, "valid-job"),
        ):
            with self.subTest(resource_id=resource_id, job_id=job_id), self.assertRaises(
                IdempotencyValidationError
            ):
                self.store.bind_result(
                    self.tenant_a,
                    reservation.reservation_id,
                    resource_id,
                    job_id,
                )

        messages = []
        for reservation_id in (
            "bad",
            "idem_ffffffffffffffffffffffffffffffff",
        ):
            with self.assertRaises(IdempotencyUnavailableError) as raised:
                self.store.bind_result(
                    self.tenant_a,
                    reservation_id,
                    "valid-kb",
                    "valid-job",
                )
            messages.append(str(raised.exception))
        self.assertEqual(len(set(messages)), 1)

    def test_result_ids_accept_opaque_catalog_suffixes(self) -> None:
        reservation = self.store.reserve(
            self.tenant_a,
            "knowledge_bases.create",
            "request-key-suffix",
            request_digest(),
        )
        bound = self.store.bind_result(
            self.tenant_a,
            reservation.reservation_id,
            "kb_opaque-token_",
            "job-opaque-token-",
        )
        self.assertEqual(bound.resource_id, "kb_opaque-token_")

    def test_schema_validation_and_wal_mode(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
            connection.execute("UPDATE idempotency_meta SET schema_version = 99")
            connection.commit()
        with self.assertRaises(IdempotencySchemaError):
            IdempotencyStore(self.database)


if __name__ == "__main__":
    unittest.main()
