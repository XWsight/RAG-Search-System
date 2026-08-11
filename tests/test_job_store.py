from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from rag_system.job_contracts import (
    JobId,
    JobNotFoundError,
    JobSnapshot,
    JobStatus,
    JobStorageError,
)
from rag_system.job_store import SqliteJobSnapshotStore
from rag_system.jobs import JobManager


def snapshot(
    identifier: str,
    status: JobStatus,
    updated_at: float,
    *,
    result: dict[str, object] | None = None,
) -> JobSnapshot:
    terminal = status.terminal
    return JobSnapshot(
        job_id=JobId(identifier),
        status=status,
        created_at=0.0,
        updated_at=updated_at,
        started_at=0.0 if status is not JobStatus.QUEUED else None,
        finished_at=updated_at if terminal else None,
        result=result,
        error_code="task_failed" if status is JobStatus.FAILED else "",
        error_message="task execution failed" if status is JobStatus.FAILED else "",
    )


class JobSnapshotStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name, "jobs.sqlite3")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_snapshots_survive_reopen_and_tenant_isolation_is_indistinguishable(self) -> None:
        store = SqliteJobSnapshotStore(self.database, clock=lambda: 10.0)
        expected = snapshot(
            "job-persisted",
            JobStatus.SUCCEEDED,
            2.0,
            result={"count": 3},
        )
        store.put("tenant-a", expected)

        reopened = SqliteJobSnapshotStore(self.database, clock=lambda: 10.0)
        self.assertEqual(reopened.get("tenant-a", expected.job_id), expected)
        same_id_other_tenant = snapshot(
            "job-persisted",
            JobStatus.CANCELLED,
            3.0,
        )
        reopened.put("tenant-b", same_id_other_tenant)
        self.assertEqual(
            reopened.get("tenant-b", expected.job_id),
            same_id_other_tenant,
        )
        for tenant, identifier in (("tenant-c", expected.job_id), ("tenant-a", "missing")):
            with self.subTest(tenant=tenant), self.assertRaises(JobNotFoundError):
                reopened.get(tenant, identifier)

    def test_terminal_ttl_expires_on_read_at_the_exact_boundary(self) -> None:
        now = [100.0]
        store = SqliteJobSnapshotStore(
            self.database,
            ttl_seconds=60,
            clock=lambda: now[0],
        )
        stored = snapshot("job-expiring", JobStatus.SUCCEEDED, 50.0, result={})
        store.put("tenant-a", stored)
        now[0] = 109.999
        self.assertEqual(store.get("tenant-a", stored.job_id), stored)

        now[0] = 110.0
        with self.assertRaises(JobNotFoundError):
            store.get("tenant-a", stored.job_id)

    def test_recovery_marks_interrupted_work_failed_without_reexecuting_it(self) -> None:
        now = [10.0]
        store = SqliteJobSnapshotStore(self.database, clock=lambda: now[0])
        store.put("tenant-a", snapshot("job-running", JobStatus.RUNNING, 1.0))

        now[0] = 20.0
        self.assertEqual(store.recover_interrupted(), 1)
        recovered = store.get("tenant-a", "job-running")
        self.assertEqual(recovered.status, JobStatus.FAILED)
        self.assertEqual(recovered.error_code, "worker_restarted")
        self.assertEqual(recovered.finished_at, 20.0)
        self.assertEqual(store.recover_interrupted(), 0)

    def test_capacity_evicts_only_oldest_terminal_history(self) -> None:
        store = SqliteJobSnapshotStore(
            self.database,
            max_records_per_tenant=2,
            clock=lambda: 10.0,
        )
        store.put("tenant-a", snapshot("job-1", JobStatus.SUCCEEDED, 1.0, result={}))
        store.put("tenant-a", snapshot("job-2", JobStatus.CANCELLED, 2.0))
        store.put("tenant-a", snapshot("job-3", JobStatus.SUCCEEDED, 3.0, result={}))

        with self.assertRaises(JobNotFoundError):
            store.get("tenant-a", "job-1")
        self.assertEqual(store.get("tenant-a", "job-2").status, JobStatus.CANCELLED)
        self.assertEqual(store.get("tenant-a", "job-3").status, JobStatus.SUCCEEDED)

    def test_job_manager_falls_back_to_durable_history_after_memory_eviction(self) -> None:
        store = SqliteJobSnapshotStore(self.database)
        manager = JobManager(
            max_workers=1,
            max_jobs=1,
            max_jobs_per_tenant=1,
            snapshot_store=store,
        )
        try:
            first = manager.submit(
                "tenant-a",
                lambda _token: {"value": 1},
                idempotency_key="first",
            )
            self._wait_for_terminal(manager, first)
            second = manager.submit(
                "tenant-a",
                lambda _token: {"value": 2},
                idempotency_key="second",
            )

            archived = manager.get("tenant-a", first)
            self.assertEqual(archived.status, JobStatus.SUCCEEDED)
            self.assertEqual(archived.result, {"value": 1})
            self._wait_for_terminal(manager, second)
        finally:
            manager.shutdown(wait=True)

    def test_concurrent_writers_and_corrupt_rows_fail_safely(self) -> None:
        store = SqliteJobSnapshotStore(
            self.database,
            max_records_per_tenant=100,
            clock=lambda: 100.0,
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    store.put,
                    "tenant-a",
                    snapshot(f"job-{index}", JobStatus.SUCCEEDED, float(index), result={}),
                )
                for index in range(32)
            ]
            for future in futures:
                future.result()

        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "UPDATE job_snapshots SET result_json = ? WHERE job_id = ?",
                ("[]", "job-1"),
            )
            connection.commit()
        with self.assertRaises(JobStorageError):
            store.get("tenant-a", "job-1")

    def test_existing_database_must_match_the_exact_schema(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "CREATE TABLE job_snapshots (job_id TEXT PRIMARY KEY NOT NULL)"
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()

        with self.assertRaises(JobStorageError):
            SqliteJobSnapshotStore(self.database)

    @staticmethod
    def _wait_for_terminal(manager: JobManager, job_id: JobId) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if manager.get("tenant-a", job_id).status.terminal:
                return
            time.sleep(0.01)
        raise AssertionError("job did not reach a terminal state")


if __name__ == "__main__":
    unittest.main()
