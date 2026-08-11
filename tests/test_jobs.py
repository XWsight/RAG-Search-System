from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

from rag_system.job_contracts import JobId, JobSnapshot, JobStorageError
from rag_system.jobs import (
    JobCapacityError,
    JobManager,
    JobManagerShutdownError,
    JobNotFoundError,
    JobStatus,
)


class ManualExecutor(Executor):
    def __init__(self) -> None:
        self.pending: list[tuple[Future[Any], Any, tuple[Any, ...]]] = []
        self.shutdown_called = False

    def submit(self, fn, /, *args, **kwargs):
        if kwargs:
            raise AssertionError("manual executor only supports positional arguments")
        future: Future[Any] = Future()
        self.pending.append((future, fn, args))
        return future

    def run_next(self) -> None:
        future, function, arguments = self.pending.pop(0)
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = function(*arguments)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_called = True
        if cancel_futures:
            for future, _, _ in self.pending:
                future.cancel()


class SequentialIdFactory:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"job-{self.value}"


class DegradingSnapshotStore:
    def __init__(self) -> None:
        self.snapshot: JobSnapshot | None = None
        self.fail_writes = False

    def put(self, tenant_id: str, snapshot: JobSnapshot) -> None:
        if self.fail_writes:
            raise JobStorageError("unavailable")
        self.snapshot = snapshot

    def get(self, tenant_id: str, job_id: JobId | str) -> JobSnapshot:
        if self.snapshot is None:
            raise JobStorageError("unavailable")
        return self.snapshot

    def delete(self, tenant_id: str, job_id: JobId | str) -> bool:
        self.snapshot = None
        return True

    def healthcheck(self) -> bool:
        return True


class JobManagerTests(unittest.TestCase):
    def make_manager(self, **overrides: object) -> tuple[JobManager, ManualExecutor]:
        executor = ManualExecutor()
        options: dict[str, object] = {
            "max_workers": 1,
            "max_jobs": 4,
            "executor": executor,
            "id_factory": SequentialIdFactory(),
        }
        options.update(overrides)
        return JobManager(**options), executor

    def test_queued_running_succeeded_and_result_copy(self) -> None:
        manager, executor = self.make_manager()
        job_id = manager.submit(
            "tenant-a",
            lambda token: {"count": 3, "items": ["a", "b"]},
            idempotency_key="request-1",
        )
        self.assertEqual(manager.get("tenant-a", job_id).status, JobStatus.QUEUED)
        executor.run_next()
        first = manager.get("tenant-a", job_id)
        self.assertEqual(first.status, JobStatus.SUCCEEDED)
        self.assertEqual(first.result, {"count": 3, "items": ["a", "b"]})

        assert first.result is not None
        first.result["count"] = 999
        self.assertEqual(manager.get("tenant-a", job_id).result["count"], 3)

    def test_idempotency_is_scoped_to_tenant(self) -> None:
        manager, _ = self.make_manager()
        first = manager.submit("a", lambda token: {"ok": True}, idempotency_key="same")
        repeated = manager.submit("a", lambda token: {"ignored": True}, idempotency_key="same")
        other_tenant = manager.submit(
            "b", lambda token: {"ok": True}, idempotency_key="same"
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_tenant)

    def test_query_cancel_and_stats_are_tenant_isolated(self) -> None:
        manager, _ = self.make_manager()
        job_id = manager.submit("owner", lambda token: {"ok": True}, idempotency_key="one")

        with self.assertRaises(JobNotFoundError):
            manager.get("other", job_id)
        with self.assertRaises(JobNotFoundError):
            manager.cancel("other", job_id)
        self.assertEqual(manager.stats("other"), {status.value: 0 for status in JobStatus})

        cancelled = manager.cancel("owner", job_id)
        self.assertEqual(cancelled.status, JobStatus.CANCELLED)
        self.assertEqual(manager.stats("owner")["cancelled"], 1)

    def test_capacity_never_evicts_active_jobs_but_reuses_terminal_space(self) -> None:
        manager, executor = self.make_manager(max_jobs=2)
        first = manager.submit("t", lambda token: {"n": 1}, idempotency_key="1")
        manager.submit("t", lambda token: {"n": 2}, idempotency_key="2")
        with self.assertRaises(JobCapacityError):
            manager.submit("t", lambda token: {"n": 3}, idempotency_key="3")

        executor.run_next()
        third = manager.submit("t", lambda token: {"n": 3}, idempotency_key="3")
        with self.assertRaises(JobNotFoundError):
            manager.get("t", first)
        self.assertEqual(str(third), "job-3")

    def test_per_tenant_capacity_preserves_slots_for_other_tenants(self) -> None:
        manager, executor = self.make_manager(max_jobs=4, max_jobs_per_tenant=2)
        first = manager.submit("noisy", lambda token: {"n": 1}, idempotency_key="1")
        manager.submit("noisy", lambda token: {"n": 2}, idempotency_key="2")
        with self.assertRaises(JobCapacityError):
            manager.submit("noisy", lambda token: {"n": 3}, idempotency_key="3")

        other = manager.submit("other", lambda token: {"n": 1}, idempotency_key="1")
        self.assertNotEqual(first, other)
        executor.run_next()
        replacement = manager.submit(
            "noisy",
            lambda token: {"n": 3},
            idempotency_key="3",
        )
        self.assertEqual(str(replacement), "job-4")

    def test_terminal_jobs_use_ttl_and_lru_cleanup(self) -> None:
        now = [0.0]
        manager, executor = self.make_manager(
            max_jobs=2,
            ttl_seconds=10,
            clock=lambda: now[0],
        )
        first = manager.submit("t", lambda token: {"n": 1}, idempotency_key="1")
        executor.run_next()
        now[0] = 1.0
        second = manager.submit("t", lambda token: {"n": 2}, idempotency_key="2")
        executor.run_next()

        now[0] = 2.0
        manager.get("t", first)  # first is now the most recently used terminal job
        manager.submit("t", lambda token: {"n": 3}, idempotency_key="3")
        with self.assertRaises(JobNotFoundError):
            manager.get("t", second)
        self.assertEqual(manager.get("t", first).status, JobStatus.SUCCEEDED)

        now[0] = 12.0
        self.assertEqual(manager.cleanup(), 1)
        with self.assertRaises(JobNotFoundError):
            manager.get("t", first)

    def test_task_exception_is_safely_mapped_without_original_text(self) -> None:
        manager, executor = self.make_manager()

        def fail(token):
            raise RuntimeError("database password and internal host")

        job_id = manager.submit("t", fail, idempotency_key="failure")
        executor.run_next()
        snapshot = manager.get("t", job_id)
        self.assertEqual(snapshot.status, JobStatus.FAILED)
        self.assertEqual(snapshot.error_code, "task_failed")
        self.assertEqual(snapshot.error_message, "task execution failed")
        self.assertNotIn("password", repr(snapshot))
        self.assertIsNone(snapshot.result)

    def test_invalid_and_oversized_results_fail_safely(self) -> None:
        manager, executor = self.make_manager(max_result_bytes=40)
        invalid_id = manager.submit(
            "t", lambda token: {"bad": object()}, idempotency_key="invalid"
        )
        oversized_id = manager.submit(
            "t", lambda token: {"text": "x" * 100}, idempotency_key="large"
        )
        executor.run_next()
        executor.run_next()
        for job_id in (invalid_id, oversized_id):
            snapshot = manager.get("t", job_id)
            self.assertEqual(snapshot.status, JobStatus.FAILED)
            self.assertEqual(snapshot.error_code, "invalid_result")
            self.assertIsNone(snapshot.result)

    def test_running_job_can_be_cancelled_cooperatively(self) -> None:
        started = threading.Event()
        release = threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)
        manager = JobManager(
            max_workers=1,
            max_jobs=2,
            executor=executor,
            id_factory=SequentialIdFactory(),
        )

        def task(token):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            token.raise_if_cancelled()
            return {"ok": True}

        job_id = manager.submit("t", task, idempotency_key="running")
        self.assertTrue(started.wait(timeout=2))
        self.assertEqual(manager.get("t", job_id).status, JobStatus.RUNNING)
        self.assertEqual(manager.cancel("t", job_id).status, JobStatus.CANCELLING)
        release.set()
        manager.shutdown(wait=True)
        self.assertEqual(manager.get("t", job_id).status, JobStatus.CANCELLED)

    def test_cancelling_job_keeps_capacity_until_the_worker_exits(self) -> None:
        started = threading.Event()
        release = threading.Event()
        manager = JobManager(max_workers=1, max_jobs=1, max_jobs_per_tenant=1)

        def task(token):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            token.raise_if_cancelled()
            return {"ok": True}

        try:
            job_id = manager.submit("t", task, idempotency_key="running")
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(manager.cancel("t", job_id).status, JobStatus.CANCELLING)
            with self.assertRaises(JobCapacityError):
                manager.submit(
                    "t",
                    lambda token: {"replacement": True},
                    idempotency_key="next",
                )

            release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if manager.get("t", job_id).status is JobStatus.CANCELLED:
                    break
                time.sleep(0.01)
            self.assertEqual(manager.get("t", job_id).status, JobStatus.CANCELLED)
            replacement = manager.submit(
                "t",
                lambda token: {"replacement": True},
                idempotency_key="next",
            )
            self.assertEqual(manager.get("t", replacement).status, JobStatus.QUEUED)
        finally:
            release.set()
            manager.shutdown(wait=True)

    def test_successful_completion_can_win_a_late_cancel_request(self) -> None:
        started = threading.Event()
        release = threading.Event()
        manager = JobManager(max_workers=1, max_jobs=1)

        def task(_token):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return {"committed": True}

        try:
            job_id = manager.submit("t", task, idempotency_key="commit")
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(manager.cancel("t", job_id).status, JobStatus.CANCELLING)
            release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if manager.get("t", job_id).status.terminal:
                    break
                time.sleep(0.01)
            snapshot = manager.get("t", job_id)
            self.assertEqual(snapshot.status, JobStatus.SUCCEEDED)
            self.assertEqual(snapshot.result, {"committed": True})
        finally:
            release.set()
            manager.shutdown(wait=True)

    def test_shutdown_cancels_pending_and_rejects_new_work(self) -> None:
        manager, executor = self.make_manager()
        job_id = manager.submit("t", lambda token: {"ok": True}, idempotency_key="one")
        manager.shutdown(wait=False, cancel_pending=True)
        self.assertTrue(executor.shutdown_called)
        self.assertEqual(manager.get("t", job_id).status, JobStatus.CANCELLED)
        with self.assertRaises(JobManagerShutdownError):
            manager.submit("t", lambda token: {"ok": True}, idempotency_key="two")

    def test_archive_failure_does_not_prevent_executor_shutdown(self) -> None:
        store = DegradingSnapshotStore()
        manager, executor = self.make_manager(snapshot_store=store)
        job_id = manager.submit(
            "t",
            lambda token: {"ok": True},
            idempotency_key="one",
        )
        store.fail_writes = True

        manager.shutdown(wait=False, cancel_pending=True)

        self.assertTrue(executor.shutdown_called)
        self.assertEqual(manager.get("t", job_id).status, JobStatus.CANCELLED)
        self.assertFalse(manager.healthcheck())

    def test_configuration_and_input_validation(self) -> None:
        for options in (
            {"max_workers": 0},
            {"max_workers": 2, "max_jobs": 1},
            {"ttl_seconds": 0},
            {"max_result_bytes": 0},
            {"max_result_depth": 0},
            {"max_result_items": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                JobManager(**options)

        manager, _ = self.make_manager()
        with self.assertRaises(ValueError):
            manager.submit("", lambda token: {"ok": True}, idempotency_key="one")
        with self.assertRaises(ValueError):
            manager.submit("t", lambda token: {"ok": True}, idempotency_key="")
        with self.assertRaises(TypeError):
            manager.submit("t", "not-callable", idempotency_key="one")


if __name__ == "__main__":
    unittest.main()
