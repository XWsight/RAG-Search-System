from __future__ import annotations

import tempfile
import time
import unittest
import sqlite3
import threading
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag_system.catalog import (
    KnowledgeBaseCatalog,
    KnowledgeBaseStatus,
    KnowledgeBaseUnavailableError,
)
from rag_system.config import Settings
from rag_system.domain import (
    AnswerRequest,
    AnswerResult,
    Chunk,
    IndexRef,
    Route,
    RouteDecision,
)
from rag_system.file_store import FileStoreIOError, TenantFileStore
from rag_system.ingestion import IngestionResult
from rag_system.idempotency import IdempotencyStore
from rag_system.idempotency import IdempotencyConflictError
from rag_system.jobs import JobManager, JobNotFoundError, JobStatus
from rag_system.job_contracts import JobId, JobSnapshot
from rag_system.job_store import SqliteJobSnapshotStore
from rag_system.platform import (
    IdempotencyInProgressError,
    KnowledgeBaseNotReadyError,
    PlatformIntegrityError,
    PlatformUnavailableError,
    RagPlatform,
    UploadDocument,
)
from rag_system.tenancy import Principal, TenantId
from rag_system.text import stable_digest


class FakeIndexManager:
    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.deleted: list[str] = []
        self.closed = False

    def get(self, index_id):
        if index_id not in self.loaded:
            raise KeyError(index_id)
        return object()

    def delete(self, index_id):
        self.loaded.discard(index_id)
        self.deleted.append(index_id)
        return True

    def close(self):
        self.closed = True


class FakeService:
    def __init__(self) -> None:
        self.index_manager = FakeIndexManager()
        self.prepared_namespaces: list[str] = []
        self.last_request: AnswerRequest | None = None
        self.cleared_sessions: list[str] = []

    def prepare_index(self, paths, *, namespace=""):
        self.prepared_namespaces.append(namespace)
        content = "".join(Path(path).read_text(encoding="utf-8") for path in paths)
        index_id = "idx_" + stable_digest(["fake-v1", namespace, content])
        chunk = Chunk("chunk-1", "doc-1", "guide.txt", content, 0, 0, len(content))
        return IngestionResult(index_id, (), (chunk,))

    def create_prepared_index(self, ingestion):
        self.index_manager.loaded.add(ingestion.index_id)
        return IndexRef(ingestion.index_id, 1, len(ingestion.chunks), time.time())

    def create_index(self, paths, *, namespace=""):
        return self.create_prepared_index(self.prepare_index(paths, namespace=namespace))

    def answer(self, index_id, request):
        if index_id not in self.index_manager.loaded:
            raise AssertionError("answer used an unloaded index")
        self.last_request = request
        return AnswerResult(
            answer="grounded answer",
            decision=RouteDecision(Route.LOCAL, 0.9, "local evidence"),
        )

    def clear_session(self, session_id):
        self.cleared_sessions.append(session_id)
        return True


def principal(tenant: str) -> Principal:
    return Principal(
        subject=f"user-{tenant}",
        tenant_id=TenantId(tenant),
        roles=frozenset({"reader", "writer"}),
    )


class PlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.root = root
        self.settings = replace(
            Settings(),
            project_root=root,
            storage_root=root,
            max_file_bytes=10_000,
            max_total_bytes=50_000,
            max_concurrent_answers=1,
        ).validate()
        self.service = FakeService()
        self.jobs = JobManager(max_workers=2, max_jobs=16, ttl_seconds=30)
        self.addCleanup(self.jobs.shutdown)
        self.platform = RagPlatform(
            settings=self.settings,
            service=self.service,
            catalog=KnowledgeBaseCatalog(root / "catalog.sqlite3"),
            file_store=TenantFileStore(
                root / "documents",
                max_file_bytes=10_000,
                max_total_bytes=50_000,
                max_files_per_tenant=20,
            ),
            jobs=self.jobs,
            idempotency=IdempotencyStore(root / "idempotency.sqlite3"),
        )
        self.tenant_a = principal("tenant-a")
        self.tenant_b = principal("tenant-b")

    def _create_ready(self):
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Engineering handbook",
            documents=(UploadDocument("guide.txt", b"RAG evidence"),),
            idempotency_key="create-engineering-handbook",
        )
        snapshot = self._wait_for_job(submission.job_id.value)
        self.assertEqual(snapshot.status, JobStatus.SUCCEEDED)
        record = self.platform.get_knowledge_base(
            self.tenant_a,
            submission.knowledge_base.resource_id,
        )
        self.assertEqual(record.status, KnowledgeBaseStatus.READY)
        return record

    def _wait_for_job(self, job_id: str):
        return self._wait_for_job_on(self.platform, job_id)

    def _wait_for_job_on(self, platform: RagPlatform, job_id: str):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = platform.get_job(self.tenant_a, job_id)
            if snapshot.status.terminal:
                return snapshot
            time.sleep(0.01)
        self.fail("background job did not complete")

    def _stage_preparing(self, *, key: str, documents: tuple[UploadDocument, ...]):
        uploads = self.platform._uploads.prepare(documents)
        reservation = self.platform.idempotency.reserve(
            self.tenant_a,
            "knowledge_base.create",
            key,
            self.platform._uploads.request_digest("Interrupted upload", uploads),
        )
        record = self.platform.catalog.create(
            self.tenant_a,
            "Interrupted upload",
            idempotency_reservation_id=reservation.reservation_id,
        )
        planned = self.platform._assets.plan(
            self.tenant_a,
            uploads,
            new_document_id=self.platform._uploads.new_document_id,
        )
        record = self.platform.catalog.attach_manifest(
            self.tenant_a,
            record.resource_id,
            tuple(item.manifest for item in planned),
        )
        return record, planned

    def test_recovery_promotes_a_fully_stored_preparing_upload(self) -> None:
        documents = (UploadDocument("complete.txt", b"complete evidence"),)
        record, planned = self._stage_preparing(
            key="complete-preparing",
            documents=documents,
        )
        self.platform._assets.store(self.tenant_a, planned)

        self.assertEqual(self.platform.recover_incomplete((self.tenant_a,)), 1)
        replay = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Interrupted upload",
            documents=documents,
            idempotency_key="complete-preparing",
        )

        self.assertTrue(replay.replayed)
        self.assertEqual(replay.knowledge_base.resource_id, record.resource_id)
        self.assertEqual(
            self._wait_for_job(replay.job_id.value).status,
            JobStatus.SUCCEEDED,
        )
        self.assertEqual(
            self.platform.get_knowledge_base(self.tenant_a, record.resource_id).status,
            KnowledgeBaseStatus.READY,
        )

    def test_recovery_rolls_back_a_partial_preparing_upload_for_clean_retry(self) -> None:
        documents = (
            UploadDocument("first.txt", b"first evidence"),
            UploadDocument("second.txt", b"second evidence"),
        )
        record, planned = self._stage_preparing(
            key="partial-preparing",
            documents=documents,
        )
        self.platform._assets.store(self.tenant_a, planned[:1])

        self.assertEqual(self.platform.recover_incomplete((self.tenant_a,)), 1)
        with self.assertRaises(KnowledgeBaseUnavailableError):
            self.platform.get_knowledge_base(self.tenant_a, record.resource_id)

        retried = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Interrupted upload",
            documents=documents,
            idempotency_key="partial-preparing",
        )
        self.assertFalse(retried.replayed)
        self.assertEqual(
            self._wait_for_job(retried.job_id.value).status,
            JobStatus.SUCCEEDED,
        )

    def test_failed_preparing_cleanup_keeps_idempotency_until_retry_converges(self) -> None:
        documents = (
            UploadDocument("first.txt", b"first evidence"),
            UploadDocument("second.txt", b"second evidence"),
        )
        record, planned = self._stage_preparing(
            key="cleanup-preparing",
            documents=documents,
        )
        self.platform._assets.store(self.tenant_a, planned[:1])

        with patch.object(
            self.platform.file_store,
            "delete",
            side_effect=FileStoreIOError("transient failure"),
        ):
            self.assertEqual(self.platform.recover_incomplete((self.tenant_a,)), 1)
        self.assertEqual(
            self.platform.get_knowledge_base(self.tenant_a, record.resource_id).status,
            KnowledgeBaseStatus.DELETING,
        )
        with self.assertRaises(IdempotencyInProgressError):
            self.platform.create_knowledge_base(
                self.tenant_a,
                display_name="Interrupted upload",
                documents=documents,
                idempotency_key="cleanup-preparing",
            )

        self.assertEqual(self.platform.recover_incomplete((self.tenant_a,)), 1)
        retried = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Interrupted upload",
            documents=documents,
            idempotency_key="cleanup-preparing",
        )
        self.assertFalse(retried.replayed)
        self.assertEqual(
            self._wait_for_job(retried.job_id.value).status,
            JobStatus.SUCCEEDED,
        )

    def test_create_answer_session_isolation_and_delete(self) -> None:
        record = self._create_ready()
        expected_namespace = f"tenant-a:{record.resource_id}"
        self.assertIn(expected_namespace, self.service.prepared_namespaces)

        result = self.platform.answer(
            self.tenant_a,
            record.resource_id,
            AnswerRequest("What is RAG?", "browser-session", allow_cloud=False),
        )
        self.assertEqual(result.answer, "grounded answer")
        self.assertIsNotNone(self.service.last_request)
        scoped_session = self.service.last_request.session_id
        self.assertTrue(scoped_session.startswith("session_"))
        self.assertNotIn("browser-session", scoped_session)

        self.assertTrue(
            self.platform.clear_session(
                self.tenant_a,
                record.resource_id,
                "browser-session",
            )
        )
        self.assertEqual(self.service.cleared_sessions, [scoped_session])

        self.assertTrue(self.platform.delete_knowledge_base(self.tenant_a, record.resource_id))
        with self.assertRaises(KnowledgeBaseUnavailableError):
            self.platform.get_knowledge_base(self.tenant_a, record.resource_id)
        self.assertIn(record.internal_index_id, self.service.index_manager.deleted)

    def test_cross_tenant_access_is_indistinguishable_from_missing(self) -> None:
        record = self._create_ready()
        messages = []
        for resource_id in (record.resource_id, "kb_" + "x" * 32):
            with self.assertRaises(KnowledgeBaseUnavailableError) as raised:
                self.platform.get_knowledge_base(self.tenant_b, resource_id)
            messages.append(str(raised.exception))
        self.assertEqual(len(set(messages)), 1)

    def test_restart_reloads_index_but_rejects_tampered_document(self) -> None:
        record = self._create_ready()
        self.service.index_manager.loaded.clear()
        document_id = record.documents[0].relative_path.split("/")[1]
        path = self.platform.file_store.resolve("tenant-a", document_id)
        path.write_text("tampered", encoding="utf-8")

        with self.assertRaises(PlatformIntegrityError):
            self.platform.answer(
                self.tenant_a,
                record.resource_id,
                AnswerRequest("question", "session"),
            )

    def test_index_disappearing_during_answer_returns_safe_not_ready(self) -> None:
        record = self._create_ready()
        original_answer = self.service.answer
        def missing_answer(*_args, **_kwargs):
            raise KeyError("concurrent delete")

        self.service.answer = missing_answer
        try:
            with self.assertRaises(KnowledgeBaseNotReadyError):
                self.platform.answer(
                    self.tenant_a,
                    record.resource_id,
                    AnswerRequest("question", "session"),
                )
        finally:
            self.service.answer = original_answer

    def test_answer_admission_rejects_excess_concurrency_without_queueing(self) -> None:
        record = self._create_ready()
        original_answer = self.service.answer
        entered = threading.Event()
        release = threading.Event()
        errors: list[Exception] = []

        def slow_answer(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original_answer(*args, **kwargs)

        self.service.answer = slow_answer

        def first_request() -> None:
            try:
                self.platform.answer(
                    self.tenant_a,
                    record.resource_id,
                    AnswerRequest("first", "session-1"),
                )
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=first_request)
        worker.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaises(PlatformUnavailableError):
            self.platform.answer(
                self.tenant_a,
                record.resource_id,
                AnswerRequest("second", "session-2"),
            )
        release.set()
        worker.join(2)
        self.service.answer = original_answer
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_identical_documents_in_different_tenants_get_distinct_namespaces(self) -> None:
        first = self._create_ready()
        submission = self.platform.create_knowledge_base(
            self.tenant_b,
            display_name="Engineering handbook",
            documents=(UploadDocument("guide.txt", b"RAG evidence"),),
            idempotency_key="tenant-b-create",
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = self.platform.get_job(self.tenant_b, submission.job_id)
            if snapshot.status.terminal:
                break
            time.sleep(0.01)
        else:
            self.fail("tenant B job did not complete")
        second = self.platform.get_knowledge_base(
            self.tenant_b,
            submission.knowledge_base.resource_id,
        )

        self.assertNotEqual(first.internal_index_id, second.internal_index_id)

    def test_create_replay_returns_the_original_resource_and_job(self) -> None:
        first = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Replay safe handbook",
            documents=(UploadDocument("guide.txt", b"same request"),),
            idempotency_key="stable-create-key",
        )
        second = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Replay safe handbook",
            documents=(UploadDocument("guide.txt", b"same request"),),
            idempotency_key="stable-create-key",
        )

        self.assertFalse(first.replayed)
        self.assertTrue(second.replayed)
        self.assertEqual(first.knowledge_base.resource_id, second.knowledge_base.resource_id)
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(len(self.platform.list_knowledge_bases(self.tenant_a)), 1)

    def test_replay_replaces_an_evicted_job_with_a_pollable_status_job(self) -> None:
        original = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Eviction target",
            documents=(UploadDocument("target.txt", b"target evidence"),),
            idempotency_key="eviction-target-request",
        )
        self.assertEqual(
            self._wait_for_job(original.job_id.value).status,
            JobStatus.SUCCEEDED,
        )

        for index in range(16):
            submission = self.platform.create_knowledge_base(
                self.tenant_a,
                display_name=f"Capacity filler {index}",
                documents=(
                    UploadDocument(f"filler-{index}.txt", f"filler {index}".encode()),
                ),
                idempotency_key=f"capacity-filler-{index}",
            )
            self.assertEqual(
                self._wait_for_job(submission.job_id.value).status,
                JobStatus.SUCCEEDED,
            )

        replay = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Eviction target",
            documents=(UploadDocument("target.txt", b"target evidence"),),
            idempotency_key="eviction-target-request",
        )
        self.assertTrue(replay.replayed)
        self.assertNotEqual(replay.job_id, original.job_id)
        self.assertIn(
            self.platform.get_job(self.tenant_a, replay.job_id).status,
            {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED},
        )

    def test_cancel_does_not_signal_worker_until_intent_is_durable(self) -> None:
        started = threading.Event()
        release = threading.Event()
        original_create = self.service.create_prepared_index

        def blocked_create(ingestion):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return original_create(ingestion)

        self.service.create_prepared_index = blocked_create
        self.addCleanup(setattr, self.service, "create_prepared_index", original_create)
        self.addCleanup(release.set)
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Cancellation retry",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="cancellation-retry-request",
        )
        self.assertTrue(started.wait(timeout=2))

        original_transition = self.platform.catalog.transition
        reject_intent = [True]

        def transition(principal_value, resource_id, target, **kwargs):
            if reject_intent[0] and target is KnowledgeBaseStatus.CANCELLING:
                raise OSError("transient catalog failure")
            return original_transition(principal_value, resource_id, target, **kwargs)

        self.platform.catalog.transition = transition
        self.addCleanup(setattr, self.platform.catalog, "transition", original_transition)
        with self.assertRaises(PlatformUnavailableError):
            self.platform.cancel_job(self.tenant_a, submission.job_id)
        self.assertEqual(
            self.platform.get_job(self.tenant_a, submission.job_id).status,
            JobStatus.RUNNING,
        )

        reject_intent[0] = False
        cancelling = self.platform.cancel_job(self.tenant_a, submission.job_id)
        self.assertEqual(cancelling.status, JobStatus.CANCELLING)
        release.set()
        self.assertEqual(
            self._wait_for_job(submission.job_id.value).status,
            JobStatus.CANCELLED,
        )
        self.assertEqual(
            self.platform.get_knowledge_base(
                self.tenant_a,
                submission.knowledge_base.resource_id,
            ).status,
            KnowledgeBaseStatus.FAILED,
        )

    def test_durable_cancel_wins_before_the_in_memory_signal_arrives(self) -> None:
        worker_started = threading.Event()
        release_worker = threading.Event()
        original_prepare = self.service.prepare_index

        def blocked_prepare(paths, *, namespace=""):
            worker_started.set()
            self.assertTrue(release_worker.wait(timeout=2))
            return original_prepare(paths, namespace=namespace)

        self.service.prepare_index = blocked_prepare
        self.addCleanup(setattr, self.service, "prepare_index", original_prepare)
        self.addCleanup(release_worker.set)
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Cancellation ordering",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="cancellation-ordering-request",
        )
        self.assertTrue(worker_started.wait(timeout=2))

        original_cancel = self.platform.jobs.cancel

        def delayed_signal(tenant_id, job_id):
            release_worker.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                snapshot = self.platform.jobs.get(tenant_id, job_id)
                if snapshot.status.terminal:
                    break
                time.sleep(0.005)
            else:
                self.fail("worker did not observe the durable cancellation intent")
            return original_cancel(tenant_id, job_id)

        self.platform.jobs.cancel = delayed_signal
        self.addCleanup(setattr, self.platform.jobs, "cancel", original_cancel)
        cancelled = self.platform.cancel_job(self.tenant_a, submission.job_id)

        self.assertEqual(cancelled.status, JobStatus.CANCELLED)
        record = self.platform.get_knowledge_base(
            self.tenant_a,
            submission.knowledge_base.resource_id,
        )
        self.assertEqual(record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(record.error_code.value, "index_cancelled")

    def test_cancel_intent_survives_job_eviction_and_restart(self) -> None:
        started = threading.Event()
        release = threading.Event()
        original_create = self.service.create_prepared_index

        def blocked_create(ingestion):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return original_create(ingestion)

        self.service.create_prepared_index = blocked_create
        self.addCleanup(setattr, self.service, "create_prepared_index", original_create)
        self.addCleanup(release.set)
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Durable cancellation",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="durable-cancellation-request",
        )
        self.assertTrue(started.wait(timeout=2))

        original_transition = self.platform.catalog.transition
        reject_failed = [True]

        def transition(principal_value, resource_id, target, **kwargs):
            if reject_failed[0] and target is KnowledgeBaseStatus.FAILED:
                raise OSError("transient catalog failure")
            return original_transition(principal_value, resource_id, target, **kwargs)

        self.platform.catalog.transition = transition
        self.addCleanup(setattr, self.platform.catalog, "transition", original_transition)
        self.assertEqual(
            self.platform.cancel_job(self.tenant_a, submission.job_id).status,
            JobStatus.CANCELLING,
        )
        release.set()
        self.assertEqual(
            self._wait_for_job(submission.job_id.value).status,
            JobStatus.CANCELLED,
        )
        self.assertEqual(
            self.platform.get_knowledge_base(
                self.tenant_a,
                submission.knowledge_base.resource_id,
            ).status,
            KnowledgeBaseStatus.CANCELLING,
        )

        for index in range(16):
            filler = self.platform.create_knowledge_base(
                self.tenant_a,
                display_name=f"Cancellation filler {index}",
                documents=(UploadDocument(f"filler-{index}.txt", b"filler"),),
                idempotency_key=f"cancellation-filler-{index}",
            )
            self.assertEqual(
                self._wait_for_job(filler.job_id.value).status,
                JobStatus.SUCCEEDED,
            )
        with self.assertRaises(JobNotFoundError):
            self.platform.get_job(self.tenant_a, submission.job_id)

        reject_failed[0] = False
        restarted_jobs = JobManager(max_workers=2, max_jobs=16, ttl_seconds=30)
        restarted = RagPlatform(
            settings=self.settings,
            service=FakeService(),
            catalog=KnowledgeBaseCatalog(self.root / "catalog.sqlite3"),
            file_store=TenantFileStore(
                self.root / "documents",
                max_file_bytes=10_000,
                max_total_bytes=50_000,
                max_files_per_tenant=20,
            ),
            jobs=restarted_jobs,
            idempotency=IdempotencyStore(self.root / "idempotency.sqlite3"),
        )
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.recover_incomplete((self.tenant_a,)), 1)
        self.assertEqual(
            restarted.get_knowledge_base(
                self.tenant_a,
                submission.knowledge_base.resource_id,
            ).status,
            KnowledgeBaseStatus.FAILED,
        )

    def test_replay_repairs_an_evicted_durable_cancellation(self) -> None:
        started = threading.Event()
        release = threading.Event()
        original_create = self.service.create_prepared_index

        def blocked_create(ingestion):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return original_create(ingestion)

        self.service.create_prepared_index = blocked_create
        self.addCleanup(setattr, self.service, "create_prepared_index", original_create)
        self.addCleanup(release.set)
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Replay cancellation",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="replay-cancellation-request",
        )
        self.assertTrue(started.wait(timeout=2))

        original_transition = self.platform.catalog.transition
        reject_failed = [True]

        def transition(principal_value, resource_id, target, **kwargs):
            if reject_failed[0] and target is KnowledgeBaseStatus.FAILED:
                raise OSError("transient catalog failure")
            return original_transition(principal_value, resource_id, target, **kwargs)

        self.platform.catalog.transition = transition
        self.addCleanup(setattr, self.platform.catalog, "transition", original_transition)
        self.assertEqual(
            self.platform.cancel_job(self.tenant_a, submission.job_id).status,
            JobStatus.CANCELLING,
        )
        release.set()
        self.assertEqual(
            self._wait_for_job(submission.job_id.value).status,
            JobStatus.CANCELLED,
        )

        for index in range(16):
            filler = self.platform.create_knowledge_base(
                self.tenant_a,
                display_name=f"Replay filler {index}",
                documents=(UploadDocument(f"filler-{index}.txt", b"filler"),),
                idempotency_key=f"replay-cancellation-filler-{index}",
            )
            self.assertEqual(
                self._wait_for_job(filler.job_id.value).status,
                JobStatus.SUCCEEDED,
            )
        with self.assertRaises(JobNotFoundError):
            self.platform.get_job(self.tenant_a, submission.job_id)

        reject_failed[0] = False
        replay = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Replay cancellation",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="replay-cancellation-request",
        )
        self.assertTrue(replay.replayed)
        self.assertNotEqual(replay.job_id, submission.job_id)
        self.assertEqual(
            self._wait_for_job(replay.job_id.value).status,
            JobStatus.SUCCEEDED,
        )
        record = self.platform.get_knowledge_base(
            self.tenant_a,
            submission.knowledge_base.resource_id,
        )
        self.assertEqual(record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(record.error_code.value, "index_cancelled")

    def test_replay_repairs_a_retained_terminal_durable_cancellation(self) -> None:
        started = threading.Event()
        release = threading.Event()
        original_create = self.service.create_prepared_index

        def blocked_create(ingestion):
            started.set()
            self.assertTrue(release.wait(timeout=2))
            return original_create(ingestion)

        self.service.create_prepared_index = blocked_create
        self.addCleanup(setattr, self.service, "create_prepared_index", original_create)
        self.addCleanup(release.set)
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Retained cancellation",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="retained-cancellation-request",
        )
        self.assertTrue(started.wait(timeout=2))

        original_transition = self.platform.catalog.transition
        reject_failed = [True]

        def transition(principal_value, resource_id, target, **kwargs):
            if reject_failed[0] and target is KnowledgeBaseStatus.FAILED:
                raise OSError("transient catalog failure")
            return original_transition(principal_value, resource_id, target, **kwargs)

        self.platform.catalog.transition = transition
        self.addCleanup(setattr, self.platform.catalog, "transition", original_transition)
        self.assertEqual(
            self.platform.cancel_job(self.tenant_a, submission.job_id).status,
            JobStatus.CANCELLING,
        )
        release.set()
        self.assertEqual(
            self._wait_for_job(submission.job_id.value).status,
            JobStatus.CANCELLED,
        )

        reject_failed[0] = False
        replay = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Retained cancellation",
            documents=(UploadDocument("cancel.txt", b"cancel evidence"),),
            idempotency_key="retained-cancellation-request",
        )
        self.assertTrue(replay.replayed)
        self.assertNotEqual(replay.job_id, submission.job_id)
        self.assertEqual(
            self._wait_for_job(replay.job_id.value).status,
            JobStatus.SUCCEEDED,
        )
        record = self.platform.get_knowledge_base(
            self.tenant_a,
            submission.knowledge_base.resource_id,
        )
        self.assertEqual(record.status, KnowledgeBaseStatus.FAILED)
        self.assertEqual(record.error_code.value, "index_cancelled")

    def test_ready_commit_wins_a_late_cancel_without_state_split(self) -> None:
        committed = threading.Event()
        release = threading.Event()
        counter = self.platform.metrics.index_tasks_total
        original_increment = counter.increment

        def blocked_increment(*, labels):
            committed.set()
            self.assertTrue(release.wait(timeout=2))
            original_increment(labels=labels)

        counter.increment = blocked_increment
        self.addCleanup(setattr, counter, "increment", original_increment)
        self.addCleanup(release.set)
        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Committed knowledge base",
            documents=(UploadDocument("committed.txt", b"committed evidence"),),
            idempotency_key="committed-cancel-race",
        )
        self.assertTrue(committed.wait(timeout=2))
        self.assertEqual(
            self.platform.get_knowledge_base(
                self.tenant_a,
                submission.knowledge_base.resource_id,
            ).status,
            KnowledgeBaseStatus.READY,
        )

        cancelling = self.platform.cancel_job(self.tenant_a, submission.job_id)
        self.assertEqual(cancelling.status, JobStatus.CANCELLING)
        release.set()
        self.assertEqual(
            self._wait_for_job(submission.job_id.value).status,
            JobStatus.SUCCEEDED,
        )
        self.assertEqual(
            self.platform.get_knowledge_base(
                self.tenant_a,
                submission.knowledge_base.resource_id,
            ).status,
            KnowledgeBaseStatus.READY,
        )

    def test_restart_repairs_unbound_idempotency_and_returns_a_pollable_job(self) -> None:
        record = self._create_ready()
        reservation_id = record.idempotency_reservation_id
        self.assertIsNotNone(reservation_id)
        with closing(sqlite3.connect(self.root / "idempotency.sqlite3")) as connection:
            connection.execute(
                """
                UPDATE idempotency_entries
                SET resource_id = NULL, job_id = NULL
                WHERE reservation_id = ?
                """,
                (reservation_id,),
            )
            connection.commit()

        self.jobs.shutdown(wait=True)
        restarted_jobs = JobManager(max_workers=2, max_jobs=16, ttl_seconds=30)
        restarted = RagPlatform(
            settings=self.settings,
            service=FakeService(),
            catalog=KnowledgeBaseCatalog(self.root / "catalog.sqlite3"),
            file_store=TenantFileStore(
                self.root / "documents",
                max_file_bytes=10_000,
                max_total_bytes=50_000,
                max_files_per_tenant=20,
            ),
            jobs=restarted_jobs,
            idempotency=IdempotencyStore(self.root / "idempotency.sqlite3"),
        )
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.recover_incomplete((self.tenant_a,)), 0)

        replay = restarted.create_knowledge_base(
            self.tenant_a,
            display_name="Engineering handbook",
            documents=(UploadDocument("guide.txt", b"RAG evidence"),),
            idempotency_key="create-engineering-handbook",
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.knowledge_base.resource_id, record.resource_id)

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = restarted.get_job(self.tenant_a, replay.job_id)
            if snapshot.status.terminal:
                break
            time.sleep(0.01)
        else:
            self.fail("recovered status job did not complete")
        self.assertEqual(snapshot.status, JobStatus.SUCCEEDED)

    def test_ready_replay_replaces_an_interrupted_archived_job(self) -> None:
        record = self._create_ready()
        reservation_id = record.idempotency_reservation_id
        self.assertIsNotNone(reservation_id)
        with closing(sqlite3.connect(self.root / "idempotency.sqlite3")) as connection:
            bound_job_id = connection.execute(
                "SELECT job_id FROM idempotency_entries WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()[0]

        archive = SqliteJobSnapshotStore(self.root / "jobs.sqlite3")
        now = time.time()
        archive.put(
            self.tenant_a.tenant_id.value,
            JobSnapshot(
                job_id=JobId(bound_job_id),
                status=JobStatus.FAILED,
                created_at=now - 2,
                updated_at=now - 1,
                started_at=now - 2,
                finished_at=now - 1,
                error_code="worker_restarted",
                error_message="job execution was interrupted by a process restart",
            ),
        )
        restarted_jobs = JobManager(
            max_workers=2,
            max_jobs=16,
            ttl_seconds=30,
            snapshot_store=archive,
        )
        restarted = RagPlatform(
            settings=self.settings,
            service=FakeService(),
            catalog=KnowledgeBaseCatalog(self.root / "catalog.sqlite3"),
            file_store=TenantFileStore(
                self.root / "documents",
                max_file_bytes=10_000,
                max_total_bytes=50_000,
                max_files_per_tenant=20,
            ),
            jobs=restarted_jobs,
            idempotency=IdempotencyStore(self.root / "idempotency.sqlite3"),
        )
        self.addCleanup(restarted.close)

        replay = restarted.create_knowledge_base(
            self.tenant_a,
            display_name="Engineering handbook",
            documents=(UploadDocument("guide.txt", b"RAG evidence"),),
            idempotency_key="create-engineering-handbook",
        )

        self.assertTrue(replay.replayed)
        self.assertNotEqual(replay.job_id.value, bound_job_id)
        self.assertEqual(
            self._wait_for_job_on(restarted, replay.job_id.value).status,
            JobStatus.SUCCEEDED,
        )

    def test_idempotency_key_cannot_be_reused_for_different_content(self) -> None:
        self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Conflict handbook",
            documents=(UploadDocument("guide.txt", b"first"),),
            idempotency_key="conflicting-create-key",
        )
        with self.assertRaises(IdempotencyConflictError):
            self.platform.create_knowledge_base(
                self.tenant_a,
                display_name="Conflict handbook",
                documents=(UploadDocument("guide.txt", b"different"),),
                idempotency_key="conflicting-create-key",
            )

    def test_catalog_failure_releases_idempotency_reservation_for_retry(self) -> None:
        with patch.object(
            self.platform.catalog,
            "create",
            side_effect=RuntimeError("catalog unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
                self.platform.create_knowledge_base(
                    self.tenant_a,
                    display_name="Retryable handbook",
                    documents=(UploadDocument("guide.txt", b"same request"),),
                    idempotency_key="retryable-create-key",
                )

        submission = self.platform.create_knowledge_base(
            self.tenant_a,
            display_name="Retryable handbook",
            documents=(UploadDocument("guide.txt", b"same request"),),
            idempotency_key="retryable-create-key",
        )
        self.assertFalse(submission.replayed)


if __name__ == "__main__":
    unittest.main()
