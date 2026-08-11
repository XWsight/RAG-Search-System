from __future__ import annotations

import threading
import unittest

from rag_system.job_contracts import (
    CancellationToken,
    JobCancelledError,
    JobId,
    JobSnapshot,
    JobStatus,
)


class JobContractTests(unittest.TestCase):
    def test_job_id_is_normalized_and_bounded(self) -> None:
        self.assertEqual(JobId("  job-123  ").value, "job-123")
        for value in ("", "contains space", "x" * 129):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JobId(value)

    def test_snapshot_enforces_temporal_and_result_invariants(self) -> None:
        valid = JobSnapshot(
            job_id=JobId("job-valid"),
            status=JobStatus.SUCCEEDED,
            created_at=1.0,
            updated_at=3.0,
            started_at=2.0,
            finished_at=3.0,
            result={"ok": True},
        )
        self.assertEqual(valid.result, {"ok": True})

        invalid_arguments = (
            {"status": JobStatus.RUNNING, "finished_at": 3.0},
            {"status": JobStatus.SUCCEEDED, "finished_at": None},
            {"status": JobStatus.RUNNING, "started_at": 4.0},
            {"status": JobStatus.FAILED, "finished_at": 3.0, "result": {"bad": True}},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                JobSnapshot(
                    job_id=JobId("job-invalid"),
                    created_at=1.0,
                    updated_at=3.0,
                    **arguments,
                )

    def test_cancellation_token_exposes_only_cooperative_signal(self) -> None:
        event = threading.Event()
        token = CancellationToken(event)
        self.assertFalse(token.cancelled)
        token.raise_if_cancelled()

        event.set()
        self.assertTrue(token.cancelled)
        with self.assertRaises(JobCancelledError):
            token.raise_if_cancelled()


if __name__ == "__main__":
    unittest.main()
