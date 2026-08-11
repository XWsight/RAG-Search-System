from __future__ import annotations

import threading
import unittest

from rag_system.coordination import ResourceJobRegistry, ResourceLockPool
from rag_system.jobs import JobId


class ResourceJobRegistryTests(unittest.TestCase):
    def test_rebinding_keeps_both_directions_consistent(self) -> None:
        registry = ResourceJobRegistry()
        first = JobId("job-first")
        second = JobId("job-second")

        registry.bind("tenant", "resource", first)
        registry.bind("tenant", "resource", second)

        self.assertEqual(registry.job_for("tenant", "resource"), second)
        self.assertIsNone(registry.resource_for("tenant", first))
        self.assertEqual(registry.resource_for("tenant", second), "resource")

    def test_job_reassignment_removes_the_previous_resource_mapping(self) -> None:
        registry = ResourceJobRegistry()
        job_id = JobId("job-shared")
        registry.bind("tenant", "first", job_id)
        registry.bind("tenant", "second", job_id)

        self.assertIsNone(registry.job_for("tenant", "first"))
        self.assertEqual(registry.job_for("tenant", "second"), job_id)

    def test_conditional_unbind_does_not_remove_a_newer_job(self) -> None:
        registry = ResourceJobRegistry()
        current = JobId("job-current")
        registry.bind("tenant", "resource", current)

        removed = registry.unbind_resource(
            "tenant",
            "resource",
            expected_job_id=JobId("job-stale"),
        )

        self.assertFalse(removed)
        self.assertEqual(registry.job_for("tenant", "resource"), current)

    def test_concurrent_bind_and_lookup_preserves_valid_pairs(self) -> None:
        registry = ResourceJobRegistry()
        barrier = threading.Barrier(16)
        errors: list[Exception] = []

        def worker(index: int) -> None:
            try:
                barrier.wait()
                job_id = JobId(f"job-{index}")
                resource_id = f"resource-{index}"
                registry.bind("tenant", resource_id, job_id)
                if registry.resource_for("tenant", job_id) != resource_id:
                    raise AssertionError("reverse mapping was inconsistent")
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))


class ResourceLockPoolTests(unittest.TestCase):
    def test_same_resource_uses_the_same_lock_and_serializes_access(self) -> None:
        pool = ResourceLockPool(slots=8)
        self.assertIs(pool.lock_for("resource"), pool.lock_for("resource"))

        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        def worker() -> None:
            nonlocal active, maximum_active
            with pool.hold("resource"):
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                threading.Event().wait(0.01)
                with state_lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(maximum_active, 1)


if __name__ == "__main__":
    unittest.main()
