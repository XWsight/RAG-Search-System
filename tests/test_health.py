from __future__ import annotations

import unittest

from rag_system.health import HealthProbe, ProbeResult, ReadinessMonitor


class ReadinessMonitorTests(unittest.TestCase):
    def test_all_probes_are_evaluated_in_registration_order(self) -> None:
        calls: list[str] = []
        monitor = ReadinessMonitor(
            (
                HealthProbe("catalog", lambda: calls.append("catalog") is None),
                HealthProbe("documents", lambda: calls.append("documents") is None),
            )
        )

        self.assertTrue(monitor.ready())
        self.assertEqual(calls, ["catalog", "documents"])

    def test_false_exception_and_non_boolean_results_fail_closed(self) -> None:
        def unavailable() -> bool:
            raise RuntimeError("internal storage details")

        monitor = ReadinessMonitor(
            (
                HealthProbe("catalog", lambda: False),
                HealthProbe("documents", unavailable),
                HealthProbe("jobs", lambda: 1),
            )
        )

        self.assertEqual(
            monitor.snapshot(),
            (
                ProbeResult("catalog", False),
                ProbeResult("documents", False),
                ProbeResult("jobs", False),
            ),
        )
        self.assertFalse(monitor.ready())

    def test_probe_schema_is_fixed_and_validated(self) -> None:
        with self.assertRaises(ValueError):
            ReadinessMonitor(())
        with self.assertRaises(ValueError):
            HealthProbe("Bad Name", lambda: True)
        with self.assertRaises(ValueError):
            ReadinessMonitor(
                (
                    HealthProbe("storage", lambda: True),
                    HealthProbe("storage", lambda: True),
                )
            )


if __name__ == "__main__":
    unittest.main()
