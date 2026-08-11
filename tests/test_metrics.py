from __future__ import annotations

import math
import unittest
from concurrent.futures import ThreadPoolExecutor

from rag_system.metrics import (
    MetricCardinalityError,
    MetricRegistry,
    MetricValidationError,
    create_operational_metrics,
)


class MetricRegistryTests(unittest.TestCase):
    def test_counter_gauge_and_histogram_snapshots_and_reset(self) -> None:
        registry = MetricRegistry()
        counter = registry.counter(
            "jobs_total",
            "Completed jobs.",
            label_names=("outcome",),
        )
        gauge = registry.gauge("queue_depth", "Current queue depth.")
        histogram = registry.histogram(
            "job_duration_seconds",
            "Job duration.",
            buckets=(0.1, 0.5, 1.0),
            label_names=("route",),
        )

        counter.increment(2, labels={"outcome": "success"})
        gauge.set(3)
        gauge.decrement()
        histogram.observe(0.05, labels={"route": "local"})
        histogram.observe(0.7, labels={"route": "local"})

        snapshot = registry.snapshot()
        self.assertEqual(snapshot["jobs_total"]["series"][0]["value"], 2.0)
        self.assertEqual(snapshot["queue_depth"]["series"][0]["value"], 2.0)
        histogram_series = snapshot["job_duration_seconds"]["series"][0]
        self.assertEqual(histogram_series["count"], 2)
        self.assertEqual(histogram_series["buckets"], {0.1: 1, 0.5: 1, 1.0: 2})

        registry.reset()
        self.assertTrue(all(not value["series"] for value in registry.snapshot().values()))

    def test_names_labels_values_and_cardinality_are_strictly_bounded(self) -> None:
        registry = MetricRegistry(max_series_per_metric=2)
        with self.assertRaises(MetricValidationError):
            registry.gauge("Bad-Name", "Bad metric.")
        for sensitive in ("question", "document_id", "tenant", "user_text", "api_key"):
            with self.subTest(label=sensitive):
                with self.assertRaises(MetricValidationError):
                    registry.gauge(
                        f"bad_{sensitive.replace('api_key', 'key')}",
                        "Rejected label.",
                        label_names=(sensitive,),
                    )

        counter = registry.counter(
            "bounded_total",
            "Bounded series.",
            label_names=("route",),
            max_series=2,
        )
        counter.increment(labels={"route": "local"})
        counter.increment(labels={"route": "web"})
        with self.assertRaises(MetricCardinalityError):
            counter.increment(labels={"route": "hybrid"})
        with self.assertRaises(MetricValidationError):
            counter.increment(labels={"route": "x" * 65})
        with self.assertRaises(MetricValidationError):
            counter.increment(labels={"route": "bad\x85value"})
        with self.assertRaises(MetricValidationError):
            counter.increment(labels={"unexpected": "local"})
        self.assertEqual(len(counter.snapshot()["series"]), 2)

    def test_prometheus_output_is_sorted_and_escaped(self) -> None:
        registry = MetricRegistry()
        counter = registry.counter(
            "escaped_total",
            "First\\line\nSecond line.",
            label_names=("kind",),
        )
        registry.gauge("alpha_value", "Alphabetically first.").set(3)
        counter.increment(2, labels={"kind": 'quote" slash\\ newline\nvalue'})

        rendered = registry.render_prometheus()

        self.assertLess(rendered.index("# HELP alpha_value"), rendered.index("# HELP escaped_total"))
        self.assertIn("# HELP escaped_total First\\\\line\\nSecond line.", rendered)
        self.assertIn('kind="quote\\" slash\\\\ newline\\nvalue"', rendered)
        self.assertTrue(rendered.endswith("\n"))
        self.assertEqual(rendered, registry.render_prometheus())

    def test_histogram_exposition_has_cumulative_buckets_sum_and_count(self) -> None:
        registry = MetricRegistry()
        histogram = registry.histogram(
            "request_seconds",
            "Request duration.",
            buckets=(0.1, 1.0),
            label_names=("route",),
        )
        histogram.observe(0.05, labels={"route": "local"})
        histogram.observe(0.5, labels={"route": "local"})

        rendered = registry.render_prometheus()
        self.assertIn('request_seconds_bucket{route="local",le="0.1"} 1', rendered)
        self.assertIn('request_seconds_bucket{route="local",le="1"} 2', rendered)
        self.assertIn('request_seconds_bucket{route="local",le="+Inf"} 2', rendered)
        self.assertIn('request_seconds_sum{route="local"} 0.55', rendered)
        self.assertIn('request_seconds_count{route="local"} 2', rendered)

    def test_default_metrics_use_semantic_label_allowlists(self) -> None:
        metrics = create_operational_metrics()
        metrics.requests_total.increment(labels={"operation": "answer", "outcome": "success"})
        metrics.request_duration_seconds.observe(
            0.2,
            labels={"operation": "answer", "route": "local"},
        )
        metrics.retrieval_routes_total.increment(labels={"route": "local"})
        metrics.index_tasks_total.increment(labels={"operation": "build", "outcome": "success"})
        metrics.external_call_errors_total.increment(
            labels={
                "provider": "chat",
                "operation": "generate",
                "error_type": "timeout",
            }
        )

        with self.assertRaises(MetricValidationError):
            metrics.retrieval_routes_total.increment(labels={"route": "raw-tenant-id"})
        rendered = metrics.registry.render_prometheus()
        self.assertNotIn("raw-tenant-id", rendered)
        self.assertIn("rag_requests_total", rendered)
        self.assertIn("rag_external_call_errors_total", rendered)

    def test_invalid_numeric_updates_do_not_create_series(self) -> None:
        registry = MetricRegistry()
        counter = registry.counter("safe_total", "Safe counter.")
        gauge = registry.gauge("safe_gauge", "Safe gauge.")
        histogram = registry.histogram("safe_seconds", "Safe histogram.", buckets=(1.0,))

        for action in (
            lambda: counter.increment(-1),
            lambda: gauge.set(math.nan),
            lambda: gauge.increment(-1),
            lambda: histogram.observe(-0.1),
        ):
            with self.assertRaises(MetricValidationError):
                action()
        self.assertTrue(all(not value["series"] for value in registry.snapshot().values()))

    def test_numeric_overflow_is_rejected_without_corrupting_existing_series(self) -> None:
        registry = MetricRegistry()
        counter = registry.counter("overflow_total", "Overflow-safe counter.")
        gauge = registry.gauge("overflow_gauge", "Overflow-safe gauge.")
        histogram = registry.histogram(
            "overflow_seconds",
            "Overflow-safe histogram.",
            buckets=(1e308,),
        )
        counter.increment(1e308)
        gauge.set(1e308)
        histogram.observe(1e308)

        with self.assertRaises(MetricValidationError):
            counter.increment(1e308)
        with self.assertRaises(MetricValidationError):
            gauge.increment(1e308)
        with self.assertRaises(MetricValidationError):
            histogram.observe(1e308)

        snapshot = registry.snapshot()
        self.assertEqual(snapshot["overflow_total"]["series"][0]["value"], 1e308)
        self.assertEqual(snapshot["overflow_gauge"]["series"][0]["value"], 1e308)
        self.assertEqual(snapshot["overflow_seconds"]["series"][0]["count"], 1)

    def test_concurrent_updates_are_never_lost(self) -> None:
        registry = MetricRegistry()
        counter = registry.counter("events_total", "Concurrent events.")
        gauge = registry.gauge("workers_active", "Concurrent workers.")
        histogram = registry.histogram(
            "event_duration_seconds",
            "Concurrent event duration.",
            buckets=(0.1, 1.0),
        )

        def update(_: int) -> None:
            for _ in range(100):
                counter.increment()
                gauge.increment()
                histogram.observe(0.1)

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(update, range(20)))

        snapshot = registry.snapshot()
        self.assertEqual(snapshot["events_total"]["series"][0]["value"], 2_000.0)
        self.assertEqual(snapshot["workers_active"]["series"][0]["value"], 2_000.0)
        self.assertEqual(snapshot["event_duration_seconds"]["series"][0]["count"], 2_000)


if __name__ == "__main__":
    unittest.main()
