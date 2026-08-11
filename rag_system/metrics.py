"""Thread-safe, low-cardinality operational metrics and Prometheus export."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final


_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SENSITIVE_LABEL_PARTS: Final[tuple[str, ...]] = (
    "authorization",
    "content",
    "document",
    "header",
    "key",
    "prompt",
    "query",
    "question",
    "secret",
    "session",
    "tenant",
    "text",
    "token",
    "user",
)
_RESERVED_LABEL_NAMES: Final[frozenset[str]] = frozenset({"le", "quantile"})
_MAX_LABELS = 8
_MAX_LABEL_VALUE_CHARACTERS = 64
_MAX_HELP_CHARACTERS = 256
_MAX_BUCKETS = 50
_MAX_SERIES_HARD_LIMIT = 10_000


class MetricValidationError(ValueError):
    """A metric definition or update violates a bounded schema."""


class MetricCardinalityError(MetricValidationError):
    """A new label combination would exceed the metric series limit."""


def _validate_metric_name(name: str) -> str:
    if not isinstance(name, str) or not _METRIC_NAME.fullmatch(name):
        raise MetricValidationError("metric names must use lowercase snake_case")
    return name


def _validate_label_name(name: str) -> str:
    if not isinstance(name, str) or not _LABEL_NAME.fullmatch(name):
        raise MetricValidationError("label names must use lowercase snake_case")
    if name in _RESERVED_LABEL_NAMES or any(part in name for part in _SENSITIVE_LABEL_PARTS):
        raise MetricValidationError("sensitive or reserved label names are not allowed")
    return name


def _validate_label_value(value: object) -> str:
    if not isinstance(value, str):
        raise MetricValidationError("label values must be strings")
    if not value or len(value) > _MAX_LABEL_VALUE_CHARACTERS:
        raise MetricValidationError("label values must contain 1 to 64 characters")
    for character in value:
        codepoint = ord(character)
        if (codepoint < 32 and character != "\n") or 127 <= codepoint <= 159:
            raise MetricValidationError("label values contain unsupported control characters")
    return value


def _validate_help(help_text: str) -> str:
    if not isinstance(help_text, str) or not help_text.strip():
        raise MetricValidationError("help text cannot be empty")
    if len(help_text) > _MAX_HELP_CHARACTERS:
        raise MetricValidationError("help text is too long")
    for character in help_text:
        codepoint = ord(character)
        if (codepoint < 32 and character != "\n") or 127 <= codepoint <= 159:
            raise MetricValidationError("help text contains unsupported control characters")
    return help_text


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise MetricValidationError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise MetricValidationError(f"{name} must be a finite number") from None
    if not math.isfinite(numeric):
        raise MetricValidationError(f"{name} must be a finite number")
    return numeric


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_labels(label_names: Sequence[str], values: Sequence[str]) -> str:
    if not label_names:
        return ""
    rendered = ",".join(
        f'{name}="{_escape_label(value)}"' for name, value in zip(label_names, values, strict=True)
    )
    return f"{{{rendered}}}"


class _Metric:
    metric_type = "untyped"

    def __init__(
        self,
        name: str,
        help_text: str,
        *,
        label_names: Sequence[str] = (),
        max_series: int = 100,
        allowed_label_values: Mapping[str, Collection[str]] | None = None,
    ) -> None:
        self.name = _validate_metric_name(name)
        self.help = _validate_help(help_text)
        if isinstance(label_names, str) or len(label_names) > _MAX_LABELS:
            raise MetricValidationError("a metric cannot have more than eight labels")
        validated_names = tuple(_validate_label_name(label) for label in label_names)
        if len(set(validated_names)) != len(validated_names):
            raise MetricValidationError("label names must be unique")
        if (
            not isinstance(max_series, int)
            or isinstance(max_series, bool)
            or not 1 <= max_series <= _MAX_SERIES_HARD_LIMIT
        ):
            raise MetricValidationError("max_series must be between 1 and 10000")
        self.label_names = validated_names
        self.max_series = max_series
        self._allowed_values = self._validate_allowed_values(allowed_label_values or {})
        self._lock = threading.RLock()

    def _validate_allowed_values(
        self,
        supplied: Mapping[str, Collection[str]],
    ) -> dict[str, frozenset[str]]:
        unknown = set(supplied) - set(self.label_names)
        if unknown:
            raise MetricValidationError("allowed values reference unknown labels")
        result: dict[str, frozenset[str]] = {}
        for name, values in supplied.items():
            if isinstance(values, str):
                raise MetricValidationError("allowed label values must be a collection of strings")
            normalized = frozenset(_validate_label_value(value) for value in values)
            if not normalized:
                raise MetricValidationError("allowed label value sets cannot be empty")
            result[name] = normalized
        return result

    def _label_tuple(self, labels: Mapping[str, str] | None) -> tuple[str, ...]:
        supplied = dict(labels or {})
        if set(supplied) != set(self.label_names):
            raise MetricValidationError("labels must exactly match the registered schema")
        values: list[str] = []
        for name in self.label_names:
            value = _validate_label_value(supplied[name])
            allowed = self._allowed_values.get(name)
            if allowed is not None and value not in allowed:
                raise MetricValidationError(f"label value is not allowed for {name}")
            values.append(value)
        return tuple(values)

    def _ensure_cardinality(self, series: Mapping[tuple[str, ...], object], key: tuple[str, ...]) -> None:
        if key not in series and len(series) >= self.max_series:
            raise MetricCardinalityError(f"metric {self.name} reached its series limit")

    def _metadata_lines(self) -> list[str]:
        return [
            f"# HELP {self.name} {_escape_help(self.help)}",
            f"# TYPE {self.name} {self.metric_type}",
        ]

    def render_lines(self) -> list[str]:
        raise NotImplementedError

    def snapshot(self) -> dict[str, object]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Counter(_Metric):
    metric_type = "counter"

    def __init__(self, name: str, help_text: str, **kwargs: object) -> None:
        if not name.endswith("_total"):
            raise MetricValidationError("counter names must end with _total")
        super().__init__(name, help_text, **kwargs)
        self._series: dict[tuple[str, ...], float] = {}

    def increment(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        numeric = _finite_number(amount, name="counter increment")
        if numeric < 0:
            raise MetricValidationError("counter increments cannot be negative")
        key = self._label_tuple(labels)
        with self._lock:
            self._ensure_cardinality(self._series, key)
            updated = self._series.get(key, 0.0) + numeric
            if not math.isfinite(updated):
                raise MetricValidationError("counter value overflowed")
            self._series[key] = updated

    def render_lines(self) -> list[str]:
        with self._lock:
            series = tuple(sorted(self._series.items()))
        lines = self._metadata_lines()
        for labels, value in series:
            lines.append(
                f"{self.name}{_render_labels(self.label_names, labels)} {_format_number(value)}"
            )
        return lines

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            series = [
                {"labels": dict(zip(self.label_names, labels, strict=True)), "value": value}
                for labels, value in sorted(self._series.items())
            ]
        return {"type": self.metric_type, "series": series}

    def reset(self) -> None:
        with self._lock:
            self._series.clear()


class Gauge(_Metric):
    metric_type = "gauge"

    def __init__(self, name: str, help_text: str, **kwargs: object) -> None:
        super().__init__(name, help_text, **kwargs)
        self._series: dict[tuple[str, ...], float] = {}

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        numeric = _finite_number(value, name="gauge value")
        key = self._label_tuple(labels)
        with self._lock:
            self._ensure_cardinality(self._series, key)
            self._series[key] = numeric

    def increment(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        numeric = _finite_number(amount, name="gauge increment")
        if numeric < 0:
            raise MetricValidationError("gauge increments cannot be negative")
        key = self._label_tuple(labels)
        with self._lock:
            self._ensure_cardinality(self._series, key)
            updated = self._series.get(key, 0.0) + numeric
            if not math.isfinite(updated):
                raise MetricValidationError("gauge value overflowed")
            self._series[key] = updated

    def decrement(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        numeric = _finite_number(amount, name="gauge decrement")
        if numeric < 0:
            raise MetricValidationError("gauge decrements cannot be negative")
        key = self._label_tuple(labels)
        with self._lock:
            self._ensure_cardinality(self._series, key)
            updated = self._series.get(key, 0.0) - numeric
            if not math.isfinite(updated):
                raise MetricValidationError("gauge value overflowed")
            self._series[key] = updated

    def render_lines(self) -> list[str]:
        with self._lock:
            series = tuple(sorted(self._series.items()))
        lines = self._metadata_lines()
        for labels, value in series:
            lines.append(
                f"{self.name}{_render_labels(self.label_names, labels)} {_format_number(value)}"
            )
        return lines

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            series = [
                {"labels": dict(zip(self.label_names, labels, strict=True)), "value": value}
                for labels, value in sorted(self._series.items())
            ]
        return {"type": self.metric_type, "series": series}

    def reset(self) -> None:
        with self._lock:
            self._series.clear()


@dataclass(slots=True)
class _HistogramState:
    buckets: list[int]
    count: int = 0
    total: float = 0.0


class Histogram(_Metric):
    metric_type = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        *,
        buckets: Sequence[float],
        **kwargs: object,
    ) -> None:
        super().__init__(name, help_text, **kwargs)
        if not buckets or len(buckets) > _MAX_BUCKETS:
            raise MetricValidationError("histograms need between 1 and 50 buckets")
        normalized = tuple(_finite_number(value, name="histogram bucket") for value in buckets)
        if any(value <= 0 for value in normalized):
            raise MetricValidationError("histogram buckets must be positive")
        if any(
            right <= left
            for left, right in zip(normalized, normalized[1:], strict=False)
        ):
            raise MetricValidationError("histogram buckets must be strictly increasing")
        self.buckets = normalized
        self._series: dict[tuple[str, ...], _HistogramState] = {}

    def observe(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        numeric = _finite_number(value, name="histogram observation")
        if numeric < 0:
            raise MetricValidationError("histogram observations cannot be negative")
        key = self._label_tuple(labels)
        with self._lock:
            self._ensure_cardinality(self._series, key)
            state = self._series.get(key)
            if state is None:
                state = _HistogramState(buckets=[0] * len(self.buckets))
                self._series[key] = state
            updated_total = state.total + numeric
            if not math.isfinite(updated_total):
                raise MetricValidationError("histogram sum overflowed")
            for index, boundary in enumerate(self.buckets):
                if numeric <= boundary:
                    state.buckets[index] += 1
            state.count += 1
            state.total = updated_total

    def render_lines(self) -> list[str]:
        with self._lock:
            series = tuple(
                (
                    labels,
                    tuple(state.buckets),
                    state.count,
                    state.total,
                )
                for labels, state in sorted(self._series.items())
            )
        lines = self._metadata_lines()
        for labels, bucket_counts, count, total in series:
            for boundary, bucket_count in zip(self.buckets, bucket_counts, strict=True):
                names = (*self.label_names, "le")
                values = (*labels, _format_number(boundary))
                lines.append(f"{self.name}_bucket{_render_labels(names, values)} {bucket_count}")
            names = (*self.label_names, "le")
            values = (*labels, "+Inf")
            lines.append(f"{self.name}_bucket{_render_labels(names, values)} {count}")
            rendered = _render_labels(self.label_names, labels)
            lines.append(f"{self.name}_sum{rendered} {_format_number(total)}")
            lines.append(f"{self.name}_count{rendered} {count}")
        return lines

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            series = [
                {
                    "labels": dict(zip(self.label_names, labels, strict=True)),
                    "buckets": dict(zip(self.buckets, state.buckets, strict=True)),
                    "count": state.count,
                    "sum": state.total,
                }
                for labels, state in sorted(self._series.items())
            ]
        return {"type": self.metric_type, "buckets": self.buckets, "series": series}

    def reset(self) -> None:
        with self._lock:
            self._series.clear()


class MetricRegistry:
    """Own a bounded set of metrics and export deterministic snapshots."""

    def __init__(self, *, max_metrics: int = 64, max_series_per_metric: int = 500) -> None:
        if not isinstance(max_metrics, int) or isinstance(max_metrics, bool) or max_metrics < 1:
            raise MetricValidationError("max_metrics must be a positive integer")
        if (
            not isinstance(max_series_per_metric, int)
            or isinstance(max_series_per_metric, bool)
            or max_series_per_metric < 1
        ):
            raise MetricValidationError("max_series_per_metric must be a positive integer")
        self.max_metrics = max_metrics
        self.max_series_per_metric = max_series_per_metric
        self._metrics: dict[str, _Metric] = {}
        self._exported_names: set[str] = set()
        self._lock = threading.RLock()

    def counter(self, name: str, help_text: str, **kwargs: object) -> Counter:
        return self._register(Counter(name, help_text, **self._bounded_kwargs(kwargs)))

    def gauge(self, name: str, help_text: str, **kwargs: object) -> Gauge:
        return self._register(Gauge(name, help_text, **self._bounded_kwargs(kwargs)))

    def histogram(
        self,
        name: str,
        help_text: str,
        *,
        buckets: Sequence[float],
        **kwargs: object,
    ) -> Histogram:
        bounded = self._bounded_kwargs(kwargs)
        return self._register(Histogram(name, help_text, buckets=buckets, **bounded))

    def _bounded_kwargs(self, kwargs: Mapping[str, object]) -> dict[str, object]:
        result = dict(kwargs)
        requested = result.get("max_series", 100)
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise MetricValidationError("max_series must be a positive integer")
        if requested > self.max_series_per_metric:
            raise MetricValidationError("max_series exceeds the registry limit")
        return result

    def _register(self, metric: _Metric):
        exported = {metric.name}
        if isinstance(metric, Histogram):
            exported.update(
                {
                    f"{metric.name}_bucket",
                    f"{metric.name}_count",
                    f"{metric.name}_sum",
                }
            )
        with self._lock:
            if len(self._metrics) >= self.max_metrics:
                raise MetricCardinalityError("registry reached its metric limit")
            if exported & self._exported_names:
                raise MetricValidationError("metric name collides with an existing metric")
            self._metrics[metric.name] = metric
            self._exported_names.update(exported)
        return metric

    def snapshot(self) -> dict[str, dict[str, object]]:
        with self._lock:
            metrics = tuple(sorted(self._metrics.items()))
        return {name: metric.snapshot() for name, metric in metrics}

    def reset(self) -> None:
        with self._lock:
            metrics = tuple(self._metrics.values())
        for metric in metrics:
            metric.reset()

    def render_prometheus(self) -> str:
        with self._lock:
            metrics = tuple(self._metrics[name] for name in sorted(self._metrics))
        lines = [line for metric in metrics for line in metric.render_lines()]
        return "\n".join(lines) + ("\n" if lines else "")


@dataclass(frozen=True, slots=True)
class OperationalMetrics:
    registry: MetricRegistry
    requests_total: Counter
    request_duration_seconds: Histogram
    retrieval_routes_total: Counter
    index_tasks_total: Counter
    external_call_errors_total: Counter


def create_operational_metrics(namespace: str = "rag") -> OperationalMetrics:
    prefix = _validate_metric_name(namespace)
    registry = MetricRegistry()
    outcomes = {"success", "error", "refused", "rate_limited", "unavailable"}
    operations = {"answer", "health", "index", "ingest", "research"}
    routes = {"error", "hybrid", "local", "refused", "retrieval_only", "web"}

    requests = registry.counter(
        f"{prefix}_requests_total",
        "Total application requests.",
        label_names=("operation", "outcome"),
        allowed_label_values={"operation": operations, "outcome": outcomes},
        max_series=32,
    )
    duration = registry.histogram(
        f"{prefix}_request_duration_seconds",
        "Application request duration in seconds.",
        label_names=("operation", "route"),
        allowed_label_values={"operation": operations, "route": routes},
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        max_series=32,
    )
    route_counter = registry.counter(
        f"{prefix}_retrieval_routes_total",
        "Total retrieval route decisions.",
        label_names=("route",),
        allowed_label_values={"route": routes},
        max_series=8,
    )
    index_tasks = registry.counter(
        f"{prefix}_index_tasks_total",
        "Total index lifecycle tasks.",
        label_names=("operation", "outcome"),
        allowed_label_values={
            "operation": {"build", "evict", "load", "remove"},
            "outcome": outcomes,
        },
        max_series=24,
    )
    external_errors = registry.counter(
        f"{prefix}_external_call_errors_total",
        "Total external provider call errors.",
        label_names=("provider", "operation", "error_type"),
        allowed_label_values={
            "provider": {"chat", "embedding", "reranker", "web_search"},
            "operation": {"generate", "embed", "rerank", "search"},
            "error_type": {
                "authentication",
                "protocol",
                "rate_limit",
                "timeout",
                "unavailable",
                "unknown",
            },
        },
        max_series=96,
    )
    return OperationalMetrics(
        registry=registry,
        requests_total=requests,
        request_duration_seconds=duration,
        retrieval_routes_total=route_counter,
        index_tasks_total=index_tasks,
        external_call_errors_total=external_errors,
    )


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricCardinalityError",
    "MetricRegistry",
    "MetricValidationError",
    "OperationalMetrics",
    "create_operational_metrics",
]
