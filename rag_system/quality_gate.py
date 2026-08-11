"""Strict, reproducible quality gates for retrieval benchmark runs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.benchmark import RetrievalBenchmarkRun
from rag_system.evaluation import DatasetValidationError


SCHEMA_VERSION = 1
_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_digest",
        "top_k",
        "minimum_metrics",
        "maximum_latency_ms",
    }
)
_METRIC_NAMES = frozenset(
    {
        "recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "route_accuracy",
        "citation_validity",
        "citation_coverage",
    }
)
_LATENCY_NAMES = frozenset(
    {"total_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"}
)
_DIGEST = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True, slots=True)
class QualityGate:
    dataset_digest: str
    top_k: int
    minimum_metrics: tuple[tuple[str, float], ...]
    maximum_latency_ms: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class QualityViolation:
    field: str
    expectation: str
    expected: str | float | int
    actual: str | float | int

    def to_dict(self) -> dict[str, str | float | int]:
        return {
            "field": self.field,
            "expectation": self.expectation,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    violations: tuple[QualityViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "passed": self.passed,
            "violations": [violation.to_dict() for violation in self.violations],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        if self.passed:
            return "# 质量门禁\n\n结果：**通过**。\n"
        lines = [
            "# 质量门禁",
            "",
            "结果：**失败**。",
            "",
            "| 字段 | 要求 | 期望值 | 实际值 |",
            "| --- | --- | ---: | ---: |",
        ]
        lines.extend(
            f"| {item.field} | {item.expectation} | {item.expected} | {item.actual} |"
            for item in self.violations
        )
        return "\n".join(lines) + "\n"


def load_quality_gate(path: str | Path) -> QualityGate:
    """Load a strict JSON gate and reject ambiguous or unknown configuration."""

    gate_path = Path(path)
    try:
        content = gate_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DatasetValidationError(f"cannot read quality gate {gate_path}: {error}") from error
    if not content.strip():
        raise DatasetValidationError("quality gate cannot be empty")
    try:
        payload = json.loads(content, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            f"quality gate contains invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error
    return quality_gate_from_mapping(payload)


def quality_gate_from_mapping(payload: object) -> QualityGate:
    if not isinstance(payload, Mapping):
        raise DatasetValidationError("quality gate must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise DatasetValidationError("quality gate field names must be strings")
    keys = set(payload)
    missing = sorted(_GATE_FIELDS - keys)
    unknown = sorted(keys - _GATE_FIELDS)
    if missing:
        raise DatasetValidationError(f"quality gate missing fields: {', '.join(missing)}")
    if unknown:
        raise DatasetValidationError(f"quality gate has unknown fields: {', '.join(unknown)}")

    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise DatasetValidationError(
            f"quality gate schema_version must equal {SCHEMA_VERSION}"
        )
    digest = payload["dataset_digest"]
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise DatasetValidationError("quality gate dataset_digest must be 16 lowercase hex digits")
    top_k = payload["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise DatasetValidationError("quality gate top_k must be a positive integer")

    minimum_metrics = _bounded_mapping(
        payload["minimum_metrics"],
        field="minimum_metrics",
        allowed_names=_METRIC_NAMES,
        minimum=0.0,
        maximum=1.0,
        require_value=True,
    )
    maximum_latency = _bounded_mapping(
        payload["maximum_latency_ms"],
        field="maximum_latency_ms",
        allowed_names=_LATENCY_NAMES,
        minimum=0.0,
        maximum=3_600_000.0,
        require_value=False,
        minimum_exclusive=True,
    )
    return QualityGate(
        dataset_digest=digest,
        top_k=top_k,
        minimum_metrics=minimum_metrics,
        maximum_latency_ms=maximum_latency,
    )


def evaluate_quality_gate(
    run: RetrievalBenchmarkRun,
    gate: QualityGate,
) -> QualityGateResult:
    """Compare a run with its frozen dataset contract and minimum quality levels."""

    violations: list[QualityViolation] = []
    report = run.report
    if report.dataset_digest != gate.dataset_digest:
        violations.append(
            QualityViolation(
                "dataset_digest",
                "must equal",
                gate.dataset_digest,
                report.dataset_digest,
            )
        )
    if report.top_k != gate.top_k:
        violations.append(QualityViolation("top_k", "must equal", gate.top_k, report.top_k))

    metrics = report.metrics.to_dict()
    for name, minimum in gate.minimum_metrics:
        if name in {"citation_validity", "citation_coverage"} and not report.citation_case_count:
            violations.append(QualityViolation(name, "minimum", minimum, "N/A"))
            continue
        actual = metrics[name]
        if actual < minimum:
            violations.append(QualityViolation(name, "minimum", minimum, actual))

    for name, maximum in gate.maximum_latency_ms:
        actual = float(getattr(run.latency, name))
        if actual > maximum:
            violations.append(QualityViolation(name, "maximum milliseconds", maximum, actual))

    return QualityGateResult(passed=not violations, violations=tuple(violations))


def _bounded_mapping(
    value: object,
    *,
    field: str,
    allowed_names: frozenset[str],
    minimum: float,
    maximum: float,
    require_value: bool,
    minimum_exclusive: bool = False,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"quality gate {field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DatasetValidationError(f"quality gate {field} field names must be strings")
    if require_value and not value:
        raise DatasetValidationError(f"quality gate {field} cannot be empty")
    unknown = sorted(set(value) - allowed_names)
    if unknown:
        raise DatasetValidationError(
            f"quality gate {field} has unknown fields: {', '.join(unknown)}"
        )
    resolved: list[tuple[str, float]] = []
    for name, raw_value in value.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise DatasetValidationError(f"quality gate {field}.{name} must be a number")
        number = float(raw_value)
        valid_minimum = number > minimum if minimum_exclusive else number >= minimum
        if not math.isfinite(number) or not valid_minimum or number > maximum:
            boundary = f"greater than {minimum}" if minimum_exclusive else f"at least {minimum}"
            raise DatasetValidationError(
                f"quality gate {field}.{name} must be finite, {boundary}, and at most {maximum}"
            )
        resolved.append((str(name), number))
    resolved.sort()
    return tuple(resolved)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetValidationError(f"quality gate contains duplicate key {key!r}")
        result[key] = value
    return result


__all__ = [
    "QualityGate",
    "QualityGateResult",
    "QualityViolation",
    "evaluate_quality_gate",
    "load_quality_gate",
    "quality_gate_from_mapping",
]
