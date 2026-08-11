"""Quality gates for non-deterministic structured-answer benchmark runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.answer_benchmark import AnswerBenchmarkReport, AnswerDatasetError


SCHEMA_VERSION = 1
_METRICS = frozenset(
    {
        "contract_success_rate",
        "refusal_accuracy",
        "fact_recall",
        "atomic_claim_rate",
        "attribution_precision",
    }
)
_FIELDS = frozenset({"schema_version", "dataset_digest", "minimum_metrics"})


@dataclass(frozen=True, slots=True)
class AnswerQualityGate:
    dataset_digest: str
    minimum_metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class AnswerGateViolation:
    metric: str
    minimum: float
    actual: float


@dataclass(frozen=True, slots=True)
class AnswerGateResult:
    violations: tuple[AnswerGateViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def load_answer_quality_gate(path: str | Path) -> AnswerQualityGate:
    try:
        content = Path(path).read_text(encoding="utf-8")
        payload = json.loads(content, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AnswerDatasetError(f"cannot load answer quality gate: {error}") from error
    return answer_quality_gate_from_mapping(payload)


def answer_quality_gate_from_mapping(payload: object) -> AnswerQualityGate:
    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise AnswerDatasetError("answer quality gate fields do not match the schema")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise AnswerDatasetError("unsupported answer quality gate schema version")
    digest = payload["dataset_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 16
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise AnswerDatasetError("answer quality gate dataset digest is invalid")
    metrics = payload["minimum_metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != _METRICS:
        raise AnswerDatasetError("answer quality gate metrics do not match the schema")
    resolved: list[tuple[str, float]] = []
    for name in sorted(_METRICS):
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnswerDatasetError(f"answer quality metric {name!r} must be numeric")
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise AnswerDatasetError(f"answer quality metric {name!r} is out of range")
        resolved.append((name, numeric))
    return AnswerQualityGate(digest, tuple(resolved))


def evaluate_answer_quality_gate(
    report: AnswerBenchmarkReport,
    gate: AnswerQualityGate,
) -> AnswerGateResult:
    if report.dataset_digest != gate.dataset_digest:
        raise AnswerDatasetError("answer benchmark dataset digest does not match the gate")
    actual = report.metrics.to_dict()
    violations = tuple(
        AnswerGateViolation(name, minimum, actual[name])
        for name, minimum in gate.minimum_metrics
        if actual[name] < minimum
    )
    return AnswerGateResult(violations)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError("duplicate JSON key")
        resolved[key] = value
    return resolved
