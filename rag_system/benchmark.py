"""Execute annotated retrieval cases against a real retriever without cloud calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.domain import Route, SearchHit
from rag_system.evaluation import (
    DatasetValidationError,
    EvaluationCase,
    EvaluationReport,
    evaluate_cases,
)
from rag_system.ports import Retriever
from rag_system.retrieval import RoutingPolicy


_CASE_FIELDS = frozenset(
    {"case_id", "question", "relevance", "expected_route", "allow_web"}
)


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkCase:
    case_id: str
    question: str
    relevance: tuple[tuple[str, int], ...]
    expected_route: Route
    allow_web: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalPrediction:
    case_id: str
    retrieved_sources: tuple[str, ...]
    predicted_route: Route
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "retrieved_sources": list(self.retrieved_sources),
            "predicted_route": self.predicted_route.value,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkRun:
    report: EvaluationReport
    predictions: tuple[RetrievalPrediction, ...]

    def to_json(self) -> str:
        payload = {
            "report": self.report.to_dict(),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        lines = [self.report.to_markdown().rstrip(), "", "## 逐题结果", ""]
        lines.extend(
            [
                "| 样例 | 路由 | 置信度 | 检索来源 |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for prediction in self.predictions:
            sources = ", ".join(prediction.retrieved_sources) or "—"
            lines.append(
                f"| {prediction.case_id} | {prediction.predicted_route.value} | "
                f"{prediction.confidence:.4f} | {sources} |"
            )
        return "\n".join(lines) + "\n"


def load_retrieval_benchmark(path: str | Path) -> tuple[RetrievalBenchmarkCase, ...]:
    """Load strict JSONL ground truth that contains no system predictions."""

    dataset_path = Path(path)
    try:
        content = dataset_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DatasetValidationError(f"cannot read benchmark {dataset_path}: {error}") from error
    if not content:
        raise DatasetValidationError("benchmark cannot be empty")

    cases: list[RetrievalBenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            raise DatasetValidationError(f"line {line_number}: blank lines are not allowed")
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(f"line {line_number}: invalid JSON") from error
        case = retrieval_case_from_mapping(payload, location=f"line {line_number}")
        if case.case_id in seen_ids:
            raise DatasetValidationError(f"line {line_number}: duplicate case_id {case.case_id!r}")
        seen_ids.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def retrieval_case_from_mapping(
    payload: object,
    *,
    location: str = "case",
) -> RetrievalBenchmarkCase:
    if not isinstance(payload, Mapping):
        raise DatasetValidationError(f"{location}: expected a JSON object")
    keys = set(payload)
    missing = sorted(_CASE_FIELDS - keys)
    unknown = sorted(keys - _CASE_FIELDS)
    if missing:
        raise DatasetValidationError(f"{location}: missing fields: {', '.join(missing)}")
    if unknown:
        raise DatasetValidationError(f"{location}: unknown fields: {', '.join(unknown)}")

    case_id = _nonempty(payload["case_id"], "case_id", location)
    question = _nonempty(payload["question"], "question", location)
    relevance_value = payload["relevance"]
    if not isinstance(relevance_value, Mapping):
        raise DatasetValidationError(f"{location}: relevance must be an object")
    relevance: list[tuple[str, int]] = []
    for source_name, grade in relevance_value.items():
        source = _nonempty(source_name, "relevance key", location)
        if isinstance(grade, bool) or not isinstance(grade, int) or not 1 <= grade <= 3:
            raise DatasetValidationError(
                f"{location}: relevance grade for {source!r} must be an integer from 1 to 3"
            )
        relevance.append((source, grade))
    relevance.sort()

    try:
        expected_route = Route(payload["expected_route"])
    except (TypeError, ValueError):
        raise DatasetValidationError(f"{location}: invalid expected_route") from None
    allow_web = payload["allow_web"]
    if not isinstance(allow_web, bool):
        raise DatasetValidationError(f"{location}: allow_web must be a boolean")
    return RetrievalBenchmarkCase(
        case_id=case_id,
        question=question,
        relevance=tuple(relevance),
        expected_route=expected_route,
        allow_web=allow_web,
    )


def run_retrieval_benchmark(
    cases: Sequence[RetrievalBenchmarkCase],
    retriever: Retriever,
    routing: RoutingPolicy,
    *,
    top_k: int = 5,
) -> RetrievalBenchmarkRun:
    """Run the configured retrieval/routing pipeline and score real predictions."""

    if not cases:
        raise ValueError("cases cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    predictions: list[RetrievalPrediction] = []
    evaluation_cases: list[EvaluationCase] = []
    for case in cases:
        hits = tuple(retriever.search(case.question, top_k=top_k))
        decision = routing.decide(hits, allow_web=case.allow_web)
        retrieved_sources = _unique_sources(hits, top_k)
        predictions.append(
            RetrievalPrediction(
                case_id=case.case_id,
                retrieved_sources=retrieved_sources,
                predicted_route=decision.route,
                confidence=decision.confidence,
            )
        )
        evaluation_cases.append(
            EvaluationCase(
                case_id=case.case_id,
                question=case.question,
                relevance=case.relevance,
                retrieved_ids=retrieved_sources,
                expected_route=case.expected_route,
                predicted_route=decision.route,
                allowed_citation_ids=(),
                answer="",
                citation_required=False,
            )
        )

    return RetrievalBenchmarkRun(
        report=evaluate_cases(evaluation_cases, top_k=top_k),
        predictions=tuple(predictions),
    )


def _unique_sources(hits: Sequence[SearchHit], limit: int) -> tuple[str, ...]:
    sources: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        source = hit.chunk.source_name
        if source in seen:
            continue
        sources.append(source)
        seen.add(source)
        if len(sources) >= limit:
            break
    return tuple(sources)


def _nonempty(value: object, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location}: {field} must be a non-empty string")
    return value.strip()


__all__ = [
    "RetrievalBenchmarkCase",
    "RetrievalBenchmarkRun",
    "RetrievalPrediction",
    "load_retrieval_benchmark",
    "retrieval_case_from_mapping",
    "run_retrieval_benchmark",
]
