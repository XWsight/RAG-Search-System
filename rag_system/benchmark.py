"""Execute annotated retrieval cases against a real retriever without cloud calls."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from rag_system.retrieval import RoutingPolicy, RoutingSignal


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "relevance": dict(self.relevance),
            "expected_route": self.expected_route.value,
            "allow_web": self.allow_web,
        }


@dataclass(frozen=True, slots=True)
class RetrievalPrediction:
    case_id: str
    relevant_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...]
    missing_relevant_sources: tuple[str, ...]
    expected_route: Route
    predicted_route: Route
    confidence: float
    first_relevant_rank: int | None
    route_correct: bool
    latency_ms: float
    routing_signal: RoutingSignal

    @property
    def retrieval_correct(self) -> bool:
        return not self.missing_relevant_sources

    @property
    def passed(self) -> bool:
        return self.retrieval_correct and self.route_correct

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "relevant_sources": list(self.relevant_sources),
            "retrieved_sources": list(self.retrieved_sources),
            "missing_relevant_sources": list(self.missing_relevant_sources),
            "expected_route": self.expected_route.value,
            "predicted_route": self.predicted_route.value,
            "confidence": round(self.confidence, 12),
            "first_relevant_rank": self.first_relevant_rank,
            "route_correct": self.route_correct,
            "retrieval_correct": self.retrieval_correct,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 6),
            "routing_signal": self.routing_signal.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RetrievalLatency:
    case_count: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "case_count": self.case_count,
            "total_ms": round(self.total_ms, 6),
            "mean_ms": round(self.mean_ms, 6),
            "p50_ms": round(self.p50_ms, 6),
            "p95_ms": round(self.p95_ms, 6),
            "p99_ms": round(self.p99_ms, 6),
            "max_ms": round(self.max_ms, 6),
        }


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkRun:
    report: EvaluationReport
    predictions: tuple[RetrievalPrediction, ...]
    latency: RetrievalLatency

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "report": self.report.to_dict(),
            "latency": self.latency.to_dict(),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        latency = self.latency
        lines = [
            self.report.to_markdown().rstrip(),
            "",
            "## 检索延迟",
            "",
            "> 单进程逐题延迟只适合发现同一环境中的明显回归，不能替代并发压测或生产 SLA。",
            "",
            "| 平均 | P50 | P95 | P99 | 最大值 |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {latency.mean_ms:.3f} ms | {latency.p50_ms:.3f} ms | "
            f"{latency.p95_ms:.3f} ms | {latency.p99_ms:.3f} ms | {latency.max_ms:.3f} ms |",
        ]
        failures = tuple(prediction for prediction in self.predictions if not prediction.passed)
        lines.extend(["", "## 失败诊断", ""])
        if failures:
            lines.extend(
                [
                    "| 样例 | 检索缺失 | 期望路由 | 实际路由 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for prediction in failures:
                missing = ", ".join(prediction.missing_relevant_sources) or "—"
                lines.append(
                    f"| {_markdown_cell(prediction.case_id)} | {_markdown_cell(missing)} | "
                    f"{prediction.expected_route.value} | {prediction.predicted_route.value} |"
                )
        else:
            lines.append("全部样例均通过检索完整性与路由检查。")

        lines.extend(["", "## 逐题结果", ""])
        lines.extend(
            [
                "| 样例 | 状态 | 路由（期望 → 实际） | 首个相关排名 | 置信度 | 延迟 | 检索来源 |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for prediction in self.predictions:
            sources = ", ".join(prediction.retrieved_sources) or "—"
            first_rank = str(prediction.first_relevant_rank or "—")
            status = "通过" if prediction.passed else "失败"
            lines.append(
                f"| {_markdown_cell(prediction.case_id)} | {status} | "
                f"{prediction.expected_route.value} → {prediction.predicted_route.value} | "
                f"{first_rank} | {prediction.confidence:.4f} | "
                f"{prediction.latency_ms:.3f} ms | {_markdown_cell(sources)} |"
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
    clock: Callable[[], float] = time.perf_counter,
) -> RetrievalBenchmarkRun:
    """Run the configured retrieval/routing pipeline and score real predictions."""

    if not cases:
        raise ValueError("cases cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    predictions: list[RetrievalPrediction] = []
    evaluation_cases: list[EvaluationCase] = []
    latencies: list[float] = []
    for case in cases:
        started_at = clock()
        hits = tuple(retriever.search(case.question, top_k=top_k))
        assessment = routing.assess(hits, allow_web=case.allow_web)
        decision = assessment.decision
        finished_at = clock()
        if not all(math.isfinite(value) for value in (started_at, finished_at)):
            raise ValueError("clock must return finite values")
        if finished_at < started_at:
            raise ValueError("clock must be monotonic")
        latency_ms = (finished_at - started_at) * 1_000
        latencies.append(latency_ms)
        retrieved_sources = _unique_sources(hits, top_k)
        relevant_sources = tuple(source for source, _grade in case.relevance)
        relevant_set = set(relevant_sources)
        missing_sources = tuple(
            source for source in relevant_sources if source not in retrieved_sources
        )
        first_relevant_rank = next(
            (
                rank
                for rank, source in enumerate(retrieved_sources, start=1)
                if source in relevant_set
            ),
            None,
        )
        predictions.append(
            RetrievalPrediction(
                case_id=case.case_id,
                relevant_sources=relevant_sources,
                retrieved_sources=retrieved_sources,
                missing_relevant_sources=missing_sources,
                expected_route=case.expected_route,
                predicted_route=decision.route,
                confidence=decision.confidence,
                first_relevant_rank=first_relevant_rank,
                route_correct=decision.route == case.expected_route,
                latency_ms=latency_ms,
                routing_signal=assessment.signal,
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

    report = replace(
        evaluate_cases(evaluation_cases, top_k=top_k),
        dataset_digest=retrieval_dataset_digest(cases),
    )
    return RetrievalBenchmarkRun(
        report=report,
        predictions=tuple(predictions),
        latency=_latency_summary(latencies),
    )


def _latency_summary(values: Sequence[float]) -> RetrievalLatency:
    ordered = sorted(values)
    total = sum(ordered)
    return RetrievalLatency(
        case_count=len(ordered),
        total_ms=total,
        mean_ms=total / len(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        p99_ms=_percentile(ordered, 0.99),
        max_ms=ordered[-1],
    )


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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


def retrieval_dataset_digest(cases: Sequence[RetrievalBenchmarkCase]) -> str:
    """Return the stable digest binding a run to retrieval ground truth."""

    canonical = json.dumps(
        [case.to_dict() for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _nonempty(value: object, field: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location}: {field} must be a non-empty string")
    return value.strip()


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


__all__ = [
    "RetrievalBenchmarkCase",
    "RetrievalBenchmarkRun",
    "RetrievalLatency",
    "RetrievalPrediction",
    "load_retrieval_benchmark",
    "retrieval_dataset_digest",
    "retrieval_case_from_mapping",
    "run_retrieval_benchmark",
]
