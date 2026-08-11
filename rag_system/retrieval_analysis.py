"""Slice-aware diagnostics for governed retrieval benchmark runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_system.benchmark import (
    RetrievalBenchmarkCase,
    RetrievalBenchmarkRun,
    RetrievalPrediction,
    retrieval_dataset_digest,
)
from rag_system.benchmark_suite import (
    RetrievalBenchmarkSuite,
    RetrievalCaseFamily,
)
from rag_system.evaluation import EvaluationCase, EvaluationMetrics, evaluate_cases
from rag_system.evaluation_suite import EvaluationSuiteError


SCHEMA_VERSION = 1
_DIMENSIONS = ("split", "category", "difficulty", "expected_route")


@dataclass(frozen=True, slots=True)
class RetrievalConfidenceSummary:
    minimum: float
    p50: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum": round(self.minimum, 12),
            "p50": round(self.p50, 12),
            "maximum": round(self.maximum, 12),
        }


@dataclass(frozen=True, slots=True)
class RetrievalSignalSummary:
    """Aggregate privacy-safe routing evidence for one governed slice."""

    top_score: RetrievalConfidenceSummary
    margin: RetrievalConfidenceSummary
    lexical_support: RetrievalConfidenceSummary
    ranker_agreement_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_score": self.top_score.to_dict(),
            "margin": self.margin.to_dict(),
            "lexical_support": self.lexical_support.to_dict(),
            "ranker_agreement_rate": round(self.ranker_agreement_rate, 12),
        }


@dataclass(frozen=True, slots=True)
class RetrievalMetricSlice:
    dimension: str
    value: str
    case_count: int
    retrieval_case_count: int
    passed_case_count: int
    retrieval_failure_case_ids: tuple[str, ...]
    route_failure_case_ids: tuple[str, ...]
    confidence: RetrievalConfidenceSummary
    routing_signals: RetrievalSignalSummary
    metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "case_count": self.case_count,
            "retrieval_case_count": self.retrieval_case_count,
            "passed_case_count": self.passed_case_count,
            "failed_case_count": self.case_count - self.passed_case_count,
            "retrieval_failure_case_ids": list(self.retrieval_failure_case_ids),
            "route_failure_case_ids": list(self.route_failure_case_ids),
            "confidence": self.confidence.to_dict(),
            "routing_signals": self.routing_signals.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RouteConfusion:
    expected: str
    predicted: str
    case_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "expected": self.expected,
            "predicted": self.predicted,
            "case_count": self.case_count,
        }


@dataclass(frozen=True, slots=True)
class RetrievalSuiteBenchmarkReport:
    suite_id: str
    suite_digest: str
    evaluated_split: str
    benchmark: RetrievalBenchmarkRun
    route_confusion: tuple[RouteConfusion, ...]
    slices: tuple[RetrievalMetricSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.benchmark.to_dict())
        payload.update({
            "analysis_schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_digest": self.suite_digest,
            "evaluated_split": self.evaluated_split,
            "route_confusion": [item.to_dict() for item in self.route_confusion],
            "slices": [item.to_dict() for item in self.slices],
        })
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        metrics = self.benchmark.report.metrics
        lines = [
            f"# 检索套件报告：{self.suite_id}",
            "",
            f"- 套件摘要：`{self.suite_digest}`",
            f"- 运行数据摘要：`{self.benchmark.report.dataset_digest}`",
            f"- 评测分段：`{self.evaluated_split}`",
            f"- 样例数：`{self.benchmark.report.case_count}`",
            "",
            "## 总体指标",
            "",
            f"| Recall@{self.benchmark.report.top_k} | MRR | nDCG | 路由准确率 |",
            "| ---: | ---: | ---: | ---: |",
            f"| {metrics.recall_at_k:.4f} | {metrics.mrr_at_k:.4f} | "
            f"{metrics.ndcg_at_k:.4f} | {metrics.route_accuracy:.4f} |",
            "",
            "## 路由混淆矩阵",
            "",
            "| 期望 | 实际 | 样例 |",
            "| --- | --- | ---: |",
        ]
        lines.extend(
            f"| {item.expected} | {item.predicted} | {item.case_count} |"
            for item in self.route_confusion
        )
        lines.extend(
            [
                "",
                "## 质量切片",
                "",
                "| 维度 | 值 | 通过/样例 | Recall | MRR | nDCG | 路由 | 置信度 min/p50/max | top p50 | margin p50 | lexical p50 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for item in self.slices:
            values = item.metrics
            retrieval_applicable = item.retrieval_case_count > 0
            confidence = item.confidence
            lines.append(
                f"| {item.dimension} | {item.value} | "
                f"{item.passed_case_count}/{item.case_count} | "
                f"{_display(values.recall_at_k, retrieval_applicable)} | "
                f"{_display(values.mrr_at_k, retrieval_applicable)} | "
                f"{_display(values.ndcg_at_k, retrieval_applicable)} | "
                f"{values.route_accuracy:.4f} | "
                f"{confidence.minimum:.4f}/{confidence.p50:.4f}/{confidence.maximum:.4f} | "
                f"{item.routing_signals.top_score.p50:.4f} | "
                f"{item.routing_signals.margin.p50:.4f} | "
                f"{item.routing_signals.lexical_support.p50:.4f} |"
            )
        lines.extend(["", "## 失败定位", ""])
        failures = tuple(
            prediction for prediction in self.benchmark.predictions if not prediction.passed
        )
        if failures:
            lines.extend(
                f"- `{item.case_id}`：route={item.expected_route.value}→"
                f"{item.predicted_route.value}, missing="
                f"{','.join(item.missing_relevant_sources) or 'none'}, "
                f"confidence={item.confidence:.4f}, "
                f"top={item.routing_signal.top_score:.4f}, "
                f"margin={item.routing_signal.margin:.4f}, "
                f"lexical={item.routing_signal.lexical_support:.4f}"
                for item in failures
            )
        else:
            lines.append("无。")
        return "\n".join(lines) + "\n"


def build_retrieval_suite_report(
    suite: RetrievalBenchmarkSuite,
    benchmark: RetrievalBenchmarkRun,
    *,
    split: str | None = None,
) -> RetrievalSuiteBenchmarkReport:
    """Bind retrieval predictions to suite metadata and calculate diagnostic slices."""

    selected = _selected_cases(suite, split)
    cases = tuple(item[1] for item in selected)
    expected_ids = tuple(case.case_id for case in cases)
    actual_ids = tuple(item.case_id for item in benchmark.predictions)
    if benchmark.report.dataset_digest != retrieval_dataset_digest(cases):
        raise EvaluationSuiteError("retrieval benchmark digest does not match selected suite cases")
    if actual_ids != expected_ids:
        raise EvaluationSuiteError("retrieval prediction order does not match selected suite cases")
    if (
        benchmark.report.case_count != len(cases)
        or benchmark.latency.case_count != len(cases)
    ):
        raise EvaluationSuiteError("retrieval benchmark counts do not match selected suite cases")

    grouped: dict[
        tuple[str, str],
        list[tuple[RetrievalBenchmarkCase, RetrievalPrediction]],
    ] = defaultdict(list)
    for (family, case), prediction in zip(
        selected, benchmark.predictions, strict=True
    ):
        pair = (case, prediction)
        grouped[("split", family.split)].append(pair)
        grouped[("category", family.category)].append(pair)
        grouped[("difficulty", family.difficulty)].append(pair)
        grouped[("expected_route", family.expected_route.value)].append(pair)

    order = {name: index for index, name in enumerate(_DIMENSIONS)}
    slices = tuple(
        _metric_slice(dimension, value, pairs, benchmark.report.top_k)
        for (dimension, value), pairs in sorted(
            grouped.items(), key=lambda item: (order[item[0][0]], item[0][1])
        )
    )
    confusion_counts = Counter(
        (item.expected_route.value, item.predicted_route.value)
        for item in benchmark.predictions
    )
    confusion = tuple(
        RouteConfusion(expected, predicted, count)
        for (expected, predicted), count in sorted(confusion_counts.items())
    )
    return RetrievalSuiteBenchmarkReport(
        suite_id=suite.suite_id,
        suite_digest=suite.bundle_digest,
        evaluated_split=split or "all",
        benchmark=benchmark,
        route_confusion=confusion,
        slices=slices,
    )


def _selected_cases(
    suite: RetrievalBenchmarkSuite, split: str | None
) -> tuple[tuple[RetrievalCaseFamily, RetrievalBenchmarkCase], ...]:
    if split is not None:
        suite.cases_for_split(split)
    return tuple(
        (family, case)
        for family in suite.families
        if split is None or family.split == split
        for case in family.expand()
    )


def _metric_slice(
    dimension: str,
    value: str,
    pairs: Sequence[tuple[RetrievalBenchmarkCase, RetrievalPrediction]],
    top_k: int,
) -> RetrievalMetricSlice:
    predictions = tuple(item[1] for item in pairs)
    evaluation = evaluate_cases(
        tuple(_evaluation_case(case, prediction) for case, prediction in pairs),
        top_k=top_k,
    )
    confidences = sorted(item.confidence for item in predictions)
    top_scores = sorted(item.routing_signal.top_score for item in predictions)
    margins = sorted(item.routing_signal.margin for item in predictions)
    lexical_supports = sorted(
        item.routing_signal.lexical_support for item in predictions
    )
    return RetrievalMetricSlice(
        dimension=dimension,
        value=value,
        case_count=len(pairs),
        retrieval_case_count=evaluation.retrieval_case_count,
        passed_case_count=sum(item.passed for item in predictions),
        retrieval_failure_case_ids=tuple(
            item.case_id for item in predictions if not item.retrieval_correct
        ),
        route_failure_case_ids=tuple(
            item.case_id for item in predictions if not item.route_correct
        ),
        confidence=RetrievalConfidenceSummary(
            minimum=confidences[0],
            p50=_percentile(confidences, 0.5),
            maximum=confidences[-1],
        ),
        routing_signals=RetrievalSignalSummary(
            top_score=_summary(top_scores),
            margin=_summary(margins),
            lexical_support=_summary(lexical_supports),
            ranker_agreement_rate=sum(
                item.routing_signal.ranker_agreement for item in predictions
            )
            / len(predictions),
        ),
        metrics=evaluation.metrics,
    )


def _evaluation_case(
    case: RetrievalBenchmarkCase, prediction: RetrievalPrediction
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case.case_id,
        question=case.question,
        relevance=case.relevance,
        retrieved_ids=prediction.retrieved_sources,
        expected_route=case.expected_route,
        predicted_route=prediction.predicted_route,
        allowed_citation_ids=(),
        answer="",
        citation_required=False,
    )


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _display(value: float, applicable: bool) -> str:
    return f"{value:.4f}" if applicable else "N/A"


def _summary(ordered: Sequence[float]) -> RetrievalConfidenceSummary:
    return RetrievalConfidenceSummary(
        minimum=ordered[0],
        p50=_percentile(ordered, 0.5),
        maximum=ordered[-1],
    )


__all__ = [
    "RetrievalConfidenceSummary",
    "RetrievalMetricSlice",
    "RetrievalSignalSummary",
    "RetrievalSuiteBenchmarkReport",
    "RouteConfusion",
    "build_retrieval_suite_report",
]
