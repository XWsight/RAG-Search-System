"""Slice-aware diagnostics for governed structured-answer benchmark runs."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_system.answer_benchmark import (
    AnswerBenchmarkMetrics,
    AnswerBenchmarkReport,
    AnswerCaseResult,
    answer_dataset_digest,
    summarize_answer_results,
)
from rag_system.answer_suite import (
    AnswerBenchmarkSuite,
    AnswerSuiteCase,
)
from rag_system.evaluation_suite import EvaluationSuiteError


SCHEMA_VERSION = 1
_DIMENSIONS = ("split", "category", "difficulty", "risk_tag")


@dataclass(frozen=True, slots=True)
class AnswerMetricSlice:
    dimension: str
    value: str
    case_count: int
    fact_count: int
    claim_count: int
    passed_case_count: int
    failure_case_ids: tuple[str, ...]
    metrics: AnswerBenchmarkMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "case_count": self.case_count,
            "fact_count": self.fact_count,
            "claim_count": self.claim_count,
            "passed_case_count": self.passed_case_count,
            "failed_case_count": len(self.failure_case_ids),
            "failure_case_ids": list(self.failure_case_ids),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AnswerSuiteBenchmarkReport:
    suite_id: str
    suite_digest: str
    evaluated_split: str
    benchmark: AnswerBenchmarkReport
    slices: tuple[AnswerMetricSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_digest": self.suite_digest,
            "evaluated_split": self.evaluated_split,
            "benchmark": self.benchmark.to_dict(),
            "slices": [item.to_dict() for item in self.slices],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        metrics = self.benchmark.metrics
        lines = [
            f"# 结构化回答套件报告：{self.suite_id}",
            "",
            f"- 套件摘要：`{self.suite_digest}`",
            f"- 运行数据摘要：`{self.benchmark.dataset_digest}`",
            f"- 评测分段：`{self.evaluated_split}`",
            f"- 样例/事实：`{self.benchmark.case_count}` / `{self.benchmark.fact_count}`",
            "",
            "## 总体指标",
            "",
            "| 契约成功率 | 拒答准确率 | 事实召回率 | 原子结论率 | 归因精确率 |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {metrics.contract_success_rate:.4f} | {metrics.refusal_accuracy:.4f} | "
            f"{metrics.fact_recall:.4f} | {metrics.atomic_claim_rate:.4f} | "
            f"{metrics.attribution_precision:.4f} |",
            "",
            "## 质量切片",
            "",
            "| 维度 | 值 | 通过/样例 | 事实 | 契约 | 拒答 | 召回 | 原子性 | 归因 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in self.slices:
            values = item.metrics
            content_applicable = item.fact_count > 0
            lines.append(
                f"| {item.dimension} | {item.value} | "
                f"{item.passed_case_count}/{item.case_count} | {item.fact_count} | "
                f"{values.contract_success_rate:.4f} | {values.refusal_accuracy:.4f} | "
                f"{_display_metric(values.fact_recall, applicable=content_applicable)} | "
                f"{_display_metric(values.atomic_claim_rate, applicable=content_applicable)} | "
                f"{_display_metric(values.attribution_precision, applicable=content_applicable)} |"
            )
        failures = tuple(
            result.case_id for result in self.benchmark.results if not result.passed
        )
        lines.extend(["", "## 失败样例", ""])
        if failures:
            lines.extend(f"- `{case_id}`" for case_id in failures)
        else:
            lines.append("无。")
        return "\n".join(lines) + "\n"


def build_answer_suite_report(
    suite: AnswerBenchmarkSuite,
    benchmark: AnswerBenchmarkReport,
    *,
    split: str | None = None,
) -> AnswerSuiteBenchmarkReport:
    """Bind a benchmark run to its governed metadata and calculate all slices."""

    selected = _selected_cases(suite, split)
    expected_cases = tuple(item.benchmark_case for item in selected)
    expected_ids = tuple(case.case_id for case in expected_cases)
    actual_ids = tuple(result.case_id for result in benchmark.results)
    expected_facts = sum(len(case.facts) for case in expected_cases)
    if benchmark.dataset_digest != answer_dataset_digest(expected_cases):
        raise EvaluationSuiteError("answer benchmark digest does not match selected suite cases")
    if actual_ids != expected_ids:
        raise EvaluationSuiteError("answer benchmark result order does not match selected suite cases")
    if benchmark.case_count != len(selected) or benchmark.fact_count != expected_facts:
        raise EvaluationSuiteError("answer benchmark counts do not match selected suite cases")

    grouped: dict[tuple[str, str], list[AnswerCaseResult]] = defaultdict(list)
    for item, result in zip(selected, benchmark.results, strict=True):
        grouped[("split", item.split)].append(result)
        grouped[("category", item.category)].append(result)
        grouped[("difficulty", item.difficulty)].append(result)
        for risk_tag in item.risk_tags:
            grouped[("risk_tag", risk_tag)].append(result)

    dimension_order = {name: index for index, name in enumerate(_DIMENSIONS)}
    slices = tuple(
        _metric_slice(dimension, value, results)
        for (dimension, value), results in sorted(
            grouped.items(),
            key=lambda pair: (dimension_order[pair[0][0]], pair[0][1]),
        )
    )
    return AnswerSuiteBenchmarkReport(
        suite_id=suite.suite_id,
        suite_digest=suite.bundle_digest,
        evaluated_split=split or "all",
        benchmark=benchmark,
        slices=slices,
    )


def _selected_cases(
    suite: AnswerBenchmarkSuite, split: str | None
) -> tuple[AnswerSuiteCase, ...]:
    if split is None:
        return suite.cases
    suite.cases_for_split(split)
    return tuple(item for item in suite.cases if item.split == split)


def _metric_slice(
    dimension: str,
    value: str,
    results: Sequence[AnswerCaseResult],
) -> AnswerMetricSlice:
    failures = tuple(result.case_id for result in results if not result.passed)
    return AnswerMetricSlice(
        dimension=dimension,
        value=value,
        case_count=len(results),
        fact_count=sum(result.expected_fact_count for result in results),
        claim_count=sum(result.claim_count for result in results),
        passed_case_count=len(results) - len(failures),
        failure_case_ids=failures,
        metrics=summarize_answer_results(results),
    )


def _display_metric(value: float, *, applicable: bool) -> str:
    return f"{value:.4f}" if applicable else "N/A"


__all__ = [
    "AnswerMetricSlice",
    "AnswerSuiteBenchmarkReport",
    "build_answer_suite_report",
]
