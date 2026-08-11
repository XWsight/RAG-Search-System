"""Deterministic, comparable retrieval ablation experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_system.benchmark import (
    RetrievalBenchmarkCase,
    RetrievalBenchmarkRun,
    RetrievalLatency,
    run_retrieval_benchmark,
    summarize_retrieval_latency,
)
from rag_system.ports import Retriever
from rag_system.routing import RoutingPolicy


_VARIANT_NAME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_SENSITIVE_CONFIGURATION_TERMS = ("credential", "password", "secret", "token", "api_key")
_CONFIG_VALUE = str | int | float | bool


class RetrievalNonDeterminismError(RuntimeError):
    """A retrieval variant changed predictions between repeated runs."""


@dataclass(frozen=True, slots=True)
class RetrievalAblationVariant:
    name: str
    run: RetrievalBenchmarkRun
    latency: RetrievalLatency
    repetitions: int
    configuration: tuple[tuple[str, _CONFIG_VALUE], ...] = ()

    @property
    def failure_case_ids(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self.run.predictions if not item.passed)

    def to_dict(self) -> dict[str, Any]:
        payload = self.run.to_dict()
        payload["latency"] = self.latency.to_dict()
        return {
            "name": self.name,
            "repetitions": self.repetitions,
            "configuration": dict(self.configuration),
            "failure_case_ids": list(self.failure_case_ids),
            "benchmark": payload,
        }


@dataclass(frozen=True, slots=True)
class RetrievalAblationComparison:
    variant: str
    recall_delta: float
    mrr_delta: float
    ndcg_delta: float
    route_delta: float
    mean_latency_ratio: float
    gained_case_ids: tuple[str, ...]
    lost_case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "metric_delta": {
                "recall_at_k": round(self.recall_delta, 12),
                "mrr_at_k": round(self.mrr_delta, 12),
                "ndcg_at_k": round(self.ndcg_delta, 12),
                "route_accuracy": round(self.route_delta, 12),
            },
            "mean_latency_ratio": round(self.mean_latency_ratio, 6),
            "gained_case_ids": list(self.gained_case_ids),
            "lost_case_ids": list(self.lost_case_ids),
        }


@dataclass(frozen=True, slots=True)
class RetrievalAblationReport:
    baseline: str
    top_k: int
    dataset_digest: str
    suite_digest: str
    split: str
    configuration: tuple[tuple[str, _CONFIG_VALUE], ...]
    index_build_ms: float
    variants: tuple[RetrievalAblationVariant, ...]
    comparisons: tuple[RetrievalAblationComparison, ...]

    @property
    def configuration_digest(self) -> str:
        canonical = json.dumps(
            dict(self.configuration),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "baseline": self.baseline,
            "top_k": self.top_k,
            "dataset_digest": self.dataset_digest,
            "suite_digest": self.suite_digest,
            "split": self.split,
            "configuration_digest": self.configuration_digest,
            "configuration": dict(self.configuration),
            "index_build_ms": round(self.index_build_ms, 6),
            "variants": [variant.to_dict() for variant in self.variants],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        lines = [
            "# Retrieval ablation report",
            "",
            f"Dataset digest: `{self.dataset_digest}`  ",
            f"Suite digest: `{self.suite_digest or 'N/A'}`  ",
            f"Configuration digest: `{self.configuration_digest}`  ",
            f"Split: `{self.split or 'all'}`  ",
            f"Baseline: `{self.baseline}`  ",
            f"Index build: `{self.index_build_ms:.3f} ms`",
            "",
            "> 延迟来自同一进程内的轮转顺序重复运行，只用于当前机器上的相对比较，"
            "不能解释为生产 SLA。",
            "",
            "| Variant | Recall@K | MRR@K | nDCG@K | Route | Mean | P95 | Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for variant in self.variants:
            metrics = variant.run.report.metrics
            lines.append(
                f"| {variant.name} | {metrics.recall_at_k:.4f} | {metrics.mrr_at_k:.4f} | "
                f"{metrics.ndcg_at_k:.4f} | {metrics.route_accuracy:.4f} | "
                f"{variant.latency.mean_ms:.3f} ms | {variant.latency.p95_ms:.3f} ms | "
                f"{len(variant.failure_case_ids)} |"
            )

        lines.extend(
            [
                "",
                "## Delta against baseline",
                "",
                "| Variant | ΔRecall | ΔMRR | ΔnDCG | ΔRoute | Latency ratio | Gained | Lost |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in self.comparisons:
            lines.append(
                f"| {comparison.variant} | {comparison.recall_delta:+.4f} | "
                f"{comparison.mrr_delta:+.4f} | {comparison.ndcg_delta:+.4f} | "
                f"{comparison.route_delta:+.4f} | {comparison.mean_latency_ratio:.3f}x | "
                f"{len(comparison.gained_case_ids)} | {len(comparison.lost_case_ids)} |"
            )

        lines.extend(["", "## Case-level changes", ""])
        changed = False
        for comparison in self.comparisons:
            if not comparison.gained_case_ids and not comparison.lost_case_ids:
                continue
            changed = True
            gained = ", ".join(comparison.gained_case_ids) or "—"
            lost = ", ".join(comparison.lost_case_ids) or "—"
            lines.append(f"- `{comparison.variant}` gained: {gained}; lost: {lost}.")
        if not changed:
            lines.append("所有变体与基线通过/失败的样例集合一致。")
        return "\n".join(lines) + "\n"


def run_retrieval_ablation(
    cases: Sequence[RetrievalBenchmarkCase],
    retrievers: Mapping[str, Retriever],
    routing: RoutingPolicy,
    *,
    baseline: str,
    top_k: int = 5,
    repetitions: int = 3,
    suite_digest: str = "",
    split: str = "",
    configuration: Mapping[str, _CONFIG_VALUE] | None = None,
    variant_configurations: Mapping[str, Mapping[str, _CONFIG_VALUE]] | None = None,
    index_build_ms: float = 0.0,
) -> RetrievalAblationReport:
    """Compare multiple retrievers using rotated execution order and stable predictions."""

    if not cases:
        raise ValueError("cases cannot be empty")
    if len(retrievers) < 2:
        raise ValueError("at least two retrieval variants are required")
    names = tuple(retrievers)
    if any(not isinstance(name, str) or not _VARIANT_NAME.fullmatch(name) for name in names):
        raise ValueError("variant names must use lowercase letters, digits, and hyphens")
    if baseline not in retrievers:
        raise ValueError("baseline must name one retrieval variant")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("repetitions must be an integer")
    if not 1 <= repetitions <= 20:
        raise ValueError("repetitions must be between 1 and 20")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isfinite(index_build_ms) or index_build_ms < 0:
        raise ValueError("index_build_ms must be finite and non-negative")

    raw_variant_configurations = variant_configurations or {}
    unknown_configuration_names = set(raw_variant_configurations) - set(names)
    if unknown_configuration_names:
        raise ValueError("variant configuration names must match retrieval variants")

    first_runs: dict[str, RetrievalBenchmarkRun] = {}
    signatures: dict[str, tuple[tuple[object, ...], ...]] = {}
    latency_samples: dict[str, list[float]] = {name: [] for name in names}
    for repetition in range(repetitions):
        offset = repetition % len(names)
        ordered_names = (*names[offset:], *names[:offset])
        for name in ordered_names:
            run = run_retrieval_benchmark(
                cases,
                retrievers[name],
                routing,
                top_k=top_k,
            )
            signature = _prediction_signature(run)
            previous = signatures.setdefault(name, signature)
            if previous != signature:
                raise RetrievalNonDeterminismError(
                    f"retrieval variant {name!r} changed predictions between repetitions"
                )
            first_runs.setdefault(name, run)
            latency_samples[name].extend(item.latency_ms for item in run.predictions)

    variants = tuple(
        RetrievalAblationVariant(
            name=name,
            run=first_runs[name],
            latency=summarize_retrieval_latency(latency_samples[name]),
            repetitions=repetitions,
            configuration=_configuration_items(raw_variant_configurations.get(name, {})),
        )
        for name in names
    )
    baseline_variant = next(item for item in variants if item.name == baseline)
    comparisons = tuple(
        _compare(baseline_variant, variant)
        for variant in variants
        if variant.name != baseline
    )
    configuration_items = _configuration_items(configuration or {})
    return RetrievalAblationReport(
        baseline=baseline,
        top_k=top_k,
        dataset_digest=baseline_variant.run.report.dataset_digest,
        suite_digest=suite_digest,
        split=split,
        configuration=configuration_items,
        index_build_ms=index_build_ms,
        variants=variants,
        comparisons=comparisons,
    )


def _compare(
    baseline: RetrievalAblationVariant,
    candidate: RetrievalAblationVariant,
) -> RetrievalAblationComparison:
    baseline_metrics = baseline.run.report.metrics
    candidate_metrics = candidate.run.report.metrics
    baseline_passed = {item.case_id for item in baseline.run.predictions if item.passed}
    candidate_passed = {item.case_id for item in candidate.run.predictions if item.passed}
    baseline_latency = baseline.latency.mean_ms
    latency_ratio = candidate.latency.mean_ms / baseline_latency if baseline_latency else 1.0
    return RetrievalAblationComparison(
        variant=candidate.name,
        recall_delta=candidate_metrics.recall_at_k - baseline_metrics.recall_at_k,
        mrr_delta=candidate_metrics.mrr_at_k - baseline_metrics.mrr_at_k,
        ndcg_delta=candidate_metrics.ndcg_at_k - baseline_metrics.ndcg_at_k,
        route_delta=candidate_metrics.route_accuracy - baseline_metrics.route_accuracy,
        mean_latency_ratio=latency_ratio,
        gained_case_ids=tuple(sorted(candidate_passed - baseline_passed)),
        lost_case_ids=tuple(sorted(baseline_passed - candidate_passed)),
    )


def _prediction_signature(run: RetrievalBenchmarkRun) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.case_id,
            item.retrieved_sources,
            item.predicted_route.value,
            round(item.confidence, 12),
        )
        for item in run.predictions
    )


def _configuration_items(
    configuration: Mapping[str, _CONFIG_VALUE],
) -> tuple[tuple[str, _CONFIG_VALUE], ...]:
    items: list[tuple[str, _CONFIG_VALUE]] = []
    for key, value in configuration.items():
        if not isinstance(key, str) or not _VARIANT_NAME.fullmatch(key.replace("_", "-")):
            raise ValueError("configuration keys must be bounded identifiers")
        if any(term in key.casefold() for term in _SENSITIVE_CONFIGURATION_TERMS):
            raise ValueError("configuration keys cannot name sensitive values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("configuration values must be finite")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError("configuration values must be scalar")
        if isinstance(value, str) and len(value) > 512:
            raise ValueError("configuration string values must be bounded")
        items.append((key, value))
    return tuple(sorted(items))


__all__ = [
    "RetrievalAblationComparison",
    "RetrievalAblationReport",
    "RetrievalAblationVariant",
    "RetrievalNonDeterminismError",
    "run_retrieval_ablation",
]
