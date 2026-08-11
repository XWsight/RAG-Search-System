"""Cost-aware threshold calibration for local-answer routing."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfidenceSample:
    case_id: str
    confidence: float
    answerable: bool

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    threshold: float
    sample_count: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    weighted_error: float
    stability_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "sample_count": self.sample_count,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "weighted_error": self.weighted_error,
            "stability_margin": self.stability_margin,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        return (
            "# 路由阈值校准\n\n"
            f"- 推荐阈值：`{self.threshold:.6f}`\n"
            f"- 样例数：`{self.sample_count}`\n"
            f"- 加权错误：`{self.weighted_error:.4f}`\n"
            f"- Precision：`{self.precision:.4f}`\n"
            f"- Recall：`{self.recall:.4f}`\n"
            f"- F1：`{self.f1:.4f}`\n"
            f"- 最近样例距离：`{self.stability_margin:.6f}`\n"
            f"- FP / FN：`{self.false_positives}` / `{self.false_negatives}`\n\n"
            "应用到 `.env`：\n\n"
            f"```dotenv\nRAG_LOCAL_CONFIDENCE={self.threshold:.6f}\n```\n"
        )


def calibrate_threshold(
    samples: Sequence[ConfidenceSample],
    *,
    false_positive_cost: float = 2.0,
    false_negative_cost: float = 1.0,
) -> CalibrationReport:
    """Select a cost-aware threshold with a stable margin between samples."""

    if not samples:
        raise ValueError("samples cannot be empty")
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("error costs cannot be negative")
    if false_positive_cost == 0 and false_negative_cost == 0:
        raise ValueError("at least one error cost must be positive")
    identifiers = [sample.case_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("case_id values must be unique")

    confidence_levels = sorted({sample.confidence for sample in samples})
    candidates = set(confidence_levels)
    candidates.update(
        (lower + upper) / 2
        for lower, upper in zip(confidence_levels, confidence_levels[1:], strict=False)
    )
    if confidence_levels[-1] < 1.0:
        candidates.add((confidence_levels[-1] + 1.0) / 2)
    reports = [
        _score(
            samples,
            threshold,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        for threshold in sorted(candidates)
    ]
    return min(
        reports,
        key=lambda report: (
            report.weighted_error,
            -report.f1,
            -report.precision,
            -report.stability_margin,
            -report.threshold,
        ),
    )


def _score(
    samples: Sequence[ConfidenceSample],
    threshold: float,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> CalibrationReport:
    true_positives = true_negatives = false_positives = false_negatives = 0
    for sample in samples:
        predicted = sample.confidence >= threshold
        if predicted and sample.answerable:
            true_positives += 1
        elif predicted:
            false_positives += 1
        elif sample.answerable:
            false_negatives += 1
        else:
            true_negatives += 1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 1.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_error = (
        false_positive_cost * false_positives + false_negative_cost * false_negatives
    ) / len(samples)
    stability_margin = min(abs(sample.confidence - threshold) for sample in samples)
    return CalibrationReport(
        threshold=round(threshold, 12),
        sample_count=len(samples),
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 12),
        recall=round(recall, 12),
        f1=round(f1, 12),
        weighted_error=round(weighted_error, 12),
        stability_margin=round(stability_margin, 12),
    )


__all__ = ["CalibrationReport", "ConfidenceSample", "calibrate_threshold"]
