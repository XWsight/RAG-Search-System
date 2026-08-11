"""Deterministic, offline evaluation for retrieval, routing, and citations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.domain import Route
from rag_system.ranking import audit_citations


SCHEMA_VERSION = 1
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "question",
        "relevance",
        "retrieved_ids",
        "expected_route",
        "predicted_route",
        "allowed_citation_ids",
        "answer",
        "citation_required",
    }
)
_CITATION_ID = re.compile(r"^(?:L|W)[1-9]\d*$")
_CITATION_REFERENCE = re.compile(r"\[(?:L|W)\d+\]")
_CLAIM_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+")


class DatasetValidationError(ValueError):
    """Raised when an offline evaluation dataset violates its schema."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    relevance: tuple[tuple[str, int], ...]
    retrieved_ids: tuple[str, ...]
    expected_route: Route
    predicted_route: Route
    allowed_citation_ids: tuple[str, ...]
    answer: str
    citation_required: bool

    @property
    def relevance_map(self) -> dict[str, int]:
        return dict(self.relevance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "relevance": dict(self.relevance),
            "retrieved_ids": list(self.retrieved_ids),
            "expected_route": self.expected_route.value,
            "predicted_route": self.predicted_route.value,
            "allowed_citation_ids": list(self.allowed_citation_ids),
            "answer": self.answer,
            "citation_required": self.citation_required,
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    route_accuracy: float
    citation_validity: float
    citation_coverage: float

    def to_dict(self) -> dict[str, float]:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "route_accuracy": self.route_accuracy,
            "citation_validity": self.citation_validity,
            "citation_coverage": self.citation_coverage,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_digest: str
    top_k: int
    case_count: int
    retrieval_case_count: int
    citation_case_count: int
    metrics: EvaluationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_digest": self.dataset_digest,
            "top_k": self.top_k,
            "case_count": self.case_count,
            "retrieval_case_count": self.retrieval_case_count,
            "citation_case_count": self.citation_case_count,
            "metrics": self.metrics.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        metrics = self.metrics
        citation_validity = (
            f"{metrics.citation_validity:.4f}" if self.citation_case_count else "N/A"
        )
        citation_coverage = (
            f"{metrics.citation_coverage:.4f}" if self.citation_case_count else "N/A"
        )
        rows = (
            (f"Recall@{self.top_k}", f"{metrics.recall_at_k:.4f}"),
            (f"MRR@{self.top_k}", f"{metrics.mrr_at_k:.4f}"),
            (f"nDCG@{self.top_k}", f"{metrics.ndcg_at_k:.4f}"),
            ("路由准确率", f"{metrics.route_accuracy:.4f}"),
            ("引用有效率", citation_validity),
            ("引用覆盖率", citation_coverage),
        )
        lines = [
            "# 离线评测报告",
            "",
            f"- 数据集摘要：`{self.dataset_digest}`",
            f"- Top-k：`{self.top_k}`",
            f"- 样例数：`{self.case_count}`",
            f"- 检索样例数：`{self.retrieval_case_count}`",
            f"- 引用样例数：`{self.citation_case_count}`",
            "",
            "| 指标 | 结果 |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {name} | {value} |" for name, value in rows)
        return "\n".join(lines) + "\n"


def load_evaluation_dataset(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load and strictly validate a UTF-8 JSONL evaluation dataset."""

    dataset_path = Path(path)
    try:
        content = dataset_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DatasetValidationError(f"cannot read dataset {dataset_path}: {error}") from error

    if not content:
        raise DatasetValidationError("dataset cannot be empty")

    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            raise DatasetValidationError(f"line {line_number}: blank lines are not allowed")
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                f"line {line_number}: invalid JSON at column {error.colno}"
            ) from error
        case = evaluation_case_from_mapping(payload, location=f"line {line_number}")
        if case.case_id in seen_case_ids:
            raise DatasetValidationError(
                f"line {line_number}: duplicate case_id {case.case_id!r}"
            )
        seen_case_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise DatasetValidationError("dataset must contain at least one case")
    return tuple(cases)


def evaluation_case_from_mapping(
    payload: object,
    *,
    location: str = "case",
) -> EvaluationCase:
    """Validate one decoded JSON object and convert it to a frozen case."""

    if not isinstance(payload, Mapping):
        raise DatasetValidationError(f"{location}: expected a JSON object")
    keys = set(payload)
    missing = sorted(_CASE_FIELDS - keys)
    unknown = sorted(keys - _CASE_FIELDS)
    if missing:
        raise DatasetValidationError(f"{location}: missing fields: {', '.join(missing)}")
    if unknown:
        raise DatasetValidationError(f"{location}: unknown fields: {', '.join(unknown)}")

    case_id = _nonempty_string(payload["case_id"], "case_id", location)
    question = _nonempty_string(payload["question"], "question", location)
    relevance_value = payload["relevance"]
    if not isinstance(relevance_value, Mapping):
        raise DatasetValidationError(f"{location}: relevance must be an object")
    relevance: list[tuple[str, int]] = []
    for document_id, grade in relevance_value.items():
        resolved_id = _nonempty_string(document_id, "relevance key", location)
        if isinstance(grade, bool) or not isinstance(grade, int) or not 1 <= grade <= 3:
            raise DatasetValidationError(
                f"{location}: relevance grade for {resolved_id!r} must be an integer from 1 to 3"
            )
        relevance.append((resolved_id, grade))
    relevance.sort(key=lambda item: item[0])

    retrieved_ids = _unique_string_list(payload["retrieved_ids"], "retrieved_ids", location)
    allowed_citation_ids = _unique_string_list(
        payload["allowed_citation_ids"], "allowed_citation_ids", location
    )
    for citation_id in allowed_citation_ids:
        if not _CITATION_ID.fullmatch(citation_id):
            raise DatasetValidationError(
                f"{location}: invalid citation ID {citation_id!r}; expected L1 or W1 format"
            )

    expected_route = _route(payload["expected_route"], "expected_route", location)
    predicted_route = _route(payload["predicted_route"], "predicted_route", location)
    answer_value = payload["answer"]
    if not isinstance(answer_value, str):
        raise DatasetValidationError(f"{location}: answer must be a string")
    citation_required = payload["citation_required"]
    if not isinstance(citation_required, bool):
        raise DatasetValidationError(f"{location}: citation_required must be a boolean")
    if citation_required and not answer_value.strip():
        raise DatasetValidationError(
            f"{location}: answer cannot be empty when citations are required"
        )
    if citation_required and not allowed_citation_ids:
        raise DatasetValidationError(
            f"{location}: allowed_citation_ids cannot be empty when citations are required"
        )

    return EvaluationCase(
        case_id=case_id,
        question=question,
        relevance=tuple(relevance),
        retrieved_ids=retrieved_ids,
        expected_route=expected_route,
        predicted_route=predicted_route,
        allowed_citation_ids=allowed_citation_ids,
        answer=answer_value,
        citation_required=citation_required,
    )


def evaluate_cases(cases: Sequence[EvaluationCase], *, top_k: int = 5) -> EvaluationReport:
    """Compute macro retrieval metrics and micro route/citation metrics."""

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("cases cannot be empty")
    if any(not isinstance(case, EvaluationCase) for case in cases):
        raise TypeError("cases must contain only EvaluationCase objects")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")

    retrieval_cases = [case for case in cases if case.relevance]
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for case in retrieval_cases:
        relevance = case.relevance_map
        retrieved = case.retrieved_ids[:top_k]
        relevant_retrieved = sum(1 for document_id in retrieved if document_id in relevance)
        recalls.append(relevant_retrieved / len(relevance))

        first_relevant_rank = next(
            (rank for rank, document_id in enumerate(retrieved, start=1) if document_id in relevance),
            None,
        )
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)
        ndcgs.append(_ndcg(retrieved, relevance, top_k))

    correct_routes = sum(case.expected_route == case.predicted_route for case in cases)

    citation_cases = [case for case in cases if case.citation_required]
    citation_count = 0
    valid_citation_count = 0
    cited_sentence_count = 0
    claim_sentence_count = 0
    for case in citation_cases:
        audit = audit_citations(case.answer, case.allowed_citation_ids)
        citation_count += len(audit.cited_ids)
        valid_citation_count += len(audit.cited_ids) - len(audit.invalid_ids)
        case_cited_sentences, case_claim_sentences = _citation_sentence_counts(case.answer)
        cited_sentence_count += case_cited_sentences
        claim_sentence_count += case_claim_sentences

    metrics = EvaluationMetrics(
        recall_at_k=_rounded_mean(recalls),
        mrr_at_k=_rounded_mean(reciprocal_ranks),
        ndcg_at_k=_rounded_mean(ndcgs),
        route_accuracy=round(correct_routes / len(cases), 12),
        citation_validity=round(
            valid_citation_count / citation_count if citation_count else 1.0,
            12,
        ),
        citation_coverage=round(
            cited_sentence_count / claim_sentence_count if claim_sentence_count else 1.0,
            12,
        ),
    )
    return EvaluationReport(
        dataset_digest=_dataset_digest(cases),
        top_k=top_k,
        case_count=len(cases),
        retrieval_case_count=len(retrieval_cases),
        citation_case_count=len(citation_cases),
        metrics=metrics,
    )


def _ndcg(retrieved: Sequence[str], relevance: Mapping[str, int], top_k: int) -> float:
    actual = [relevance.get(document_id, 0) for document_id in retrieved[:top_k]]
    ideal = sorted(relevance.values(), reverse=True)[:top_k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def _dataset_digest(cases: Sequence[EvaluationCase]) -> str:
    canonical = json.dumps(
        [case.to_dict() for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _citation_sentence_counts(answer: str) -> tuple[int, int]:
    """Count cited factual sentences without splitting dots inside identifiers."""

    claim_count = 0
    cited_count = 0
    for sentence in _CLAIM_SENTENCE_SPLIT.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence or len(sentence) < 4:
            continue
        claim_count += 1
        if _CITATION_REFERENCE.search(sentence):
            cited_count += 1
    return cited_count, claim_count


def _rounded_mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 12) if values else 0.0


def _nonempty_string(value: object, field: str, location: str) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{location}: {field} must be a string")
    if not value.strip():
        raise DatasetValidationError(f"{location}: {field} cannot be empty")
    return value.strip()


def _unique_string_list(value: object, field: str, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{location}: {field} must be an array")
    resolved = tuple(_nonempty_string(item, field, location) for item in value)
    if len(resolved) != len(set(resolved)):
        raise DatasetValidationError(f"{location}: {field} cannot contain duplicates")
    return resolved


def _route(value: object, field: str, location: str) -> Route:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{location}: {field} must be a string")
    try:
        return Route(value)
    except ValueError as error:
        choices = ", ".join(route.value for route in Route)
        raise DatasetValidationError(
            f"{location}: {field} must be one of: {choices}"
        ) from error
