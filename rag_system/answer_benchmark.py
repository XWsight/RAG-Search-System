"""Ground-truth answer evaluation for structured generated claims."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.domain import AnswerClaim, GeneratedAnswer
from rag_system.grounding import validate_grounded_answer


SCHEMA_VERSION = 1
_CASE_FIELDS = frozenset({"case_id", "question", "evidence", "facts", "should_refuse"})
_FACT_FIELDS = frozenset({"fact_id", "term_groups", "supporting_citation_ids"})
_EVIDENCE_FIELDS = frozenset({"citation_id", "text"})
_CITATION_ID = re.compile(r"^(?:L|W)[1-9]\d*$")


class AnswerDatasetError(ValueError):
    """Raised when answer ground truth is ambiguous or malformed."""


@dataclass(frozen=True, slots=True)
class FactSpec:
    fact_id: str
    term_groups: tuple[tuple[str, ...], ...]
    supporting_citation_ids: tuple[str, ...]

    def matches(self, text: str) -> bool:
        normalized = text.casefold()
        return all(
            any(alternative.casefold() in normalized for alternative in group)
            for group in self.term_groups
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "term_groups": [list(group) for group in self.term_groups],
            "supporting_citation_ids": list(self.supporting_citation_ids),
        }


@dataclass(frozen=True, slots=True)
class AnswerBenchmarkCase:
    case_id: str
    question: str
    evidence: tuple[tuple[str, str], ...]
    facts: tuple[FactSpec, ...]
    should_refuse: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "evidence": [
                {"citation_id": citation_id, "text": text}
                for citation_id, text in self.evidence
            ],
            "facts": [fact.to_dict() for fact in self.facts],
            "should_refuse": self.should_refuse,
        }


@dataclass(frozen=True, slots=True)
class AnswerCaseResult:
    case_id: str
    contract_valid: bool
    refusal_correct: bool
    expected_fact_count: int
    recovered_fact_ids: tuple[str, ...]
    claim_count: int
    atomic_claim_count: int
    grounded_claim_count: int
    claims: tuple[AnswerClaim, ...]
    error_code: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.contract_valid
            and self.refusal_correct
            and len(self.recovered_fact_ids) == self.expected_fact_count
            and self.atomic_claim_count == self.claim_count
            and self.grounded_claim_count == self.claim_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "contract_valid": self.contract_valid,
            "refusal_correct": self.refusal_correct,
            "expected_fact_count": self.expected_fact_count,
            "recovered_fact_ids": list(self.recovered_fact_ids),
            "claim_count": self.claim_count,
            "atomic_claim_count": self.atomic_claim_count,
            "grounded_claim_count": self.grounded_claim_count,
            "claims": [
                {"text": claim.text, "citation_ids": list(claim.citation_ids)}
                for claim in self.claims
            ],
            "error_code": self.error_code,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class AnswerBenchmarkMetrics:
    contract_success_rate: float
    refusal_accuracy: float
    fact_recall: float
    atomic_claim_rate: float
    attribution_precision: float

    def to_dict(self) -> dict[str, float]:
        return {
            "contract_success_rate": self.contract_success_rate,
            "refusal_accuracy": self.refusal_accuracy,
            "fact_recall": self.fact_recall,
            "atomic_claim_rate": self.atomic_claim_rate,
            "attribution_precision": self.attribution_precision,
        }


@dataclass(frozen=True, slots=True)
class AnswerBenchmarkReport:
    dataset_digest: str
    case_count: int
    fact_count: int
    metrics: AnswerBenchmarkMetrics
    results: tuple[AnswerCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_digest": self.dataset_digest,
            "case_count": self.case_count,
            "fact_count": self.fact_count,
            "metrics": self.metrics.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        rows = (
            ("结构契约成功率", self.metrics.contract_success_rate),
            ("拒答准确率", self.metrics.refusal_accuracy),
            ("事实召回率", self.metrics.fact_recall),
            ("原子结论率", self.metrics.atomic_claim_rate),
            ("归因精确率", self.metrics.attribution_precision),
        )
        lines = [
            "# 结构化回答评测",
            "",
            f"- 数据集摘要：`{self.dataset_digest}`",
            f"- 样例数：`{self.case_count}`",
            f"- 标注事实数：`{self.fact_count}`",
            "",
            "| 指标 | 结果 |",
            "| --- | ---: |",
            *(f"| {name} | {value:.4f} |" for name, value in rows),
            "",
            "## 失败样例",
            "",
        ]
        failures = [result for result in self.results if not result.passed]
        if not failures:
            lines.append("无。")
        else:
            lines.extend(
                f"- `{item.case_id}`：error=`{item.error_code or 'none'}`，"
                f"facts={len(item.recovered_fact_ids)}/{item.expected_fact_count}，"
                f"claims={item.grounded_claim_count}/{item.claim_count}"
                for item in failures
            )
        return "\n".join(lines) + "\n"


AnswerGenerator = Callable[[str, Sequence[tuple[str, str]]], GeneratedAnswer]


def load_answer_benchmark(path: str | Path) -> tuple[AnswerBenchmarkCase, ...]:
    dataset_path = Path(path)
    try:
        content = dataset_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AnswerDatasetError(f"cannot read dataset {dataset_path}: {error}") from error
    if not content:
        raise AnswerDatasetError("dataset cannot be empty")

    cases: list[AnswerBenchmarkCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise AnswerDatasetError(f"line {line_number}: blank lines are not allowed")
        try:
            payload = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError):
            raise AnswerDatasetError(f"line {line_number}: invalid JSON") from None
        case = answer_case_from_mapping(payload, location=f"line {line_number}")
        if case.case_id in seen:
            raise AnswerDatasetError(f"line {line_number}: duplicate case_id")
        seen.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def answer_case_from_mapping(
    payload: object,
    *,
    location: str = "case",
) -> AnswerBenchmarkCase:
    mapping = _exact_mapping(payload, _CASE_FIELDS, location)
    case_id = _bounded_string(mapping["case_id"], "case_id", location, 128)
    question = _bounded_string(mapping["question"], "question", location, 10_000)
    should_refuse = mapping["should_refuse"]
    if not isinstance(should_refuse, bool):
        raise AnswerDatasetError(f"{location}: should_refuse must be a boolean")

    evidence_value = mapping["evidence"]
    if not isinstance(evidence_value, list) or not 1 <= len(evidence_value) <= 24:
        raise AnswerDatasetError(f"{location}: evidence must contain 1 to 24 items")
    evidence: list[tuple[str, str]] = []
    for index, item in enumerate(evidence_value, start=1):
        item_mapping = _exact_mapping(item, _EVIDENCE_FIELDS, f"{location}.evidence[{index}]")
        citation_id = _bounded_string(
            item_mapping["citation_id"], "citation_id", location, 16
        )
        if _CITATION_ID.fullmatch(citation_id) is None:
            raise AnswerDatasetError(f"{location}: evidence citation ID is invalid")
        text = _bounded_string(item_mapping["text"], "text", location, 20_000)
        evidence.append((citation_id, text))
    evidence_ids = tuple(item[0] for item in evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise AnswerDatasetError(f"{location}: evidence citation IDs must be unique")

    facts_value = mapping["facts"]
    if not isinstance(facts_value, list) or len(facts_value) > 24:
        raise AnswerDatasetError(f"{location}: facts must be an array with at most 24 items")
    facts = tuple(
        _fact_from_mapping(item, evidence_ids, f"{location}.facts[{index}]")
        for index, item in enumerate(facts_value, start=1)
    )
    fact_ids = tuple(fact.fact_id for fact in facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise AnswerDatasetError(f"{location}: fact IDs must be unique")
    if should_refuse == bool(facts):
        raise AnswerDatasetError(
            f"{location}: refusal cases require zero facts; answerable cases require facts"
        )
    return AnswerBenchmarkCase(case_id, question, tuple(evidence), facts, should_refuse)


def run_answer_benchmark(
    cases: Sequence[AnswerBenchmarkCase],
    generator: AnswerGenerator,
) -> AnswerBenchmarkReport:
    if not cases:
        raise ValueError("cases cannot be empty")
    results: list[AnswerCaseResult] = []
    total_facts = sum(len(case.facts) for case in cases)
    total_claims = 0
    total_atomic = 0
    total_grounded = 0

    for case in cases:
        try:
            draft = generator(case.question, case.evidence)
            validate_grounded_answer(draft, tuple(item[0] for item in case.evidence))
        except Exception as error:  # The report records only the safe exception type.
            error_code = getattr(error, "code", type(error).__name__)
            if not isinstance(error_code, str) or not error_code.isascii() or len(error_code) > 64:
                error_code = type(error).__name__
            results.append(
                AnswerCaseResult(
                    case_id=case.case_id,
                    contract_valid=False,
                    refusal_correct=False,
                    expected_fact_count=len(case.facts),
                    recovered_fact_ids=(),
                    claim_count=0,
                    atomic_claim_count=0,
                    grounded_claim_count=0,
                    claims=(),
                    error_code=error_code,
                )
            )
            continue

        recovered: list[str] = []
        atomic_count = 0
        grounded_count = 0
        for claim in draft.claims:
            matched = tuple(fact for fact in case.facts if fact.matches(claim.text))
            if len(matched) != 1:
                continue
            atomic_count += 1
            fact = matched[0]
            cited = set(claim.citation_ids)
            supporting = set(fact.supporting_citation_ids)
            if cited and cited <= supporting:
                grounded_count += 1
                if fact.fact_id not in recovered:
                    recovered.append(fact.fact_id)

        claim_count = len(draft.claims)
        total_claims += claim_count
        total_atomic += atomic_count
        total_grounded += grounded_count
        results.append(
            AnswerCaseResult(
                case_id=case.case_id,
                contract_valid=True,
                refusal_correct=draft.insufficient == case.should_refuse,
                expected_fact_count=len(case.facts),
                recovered_fact_ids=tuple(recovered),
                claim_count=claim_count,
                atomic_claim_count=atomic_count,
                grounded_claim_count=grounded_count,
                claims=draft.claims,
            )
        )

    metrics = AnswerBenchmarkMetrics(
        contract_success_rate=_ratio(sum(item.contract_valid for item in results), len(cases)),
        refusal_accuracy=_ratio(sum(item.refusal_correct for item in results), len(cases)),
        fact_recall=_ratio(sum(len(item.recovered_fact_ids) for item in results), total_facts),
        atomic_claim_rate=_ratio(
            total_atomic,
            total_claims,
            empty_value=0.0 if total_facts else 1.0,
        ),
        attribution_precision=_ratio(
            total_grounded,
            total_claims,
            empty_value=0.0 if total_facts else 1.0,
        ),
    )
    return AnswerBenchmarkReport(
        dataset_digest=_dataset_digest(cases),
        case_count=len(cases),
        fact_count=total_facts,
        metrics=metrics,
        results=tuple(results),
    )


def _fact_from_mapping(
    payload: object,
    evidence_ids: Sequence[str],
    location: str,
) -> FactSpec:
    mapping = _exact_mapping(payload, _FACT_FIELDS, location)
    fact_id = _bounded_string(mapping["fact_id"], "fact_id", location, 128)
    groups_value = mapping["term_groups"]
    if not isinstance(groups_value, list) or not 1 <= len(groups_value) <= 12:
        raise AnswerDatasetError(f"{location}: term_groups must contain 1 to 12 groups")
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(groups_value, start=1):
        if not isinstance(group, list) or not 1 <= len(group) <= 8:
            raise AnswerDatasetError(f"{location}.term_groups[{index}] is invalid")
        alternatives = tuple(
            _bounded_string(item, "term alternative", location, 100) for item in group
        )
        if len(alternatives) != len(set(item.casefold() for item in alternatives)):
            raise AnswerDatasetError(f"{location}: duplicate term alternatives")
        groups.append(alternatives)

    supporting_value = mapping["supporting_citation_ids"]
    if not isinstance(supporting_value, list) or not supporting_value:
        raise AnswerDatasetError(f"{location}: supporting citations cannot be empty")
    supporting = tuple(
        _bounded_string(item, "supporting citation", location, 16)
        for item in supporting_value
    )
    if len(supporting) != len(set(supporting)) or not set(supporting) <= set(evidence_ids):
        raise AnswerDatasetError(f"{location}: supporting citations are invalid")
    return FactSpec(fact_id, tuple(groups), supporting)


def _exact_mapping(payload: object, fields: frozenset[str], location: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise AnswerDatasetError(f"{location}: expected an object")
    missing = fields - set(payload)
    unknown = set(payload) - fields
    if missing or unknown:
        raise AnswerDatasetError(f"{location}: fields do not match the schema")
    return payload


def _bounded_string(value: object, field: str, location: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise AnswerDatasetError(f"{location}: {field} is empty or too long")
    return value.strip()


def _dataset_digest(cases: Sequence[AnswerBenchmarkCase]) -> str:
    canonical = json.dumps(
        [case.to_dict() for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _ratio(numerator: int, denominator: int, *, empty_value: float = 1.0) -> float:
    return round(numerator / denominator, 12) if denominator else empty_value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in pairs:
        if key in resolved:
            raise ValueError("duplicate JSON key")
        resolved[key] = value
    return resolved
