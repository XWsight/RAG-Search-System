"""Governed answer-evaluation suites with frozen evidence and coverage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.answer_benchmark import (
    AnswerBenchmarkCase,
    AnswerDatasetError,
    answer_case_from_mapping,
)
from rag_system.evaluation_suite import (
    EvaluationSuiteError,
    bounded_text,
    canonical_bundle_digest,
    enum_value,
    exact_fields,
    identifier,
    normalized_text_fingerprint,
    positive_int,
    read_json_object,
    validate_frozen_contract,
)


_ROOT_FIELDS = frozenset(
    {"schema_version", "suite_id", "description", "requirements", "cases"}
)
_REQUIREMENT_FIELDS = frozenset(
    {
        "minimum_cases",
        "minimum_categories",
        "minimum_facts",
        "minimum_answerable_cases",
        "minimum_refusal_cases",
        "minimum_risk_tags",
        "minimum_cases_per_split",
        "minimum_cases_per_difficulty",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "category",
        "split",
        "difficulty",
        "risk_tags",
        "question",
        "evidence",
        "facts",
        "should_refuse",
    }
)
_GROUND_TRUTH_FIELDS = frozenset(
    {"case_id", "question", "evidence", "facts", "should_refuse"}
)
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "suite_digest",
        "case_count",
        "category_count",
        "fact_count",
        "answerable_case_count",
        "refusal_case_count",
        "risk_tag_count",
        "cases_by_split",
        "cases_by_difficulty",
        "cases_by_category",
    }
)
_SPLITS = ("development", "validation", "test")
_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True, slots=True)
class AnswerSuiteRequirements:
    minimum_cases: int
    minimum_categories: int
    minimum_facts: int
    minimum_answerable_cases: int
    minimum_refusal_cases: int
    minimum_risk_tags: int
    minimum_cases_per_split: tuple[tuple[str, int], ...]
    minimum_cases_per_difficulty: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class AnswerSuiteCase:
    category: str
    split: str
    difficulty: str
    risk_tags: tuple[str, ...]
    benchmark_case: AnswerBenchmarkCase


@dataclass(frozen=True, slots=True)
class AnswerBenchmarkSuite:
    suite_id: str
    description: str
    bundle_digest: str
    manifest_path: Path
    requirements: AnswerSuiteRequirements
    cases: tuple[AnswerSuiteCase, ...]

    @property
    def benchmark_cases(self) -> tuple[AnswerBenchmarkCase, ...]:
        return tuple(item.benchmark_case for item in self.cases)

    def cases_for_split(self, split: str) -> tuple[AnswerBenchmarkCase, ...]:
        resolved = enum_value(split, _SPLITS, location="split")
        return tuple(
            item.benchmark_case for item in self.cases if item.split == resolved
        )

    def summary(self) -> dict[str, Any]:
        benchmark_cases = self.benchmark_cases
        return {
            "schema_version": 1,
            "suite_id": self.suite_id,
            "suite_digest": self.bundle_digest,
            "case_count": len(self.cases),
            "category_count": len({item.category for item in self.cases}),
            "fact_count": sum(len(item.facts) for item in benchmark_cases),
            "answerable_case_count": sum(not item.should_refuse for item in benchmark_cases),
            "refusal_case_count": sum(item.should_refuse for item in benchmark_cases),
            "risk_tag_count": len({tag for item in self.cases for tag in item.risk_tags}),
            "cases_by_split": _counts(self.cases, "split"),
            "cases_by_difficulty": _counts(self.cases, "difficulty"),
            "cases_by_category": _counts(self.cases, "category"),
        }

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = [
            f"# Answer benchmark suite: {self.suite_id}",
            "",
            self.description,
            "",
            f"Bundle digest: `{self.bundle_digest}`",
            "",
            "| Cases | Facts | Answerable | Refusal | Categories | Risk tags |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {summary['case_count']} | {summary['fact_count']} | "
            f"{summary['answerable_case_count']} | {summary['refusal_case_count']} | "
            f"{summary['category_count']} | {summary['risk_tag_count']} |",
            "",
            "## Coverage matrix",
            "",
            "| Dimension | Value | Cases |",
            "| --- | --- | ---: |",
        ]
        for dimension, key in (
            ("split", "cases_by_split"),
            ("difficulty", "cases_by_difficulty"),
            ("category", "cases_by_category"),
        ):
            for value, count in summary[key].items():
                lines.append(f"| {dimension} | {value} | {count} |")
        return "\n".join(lines) + "\n"


def load_answer_suite(path: str | Path) -> AnswerBenchmarkSuite:
    manifest_path, payload = read_json_object(path, label="answer suite manifest")
    exact_fields(payload, _ROOT_FIELDS, location="answer suite")
    if payload["schema_version"] != 1:
        raise EvaluationSuiteError("answer suite schema_version must be 1")
    suite_id = identifier(payload["suite_id"], location="answer suite.suite_id")
    description = bounded_text(
        payload["description"],
        location="answer suite.description",
        minimum=20,
        maximum=500,
    )
    requirements = _requirements(payload["requirements"])
    cases_value = payload["cases"]
    if not isinstance(cases_value, list) or not cases_value:
        raise EvaluationSuiteError("answer suite.cases must be a non-empty array")

    cases: list[AnswerSuiteCase] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    seen_evidence: dict[str, tuple[str, str]] = {}
    for index, value in enumerate(cases_value, start=1):
        item = _case(value, location=f"answer suite.cases[{index}]")
        case_id = item.benchmark_case.case_id
        if case_id in seen_ids:
            raise EvaluationSuiteError(f"duplicate answer case_id {case_id!r}")
        seen_ids.add(case_id)
        fingerprint = normalized_text_fingerprint(item.benchmark_case.question)
        previous = seen_questions.get(fingerprint)
        if previous is not None:
            raise EvaluationSuiteError(
                f"duplicate normalized answer question in {previous!r} and {case_id!r}"
            )
        seen_questions[fingerprint] = case_id
        for citation_id, text in item.benchmark_case.evidence:
            evidence_fingerprint = normalized_text_fingerprint(text)
            previous_evidence = seen_evidence.get(evidence_fingerprint)
            if previous_evidence is not None:
                previous_case, previous_citation = previous_evidence
                raise EvaluationSuiteError(
                    "duplicate normalized answer evidence in "
                    f"{previous_case!r}/{previous_citation!r} and "
                    f"{case_id!r}/{citation_id!r}"
                )
            seen_evidence[evidence_fingerprint] = (case_id, citation_id)
        _validate_reference_claims(item.benchmark_case)
        cases.append(item)

    _validate_requirements(cases, requirements)
    return AnswerBenchmarkSuite(
        suite_id=suite_id,
        description=description,
        bundle_digest=canonical_bundle_digest(payload),
        manifest_path=manifest_path,
        requirements=requirements,
        cases=tuple(cases),
    )


def validate_answer_suite_contract(
    suite: AnswerBenchmarkSuite, path: str | Path
) -> None:
    validate_frozen_contract(
        suite.summary(),
        path,
        fields=_CONTRACT_FIELDS,
        label="answer suite contract",
    )


def _requirements(value: object) -> AnswerSuiteRequirements:
    location = "answer suite.requirements"
    if not isinstance(value, Mapping):
        raise EvaluationSuiteError(f"{location} must be an object")
    exact_fields(value, _REQUIREMENT_FIELDS, location=location)
    return AnswerSuiteRequirements(
        minimum_cases=positive_int(value["minimum_cases"], location=f"{location}.minimum_cases"),
        minimum_categories=positive_int(
            value["minimum_categories"], location=f"{location}.minimum_categories"
        ),
        minimum_facts=positive_int(value["minimum_facts"], location=f"{location}.minimum_facts"),
        minimum_answerable_cases=positive_int(
            value["minimum_answerable_cases"],
            location=f"{location}.minimum_answerable_cases",
        ),
        minimum_refusal_cases=positive_int(
            value["minimum_refusal_cases"], location=f"{location}.minimum_refusal_cases"
        ),
        minimum_risk_tags=positive_int(
            value["minimum_risk_tags"], location=f"{location}.minimum_risk_tags"
        ),
        minimum_cases_per_split=_count_requirements(
            value["minimum_cases_per_split"], _SPLITS, f"{location}.minimum_cases_per_split"
        ),
        minimum_cases_per_difficulty=_count_requirements(
            value["minimum_cases_per_difficulty"],
            _DIFFICULTIES,
            f"{location}.minimum_cases_per_difficulty",
        ),
    )


def _case(value: object, *, location: str) -> AnswerSuiteCase:
    if not isinstance(value, Mapping):
        raise EvaluationSuiteError(f"{location} must be an object")
    exact_fields(value, _CASE_FIELDS, location=location)
    category = identifier(value["category"], location=f"{location}.category")
    split = enum_value(value["split"], _SPLITS, location=f"{location}.split")
    difficulty = enum_value(
        value["difficulty"], _DIFFICULTIES, location=f"{location}.difficulty"
    )
    risk_tags_value = value["risk_tags"]
    if not isinstance(risk_tags_value, list) or not 1 <= len(risk_tags_value) <= 8:
        raise EvaluationSuiteError(f"{location}.risk_tags must contain 1 to 8 items")
    risk_tags = tuple(
        identifier(tag, location=f"{location}.risk_tags[{index}]")
        for index, tag in enumerate(risk_tags_value, start=1)
    )
    if len(set(risk_tags)) != len(risk_tags):
        raise EvaluationSuiteError(f"{location}.risk_tags must be unique")
    ground_truth = {field: value[field] for field in _GROUND_TRUTH_FIELDS}
    try:
        benchmark_case = answer_case_from_mapping(ground_truth, location=location)
    except AnswerDatasetError as error:
        raise EvaluationSuiteError(str(error)) from error
    return AnswerSuiteCase(category, split, difficulty, risk_tags, benchmark_case)


def _validate_reference_claims(case: AnswerBenchmarkCase) -> None:
    for fact in case.facts:
        reference = " ".join(group[0] for group in fact.term_groups)
        matches = tuple(candidate for candidate in case.facts if candidate.matches(reference))
        if matches != (fact,):
            raise EvaluationSuiteError(
                f"case {case.case_id!r} has ambiguous or unsatisfiable fact {fact.fact_id!r}"
            )


def _validate_requirements(
    cases: Sequence[AnswerSuiteCase], requirements: AnswerSuiteRequirements
) -> None:
    benchmark_cases = tuple(item.benchmark_case for item in cases)
    facts = sum(len(item.facts) for item in benchmark_cases)
    answerable = sum(not item.should_refuse for item in benchmark_cases)
    refusals = sum(item.should_refuse for item in benchmark_cases)
    categories = len({item.category for item in cases})
    risk_tags = len({tag for item in cases for tag in item.risk_tags})
    checks = (
        ("cases", len(cases), requirements.minimum_cases),
        ("categories", categories, requirements.minimum_categories),
        ("facts", facts, requirements.minimum_facts),
        ("answerable cases", answerable, requirements.minimum_answerable_cases),
        ("refusal cases", refusals, requirements.minimum_refusal_cases),
        ("risk tags", risk_tags, requirements.minimum_risk_tags),
    )
    failures = [f"{name}={actual}<{minimum}" for name, actual, minimum in checks if actual < minimum]
    if failures:
        raise EvaluationSuiteError(
            "answer suite does not meet minimum coverage: " + ", ".join(failures)
        )
    _enforce_counts(
        _counts(cases, "split"), requirements.minimum_cases_per_split, "split"
    )
    _enforce_counts(
        _counts(cases, "difficulty"),
        requirements.minimum_cases_per_difficulty,
        "difficulty",
    )


def _counts(cases: Sequence[AnswerSuiteCase], attribute: str) -> dict[str, int]:
    return dict(sorted(Counter(getattr(item, attribute) for item in cases).items()))


def _count_requirements(
    value: object, allowed: tuple[str, ...], location: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise EvaluationSuiteError(f"{location} must be an object")
    exact_fields(value, frozenset(allowed), location=location)
    return tuple(
        (name, positive_int(value[name], location=f"{location}.{name}")) for name in allowed
    )


def _enforce_counts(
    actual: Mapping[str, int], required: Sequence[tuple[str, int]], dimension: str
) -> None:
    failures = [
        f"{name}={actual.get(name, 0)}<{minimum}"
        for name, minimum in required
        if actual.get(name, 0) < minimum
    ]
    if failures:
        raise EvaluationSuiteError(
            f"answer suite does not meet minimum cases per {dimension}: "
            + ", ".join(failures)
        )


__all__ = [
    "AnswerBenchmarkSuite",
    "AnswerSuiteCase",
    "AnswerSuiteRequirements",
    "load_answer_suite",
    "validate_answer_suite_contract",
]
