"""Strict, leakage-aware retrieval benchmark suite manifests."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.benchmark import RetrievalBenchmarkCase
from rag_system.domain import Route
from rag_system.evaluation import DatasetValidationError
from rag_system.json_contract import JsonContractError, decode_json_object


_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "suite_id",
        "description",
        "corpus_root",
        "requirements",
        "families",
    }
)
_REQUIREMENT_FIELDS = frozenset(
    {
        "minimum_cases",
        "minimum_families",
        "minimum_categories",
        "minimum_cases_per_split",
        "minimum_cases_per_route",
        "minimum_cases_per_difficulty",
    }
)
_FAMILY_FIELDS = frozenset(
    {
        "family_id",
        "category",
        "split",
        "difficulty",
        "expected_route",
        "allow_web",
        "relevance",
        "questions",
    }
)
_SPLITS = ("development", "validation", "test")
_DIFFICULTIES = ("easy", "medium", "hard")
_ROUTES = (Route.LOCAL.value, Route.REFUSED.value, Route.WEB.value)
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "suite_digest",
        "case_count",
        "family_count",
        "source_count",
        "category_count",
        "cases_by_split",
        "cases_by_route",
        "cases_by_difficulty",
    }
)


@dataclass(frozen=True, slots=True)
class SuiteRequirements:
    minimum_cases: int
    minimum_families: int
    minimum_categories: int
    minimum_cases_per_split: tuple[tuple[str, int], ...]
    minimum_cases_per_route: tuple[tuple[str, int], ...]
    minimum_cases_per_difficulty: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RetrievalCaseFamily:
    family_id: str
    category: str
    split: str
    difficulty: str
    expected_route: Route
    allow_web: bool
    relevance: tuple[tuple[str, int], ...]
    questions: tuple[str, ...]

    def expand(self) -> tuple[RetrievalBenchmarkCase, ...]:
        return tuple(
            RetrievalBenchmarkCase(
                case_id=f"{self.family_id}__{index:02d}",
                question=question,
                relevance=self.relevance,
                expected_route=self.expected_route,
                allow_web=self.allow_web,
            )
            for index, question in enumerate(self.questions, start=1)
        )


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkSuite:
    suite_id: str
    description: str
    bundle_digest: str
    manifest_path: Path
    corpus_root: Path
    requirements: SuiteRequirements
    families: tuple[RetrievalCaseFamily, ...]
    cases: tuple[RetrievalBenchmarkCase, ...]
    documents: tuple[Path, ...]

    def cases_for_split(self, split: str) -> tuple[RetrievalBenchmarkCase, ...]:
        _enum(split, _SPLITS, "split")
        return tuple(
            case
            for family in self.families
            if family.split == split
            for case in family.expand()
        )

    def documents_for_split(self, split: str) -> tuple[Path, ...]:
        _enum(split, _SPLITS, "split")
        sources = {
            source
            for family in self.families
            if family.split == split
            for source, _grade in family.relevance
        }
        return tuple((self.corpus_root / source).resolve(strict=True) for source in sorted(sources))

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite_id": self.suite_id,
            "suite_digest": self.bundle_digest,
            "case_count": len(self.cases),
            "family_count": len(self.families),
            "source_count": len(self.documents),
            "category_count": len({family.category for family in self.families}),
            "cases_by_split": _case_counts(self.families, "split"),
            "cases_by_route": _case_counts(self.families, "expected_route"),
            "cases_by_difficulty": _case_counts(self.families, "difficulty"),
        }

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = [
            f"# Retrieval benchmark suite: {self.suite_id}",
            "",
            self.description,
            "",
            f"Bundle digest: `{self.bundle_digest}`",
            "",
            "| Cases | Semantic families | Sources | Categories |",
            "| ---: | ---: | ---: | ---: |",
            f"| {summary['case_count']} | {summary['family_count']} | "
            f"{summary['source_count']} | {summary['category_count']} |",
            "",
            "## Coverage matrix",
            "",
            "| Dimension | Value | Cases |",
            "| --- | --- | ---: |",
        ]
        for dimension, key in (
            ("split", "cases_by_split"),
            ("route", "cases_by_route"),
            ("difficulty", "cases_by_difficulty"),
        ):
            counts = summary[key]
            for value, count in counts.items():
                lines.append(f"| {dimension} | {value} | {count} |")
        return "\n".join(lines) + "\n"


def load_retrieval_suite(path: str | Path) -> RetrievalBenchmarkSuite:
    manifest_path = Path(path).resolve(strict=True)
    try:
        content = manifest_path.read_text(encoding="utf-8")
        payload = decode_json_object(content)
    except (OSError, UnicodeError, JsonContractError) as error:
        raise DatasetValidationError(f"cannot read suite manifest: {error}") from error

    _exact_fields(payload, _ROOT_FIELDS, "suite")
    if payload["schema_version"] != 1:
        raise DatasetValidationError("suite: schema_version must be 1")
    suite_id = _identifier(payload["suite_id"], "suite.suite_id")
    description = _text(payload["description"], "suite.description", minimum=20, maximum=500)
    corpus_root = _resolve_corpus_root(manifest_path, payload["corpus_root"])
    requirements = _requirements(payload["requirements"])
    families_value = payload["families"]
    if not isinstance(families_value, list) or not families_value:
        raise DatasetValidationError("suite.families must be a non-empty array")

    families: list[RetrievalCaseFamily] = []
    seen_family_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    source_paths: dict[str, Path] = {}
    for index, value in enumerate(families_value, start=1):
        family = _family(value, location=f"suite.families[{index}]", corpus_root=corpus_root)
        if family.family_id in seen_family_ids:
            raise DatasetValidationError(f"duplicate family_id {family.family_id!r}")
        seen_family_ids.add(family.family_id)
        for question in family.questions:
            fingerprint = _question_fingerprint(question)
            previous = seen_questions.get(fingerprint)
            if previous is not None:
                raise DatasetValidationError(
                    f"duplicate normalized question in {previous!r} and {family.family_id!r}"
                )
            seen_questions[fingerprint] = family.family_id
        for source, _grade in family.relevance:
            source_splits[source].add(family.split)
            source_paths[source] = _resolve_source(corpus_root, source)
        families.append(family)

    leaking_sources = sorted(
        source for source, splits in source_splits.items() if len(splits) > 1
    )
    if leaking_sources:
        raise DatasetValidationError(
            "source documents cannot cross dataset splits: " + ", ".join(leaking_sources)
        )

    cases = tuple(case for family in families for case in family.expand())
    _validate_requirements(families, cases, requirements)
    documents = tuple(source_paths[source] for source in sorted(source_paths))
    return RetrievalBenchmarkSuite(
        suite_id=suite_id,
        description=description,
        bundle_digest=_bundle_digest(payload, documents, corpus_root),
        manifest_path=manifest_path,
        corpus_root=corpus_root,
        requirements=requirements,
        families=tuple(families),
        cases=cases,
        documents=documents,
    )


def validate_suite_contract(suite: RetrievalBenchmarkSuite, path: str | Path) -> None:
    contract_path = Path(path)
    try:
        payload = decode_json_object(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JsonContractError) as error:
        raise DatasetValidationError(f"cannot read suite contract: {error}") from error
    _exact_fields(payload, _CONTRACT_FIELDS, "suite contract")
    if payload["schema_version"] != 1:
        raise DatasetValidationError("suite contract schema_version must be 1")
    summary = suite.summary()
    mismatches = [
        field
        for field in sorted(_CONTRACT_FIELDS - {"schema_version"})
        if payload[field] != summary[field]
    ]
    if mismatches:
        raise DatasetValidationError(
            "suite contract mismatch: " + ", ".join(mismatches)
        )


def _requirements(value: object) -> SuiteRequirements:
    location = "suite.requirements"
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{location} must be an object")
    _exact_fields(value, _REQUIREMENT_FIELDS, location)
    return SuiteRequirements(
        minimum_cases=_positive_int(value["minimum_cases"], f"{location}.minimum_cases"),
        minimum_families=_positive_int(
            value["minimum_families"], f"{location}.minimum_families"
        ),
        minimum_categories=_positive_int(
            value["minimum_categories"], f"{location}.minimum_categories"
        ),
        minimum_cases_per_split=_count_requirements(
            value["minimum_cases_per_split"], _SPLITS, f"{location}.minimum_cases_per_split"
        ),
        minimum_cases_per_route=_count_requirements(
            value["minimum_cases_per_route"], _ROUTES, f"{location}.minimum_cases_per_route"
        ),
        minimum_cases_per_difficulty=_count_requirements(
            value["minimum_cases_per_difficulty"],
            _DIFFICULTIES,
            f"{location}.minimum_cases_per_difficulty",
        ),
    )


def _family(
    value: object,
    *,
    location: str,
    corpus_root: Path,
) -> RetrievalCaseFamily:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{location} must be an object")
    _exact_fields(value, _FAMILY_FIELDS, location)
    family_id = _identifier(value["family_id"], f"{location}.family_id")
    category = _identifier(value["category"], f"{location}.category")
    split = _enum(value["split"], _SPLITS, f"{location}.split")
    difficulty = _enum(value["difficulty"], _DIFFICULTIES, f"{location}.difficulty")
    try:
        expected_route = Route(value["expected_route"])
    except (TypeError, ValueError):
        raise DatasetValidationError(f"{location}.expected_route is invalid") from None
    allow_web = value["allow_web"]
    if not isinstance(allow_web, bool):
        raise DatasetValidationError(f"{location}.allow_web must be a boolean")
    relevance = _relevance(value["relevance"], location, corpus_root)
    if expected_route is Route.LOCAL and not relevance:
        raise DatasetValidationError(f"{location}: local route requires relevant sources")
    if expected_route is not Route.LOCAL and relevance:
        raise DatasetValidationError(f"{location}: non-local route cannot have relevant sources")
    if allow_web != (expected_route is Route.WEB):
        raise DatasetValidationError(
            f"{location}: allow_web must be true only for the web route"
        )
    questions_value = value["questions"]
    if not isinstance(questions_value, list) or not 2 <= len(questions_value) <= 20:
        raise DatasetValidationError(f"{location}.questions must contain 2 to 20 items")
    questions = tuple(
        _text(question, f"{location}.questions[{index}]", minimum=6, maximum=300)
        for index, question in enumerate(questions_value, start=1)
    )
    return RetrievalCaseFamily(
        family_id=family_id,
        category=category,
        split=split,
        difficulty=difficulty,
        expected_route=expected_route,
        allow_web=allow_web,
        relevance=relevance,
        questions=questions,
    )


def _relevance(value: object, location: str, corpus_root: Path) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{location}.relevance must be an object")
    relevance: list[tuple[str, int]] = []
    for raw_source, grade in value.items():
        source = _relative_source(raw_source, f"{location}.relevance")
        _resolve_source(corpus_root, source)
        if isinstance(grade, bool) or not isinstance(grade, int) or not 1 <= grade <= 3:
            raise DatasetValidationError(
                f"{location}.relevance grade for {source!r} must be an integer from 1 to 3"
            )
        relevance.append((source, grade))
    return tuple(sorted(relevance))


def _validate_requirements(
    families: Sequence[RetrievalCaseFamily],
    cases: Sequence[RetrievalBenchmarkCase],
    requirements: SuiteRequirements,
) -> None:
    if len(cases) < requirements.minimum_cases:
        raise DatasetValidationError(
            f"suite contains {len(cases)} cases; requires at least {requirements.minimum_cases}"
        )
    if len(families) < requirements.minimum_families:
        raise DatasetValidationError(
            f"suite contains {len(families)} families; requires at least "
            f"{requirements.minimum_families}"
        )
    category_count = len({family.category for family in families})
    if category_count < requirements.minimum_categories:
        raise DatasetValidationError(
            f"suite contains {category_count} categories; requires at least "
            f"{requirements.minimum_categories}"
        )
    _enforce_counts(
        _case_counts(families, "split"), requirements.minimum_cases_per_split, "split"
    )
    _enforce_counts(
        _case_counts(families, "expected_route"),
        requirements.minimum_cases_per_route,
        "route",
    )
    _enforce_counts(
        _case_counts(families, "difficulty"),
        requirements.minimum_cases_per_difficulty,
        "difficulty",
    )


def _case_counts(
    families: Sequence[RetrievalCaseFamily], attribute: str
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for family in families:
        raw_value = getattr(family, attribute)
        value = raw_value.value if isinstance(raw_value, Route) else raw_value
        counts[value] += len(family.questions)
    return dict(sorted(counts.items()))


def _enforce_counts(
    actual: Mapping[str, int], required: Sequence[tuple[str, int]], dimension: str
) -> None:
    failures = [
        f"{name}={actual.get(name, 0)}<{minimum}"
        for name, minimum in required
        if actual.get(name, 0) < minimum
    ]
    if failures:
        raise DatasetValidationError(
            f"suite does not meet minimum cases per {dimension}: " + ", ".join(failures)
        )


def _count_requirements(
    value: object, allowed: Sequence[str], location: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{location} must be an object")
    _exact_fields(value, frozenset(allowed), location)
    return tuple((name, _positive_int(value[name], f"{location}.{name}")) for name in allowed)


def _resolve_corpus_root(manifest_path: Path, value: object) -> Path:
    raw_path = _relative_source(value, "suite.corpus_root")
    root = (manifest_path.parent / Path(raw_path)).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise DatasetValidationError("suite.corpus_root must be a regular directory")
    return root


def _resolve_source(corpus_root: Path, source: str) -> Path:
    try:
        path = (corpus_root / Path(source)).resolve(strict=True)
    except OSError as error:
        raise DatasetValidationError(f"cannot resolve corpus source {source!r}: {error}") from error
    try:
        path.relative_to(corpus_root)
    except ValueError:
        raise DatasetValidationError(f"corpus source escapes root: {source!r}") from None
    if not path.is_file() or path.is_symlink():
        raise DatasetValidationError(f"corpus source must be a regular file: {source!r}")
    return path


def _relative_source(value: object, location: str) -> str:
    text = _text(value, location, minimum=1, maximum=200)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or text.startswith(("/", "\\")):
        raise DatasetValidationError(f"{location} must be a safe relative path")
    return path.as_posix()


def _question_fingerprint(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _bundle_digest(
    payload: Mapping[str, Any], documents: Sequence[Path], corpus_root: Path
) -> str:
    corpus_hashes: dict[str, str] = {}
    for document in documents:
        source = document.relative_to(corpus_root).as_posix()
        try:
            content = document.read_bytes()
        except OSError as error:
            raise DatasetValidationError(f"cannot hash corpus source {source!r}: {error}") from error
        corpus_hashes[source] = hashlib.sha256(content).hexdigest()
    canonical = json.dumps(
        {"manifest": payload, "corpus_sha256": corpus_hashes},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _exact_fields(value: Mapping[object, object], expected: frozenset[str], location: str) -> None:
    raw_keys = set(value)
    if not all(isinstance(key, str) for key in raw_keys):
        raise DatasetValidationError(f"{location} field names must be strings")
    keys = {key for key in raw_keys if isinstance(key, str)}
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise DatasetValidationError(f"{location} missing fields: {', '.join(missing)}")
    if unknown:
        raise DatasetValidationError(f"{location} unknown fields: {', '.join(unknown)}")


def _identifier(value: object, location: str) -> str:
    text = _text(value, location, minimum=3, maximum=64)
    if _IDENTIFIER.fullmatch(text) is None:
        raise DatasetValidationError(f"{location} must use lowercase snake_case")
    return text


def _enum(value: object, allowed: Sequence[str], location: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise DatasetValidationError(f"{location} must be one of: {', '.join(allowed)}")
    return value


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetValidationError(f"{location} must be a positive integer")
    return value


def _text(value: object, location: str, *, minimum: int, maximum: int) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{location} must be a string")
    text = value.strip()
    if not minimum <= len(text) <= maximum:
        raise DatasetValidationError(
            f"{location} length must be between {minimum} and {maximum} characters"
        )
    return text


__all__ = [
    "RetrievalBenchmarkSuite",
    "RetrievalCaseFamily",
    "SuiteRequirements",
    "load_retrieval_suite",
    "validate_suite_contract",
]
