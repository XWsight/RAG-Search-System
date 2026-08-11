"""Framework-independent ranking and citation quality helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


_CITATION_PATTERN = re.compile(r"\[((?:L|W)\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")


@dataclass(frozen=True, slots=True)
class RankedItem:
    item_id: str
    score: float
    contributing_rankers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationAudit:
    cited_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]
    uncited_sentences: tuple[str, ...]
    cited_sentence_count: int = 0

    @property
    def valid(self) -> bool:
        return not self.invalid_ids

    @property
    def completeness(self) -> float:
        claim_count = len(self.uncited_sentences) + self.cited_sentence_count
        if claim_count == 0:
            return 1.0
        return self.cited_sentence_count / claim_count

def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float] | None = None,
    rank_constant: int = 60,
) -> tuple[RankedItem, ...]:
    """Fuse heterogeneous rankings using weighted reciprocal rank fusion.

    RRF operates on ranks rather than incomparable raw scores. Duplicate item
    IDs inside one ranker are ignored after their first occurrence.
    """

    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    resolved_weights = dict(weights or {})
    scores: dict[str, float] = {}
    contributors: dict[str, list[str]] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0

    for ranker_name, item_ids in rankings.items():
        weight = resolved_weights.get(ranker_name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {ranker_name!r} cannot be negative")

        ranker_seen: set[str] = set()
        for rank, item_id in enumerate(item_ids, start=1):
            if not item_id or item_id in ranker_seen:
                continue
            ranker_seen.add(item_id)
            if item_id not in first_seen:
                first_seen[item_id] = seen_order
                seen_order += 1
            scores[item_id] = scores.get(item_id, 0.0) + weight / (rank_constant + rank)
            contributors.setdefault(item_id, []).append(ranker_name)

    ordered_ids = sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
    return tuple(
        RankedItem(
            item_id=item_id,
            score=scores[item_id],
            contributing_rankers=tuple(contributors[item_id]),
        )
        for item_id in ordered_ids
    )


def extract_citation_ids(answer: str) -> tuple[str, ...]:
    """Extract unique local/web citation IDs while preserving answer order."""

    return tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))


def audit_citations(answer: str, allowed_ids: Sequence[str]) -> CitationAudit:
    """Validate citation IDs and flag answer sentences without evidence tags."""

    allowed = set(allowed_ids)
    cited = extract_citation_ids(answer)
    invalid = tuple(citation_id for citation_id in cited if citation_id not in allowed)

    uncited: list[str] = []
    cited_sentence_count = 0
    for sentence in _SENTENCE_SPLIT.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence or len(sentence) < 4:
            continue
        if _CITATION_PATTERN.search(sentence):
            cited_sentence_count += 1
        else:
            uncited.append(sentence)

    return CitationAudit(
        cited_ids=cited,
        invalid_ids=invalid,
        uncited_sentences=tuple(uncited),
        cited_sentence_count=cited_sentence_count,
    )
