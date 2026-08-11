"""Structured claim-to-evidence contracts for grounded generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_system.domain import AnswerClaim, GeneratedAnswer


_CITATION_ID = re.compile(r"^(?:L|W)[1-9]\d*$")
_INLINE_CITATION = re.compile(r"\[(?:L|W)\d+\]")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_CLAIMS = 24
_MAX_CLAIM_CHARACTERS = 2_000
_MAX_ANSWER_CHARACTERS = 20_000


class GroundingContractError(ValueError):
    """Raised when generated claims cannot satisfy the evidence contract."""


@dataclass(frozen=True, slots=True)
class GroundingAudit:
    claim_count: int
    citation_count: int
    used_citation_ids: tuple[str, ...]


def validate_grounded_answer(
    draft: GeneratedAnswer,
    allowed_citation_ids: Sequence[str],
) -> GroundingAudit:
    """Validate a complete generated answer against the exact evidence registry.

    Validation is intentionally all-or-nothing. Silently deleting an invalid
    citation can leave a claim looking grounded when its evidence was removed.
    """

    if not isinstance(draft, GeneratedAnswer):
        raise GroundingContractError("generated answer has an invalid type")
    if not isinstance(draft.insufficient, bool):
        raise GroundingContractError("insufficient must be a boolean")
    if not isinstance(draft.claims, tuple):
        raise GroundingContractError("claims must be an immutable tuple")
    if len(draft.claims) > _MAX_CLAIMS:
        raise GroundingContractError("generated answer contains too many claims")
    if draft.insufficient and draft.claims:
        raise GroundingContractError("an insufficient answer cannot contain claims")
    if not draft.insufficient and not draft.claims:
        raise GroundingContractError("a sufficient answer must contain claims")

    allowed = _validated_allowed_ids(allowed_citation_ids)
    seen_claims: set[str] = set()
    used_ids: list[str] = []
    total_characters = 0
    citation_count = 0

    for claim in draft.claims:
        if not isinstance(claim, AnswerClaim):
            raise GroundingContractError("claims contain an invalid item")
        if not isinstance(claim.text, str):
            raise GroundingContractError("claim text must be a string")
        text = claim.text.strip()
        if not text or len(text) > _MAX_CLAIM_CHARACTERS:
            raise GroundingContractError("claim text is empty or too long")
        if text != claim.text:
            raise GroundingContractError("claim text must be normalized")
        if _CONTROL_CHARACTERS.search(text):
            raise GroundingContractError("claim text contains control characters")
        if _INLINE_CITATION.search(text):
            raise GroundingContractError("claim text cannot contain citation markup")
        identity = text.casefold()
        if identity in seen_claims:
            raise GroundingContractError("duplicate claims are not allowed")
        seen_claims.add(identity)
        total_characters += len(text)
        if total_characters > _MAX_ANSWER_CHARACTERS:
            raise GroundingContractError("generated answer is too long")

        if not isinstance(claim.citation_ids, tuple) or not claim.citation_ids:
            raise GroundingContractError("every claim must cite evidence")
        if len(claim.citation_ids) != len(set(claim.citation_ids)):
            raise GroundingContractError("claim citation IDs must be unique")
        for citation_id in claim.citation_ids:
            if citation_id not in allowed:
                raise GroundingContractError("claim references unavailable evidence")
            citation_count += 1
            if citation_id not in used_ids:
                used_ids.append(citation_id)

    return GroundingAudit(
        claim_count=len(draft.claims),
        citation_count=citation_count,
        used_citation_ids=tuple(used_ids),
    )


def render_grounded_answer(draft: GeneratedAnswer) -> str:
    """Render validated claims into stable plain text for existing clients."""

    if draft.insufficient:
        return "现有资料不足以回答这个问题。"
    paragraphs = [
        f"{claim.text} {' '.join(f'[{item}]' for item in claim.citation_ids)}"
        for claim in draft.claims
    ]
    return "\n\n".join(paragraphs)


def _validated_allowed_ids(values: Sequence[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise GroundingContractError("allowed citation IDs must be a sequence")
    resolved = tuple(values)
    if len(resolved) != len(set(resolved)):
        raise GroundingContractError("allowed citation IDs must be unique")
    for value in resolved:
        if not isinstance(value, str) or _CITATION_ID.fullmatch(value) is None:
            raise GroundingContractError("allowed citation ID is invalid")
    return frozenset(resolved)
