"""Bounded multi-query retrieval primitives for the research mode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from rag_system.domain import SearchHit
from rag_system.ranking import reciprocal_rank_fusion


def normalize_query_plan(
    original_question: str,
    planned_queries: Sequence[str],
    *,
    max_queries: int,
    max_characters: int = 70,
) -> tuple[str, ...]:
    """Keep a unique, bounded plan and always retain the original question."""

    if max_queries < 1 or max_characters < 1:
        raise ValueError("query plan limits must be positive")
    original = (original_question or "").strip()
    if not original:
        raise ValueError("original_question cannot be empty")

    resolved: list[str] = []
    seen: set[str] = set()
    for raw_query in (original, *planned_queries):
        if not isinstance(raw_query, str):
            continue
        query = raw_query.strip()[:max_characters]
        identity = query.casefold()
        if not query or identity in seen:
            continue
        resolved.append(query)
        seen.add(identity)
        if len(resolved) >= max_queries:
            break
    return tuple(resolved)


def fuse_query_hits(
    rankings: Mapping[str, Sequence[SearchHit]],
    *,
    top_k: int,
) -> tuple[SearchHit, ...]:
    """Fuse multiple query rankings by RRF while retaining calibrated scores."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not rankings:
        return ()

    rank_ids: dict[str, list[str]] = {}
    best_by_id: dict[str, SearchHit] = {}
    for query_id, hits in rankings.items():
        identifiers: list[str] = []
        for hit in hits:
            chunk_id = hit.chunk.chunk_id
            if chunk_id in identifiers:
                continue
            identifiers.append(chunk_id)
            existing = best_by_id.get(chunk_id)
            if existing is None or hit.score > existing.score:
                best_by_id[chunk_id] = hit
        rank_ids[query_id] = identifiers

    fused = reciprocal_rank_fusion(rank_ids)
    maximum_rrf = max(1, len(rank_ids)) / 61
    results: list[SearchHit] = []
    for item in fused:
        hit = best_by_id.get(item.item_id)
        if hit is None:
            continue
        rrf_score = min(1.0, item.score / maximum_rrf)
        score = min(1.0, 0.55 * hit.score + 0.45 * rrf_score)
        results.append(
            replace(
                hit,
                score=score,
                reasons=tuple(dict.fromkeys((*hit.reasons, "multi_query"))),
            )
        )
    results.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
    return tuple(_diverse_sources(results, top_k))


def _diverse_sources(hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
    selected: list[SearchHit] = []
    per_document: dict[str, int] = {}
    for hit in hits:
        document_id = hit.chunk.document_id
        if per_document.get(document_id, 0) >= 2:
            continue
        selected.append(hit)
        per_document[document_id] = per_document.get(document_id, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


__all__ = ["fuse_query_hits", "normalize_query_plan"]
