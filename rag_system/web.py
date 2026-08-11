"""Deterministic quality filters for untrusted web-search results."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rag_system.domain import WebSearchResult
from rag_system.security import safe_external_url
from rag_system.text import lexical_relevance, normalize_text


_TRACKING_PARAMETERS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_src",
        "spm",
    }
)


@dataclass(frozen=True, slots=True)
class RankedWebResult:
    result: WebSearchResult
    score: float
    canonical_url: str
    domain: str


def canonicalize_url(value: str) -> str:
    """Normalize a source URL and remove common tracking-only parameters."""

    safe_url = safe_external_url(value)
    if not safe_url:
        return ""
    parts = urlsplit(safe_url)
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return ""
    port = parts.port
    default_port = (parts.scheme.lower() == "http" and port == 80) or (
        parts.scheme.lower() == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_PARAMETERS
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def rank_web_results(
    question: str,
    results: Sequence[WebSearchResult],
    *,
    limit: int = 5,
    per_domain: int = 2,
) -> tuple[RankedWebResult, ...]:
    """Remove duplicate/unsafe results and favor relevant, diverse evidence."""

    if limit < 1 or per_domain < 1:
        raise ValueError("limit and per_domain must be positive")
    query = normalize_text(question)
    if not query:
        return ()

    candidates: list[RankedWebResult] = []
    seen_urls: set[str] = set()
    seen_content: set[str] = set()
    for result in results:
        title = normalize_text(result.title)
        content = normalize_text(result.content)
        canonical_url = canonicalize_url(result.url)
        if not title and not content:
            continue

        content_identity = normalize_text(f"{title}\n{content}").lower()
        fingerprint = hashlib.sha256(content_identity.encode("utf-8")).hexdigest()
        if (canonical_url and canonical_url in seen_urls) or fingerprint in seen_content:
            continue
        if canonical_url:
            seen_urls.add(canonical_url)
        seen_content.add(fingerprint)

        domain = (urlsplit(canonical_url).hostname or "") if canonical_url else "unknown"
        title_relevance = lexical_relevance(query, title)
        body_relevance = lexical_relevance(query, content)
        completeness = min(1.0, len(content) / 500) if content else 0.0
        verifiability = 1.0 if canonical_url else 0.0
        score = min(
            1.0,
            0.35 * title_relevance
            + 0.40 * body_relevance
            + 0.15 * completeness
            + 0.10 * verifiability,
        )
        candidates.append(
            RankedWebResult(
                result=WebSearchResult(
                    result_id=result.result_id,
                    title=title or "未命名来源",
                    content=content,
                    url=canonical_url,
                ),
                score=score,
                canonical_url=canonical_url,
                domain=domain,
            )
        )

    candidates.sort(
        key=lambda item: (-item.score, item.domain, item.result.result_id)
    )
    selected: list[RankedWebResult] = []
    domain_counts: dict[str, int] = {}
    for candidate in candidates:
        if domain_counts.get(candidate.domain, 0) >= per_domain:
            continue
        selected.append(candidate)
        domain_counts[candidate.domain] = domain_counts.get(candidate.domain, 0) + 1
        if len(selected) >= limit:
            break
    return tuple(selected)


__all__ = ["RankedWebResult", "canonicalize_url", "rank_web_results"]
