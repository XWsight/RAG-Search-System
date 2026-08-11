"""Framework-neutral domain objects passed across application boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class Route(StrEnum):
    LOCAL = "local"
    WEB = "web"
    HYBRID = "hybrid"
    RETRIEVAL_ONLY = "retrieval_only"
    REFUSED = "refused"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SourceDocument:
    document_id: str
    name: str
    text: str
    content_hash: str
    encoding: str = "utf-8"


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    heading: str = ""


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_distance: float | None = None
    reasons: tuple[str, ...] = ()
    lexical_score: float | None = None


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    result_id: str
    title: str
    content: str
    url: str


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    source_name: str
    excerpt: str
    url: str = ""
    score: float | None = None


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    """One atomic generated statement and the evidence IDs attributed to it."""

    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Structured output produced by a chat model before application validation."""

    claims: tuple[AnswerClaim, ...]
    insufficient: bool = False


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class IndexRef:
    index_id: str
    document_count: int
    chunk_count: int
    created_at: float


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    question: str
    session_id: str
    allow_cloud: bool = False
    allow_web: bool = False
    deep_research: bool = False


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    decision: RouteDecision
    claims: tuple[AnswerClaim, ...] = ()
    citations: tuple[Citation, ...] = ()
    hits: tuple[SearchHit, ...] = ()
    trace_id: str = ""
    latency_ms: float = 0.0
    diagnostics: Mapping[str, float | int | str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )
