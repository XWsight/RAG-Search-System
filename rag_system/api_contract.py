"""Versioned HTTP schemas and domain-to-wire projections."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_system.catalog import KnowledgeBaseRecord, KnowledgeBaseStatus
from rag_system.domain import AnswerClaim, AnswerResult, Citation, Route
from rag_system.grounding import (
    CITATION_ID_PATTERN,
    MAX_ANSWER_CLAIMS,
    MAX_CITATION_ID_CHARACTERS,
    MAX_CLAIM_CHARACTERS,
    MAX_GROUNDED_ANSWER_CHARACTERS,
)
from rag_system.job_contracts import JobSnapshot, JobStatus


RESOURCE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
SESSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
CitationId = Annotated[
    str,
    Field(
        min_length=2,
        max_length=MAX_CITATION_ID_CHARACTERS,
        pattern=CITATION_ID_PATTERN,
    ),
]


class StrictModel(BaseModel):
    """Forbid unknown input fields and implicit type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ErrorDetail(StrictModel):
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)


class ErrorEnvelope(StrictModel):
    error: ErrorDetail


class HealthResponse(StrictModel):
    status: Literal["ok", "ready"]


class DocumentResponse(StrictModel):
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeBaseResponse(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    status: KnowledgeBaseStatus
    documents: tuple[DocumentResponse, ...]
    document_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    error_code: str | None = Field(default=None, max_length=64)
    created_at: float = Field(ge=0)
    updated_at: float = Field(ge=0)
    version: int = Field(ge=1)


class KnowledgeBaseSubmissionResponse(StrictModel):
    knowledge_base: KnowledgeBaseResponse
    job_id: str = Field(min_length=1, max_length=128)
    replayed: bool


class KnowledgeBaseListResponse(StrictModel):
    items: tuple[KnowledgeBaseResponse, ...]
    count: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class DeleteResponse(StrictModel):
    deleted: bool


class JobResponse(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    status: JobStatus
    created_at: float = Field(ge=0)
    updated_at: float = Field(ge=0)
    started_at: float | None = Field(default=None, ge=0)
    finished_at: float | None = Field(default=None, ge=0)
    result: dict[str, Any] | None = None
    error_code: str | None = Field(default=None, max_length=64)


class AnswerPayload(StrictModel):
    knowledge_base_id: str = Field(min_length=1, max_length=128, pattern=RESOURCE_PATTERN)
    question: str = Field(min_length=1, max_length=10_000)
    session_id: str = Field(min_length=1, max_length=128, pattern=SESSION_PATTERN)
    allow_cloud: bool = False
    allow_web: bool = False
    deep_research: bool = False


class RouteResponse(StrictModel):
    route: Route
    confidence: float = Field(ge=0, le=1)


class CitationResponse(StrictModel):
    id: CitationId
    source_name: str = Field(min_length=1, max_length=255)
    excerpt: str = Field(max_length=8_000)
    url: str = Field(default="", max_length=4_096)
    score: float | None = None


class AnswerClaimResponse(StrictModel):
    text: str = Field(min_length=1, max_length=MAX_CLAIM_CHARACTERS)
    citation_ids: tuple[CitationId, ...] = Field(
        min_length=1,
        max_length=MAX_ANSWER_CLAIMS,
    )


class AnswerResponse(StrictModel):
    answer: str = Field(max_length=MAX_GROUNDED_ANSWER_CHARACTERS)
    decision: RouteResponse
    claims: tuple[AnswerClaimResponse, ...] = Field(max_length=MAX_ANSWER_CLAIMS)
    citations: tuple[CitationResponse, ...]
    trace_id: str = Field(min_length=1, max_length=128)
    latency_ms: float = Field(ge=0)


def knowledge_base_response(record: KnowledgeBaseRecord) -> KnowledgeBaseResponse:
    documents = tuple(
        DocumentResponse(name=item.display_name, size_bytes=item.size_bytes, sha256=item.sha256)
        for item in record.documents
    )
    return KnowledgeBaseResponse(
        id=record.resource_id,
        name=record.display_name,
        status=record.status,
        documents=documents,
        document_count=record.document_count,
        total_bytes=record.total_bytes,
        chunk_count=record.chunk_count,
        error_code=record.error_code.value if record.error_code is not None else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
        version=record.version,
    )


def job_response(snapshot: JobSnapshot) -> JobResponse:
    return JobResponse(
        id=snapshot.job_id.value,
        status=snapshot.status,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
        result=snapshot.result,
        error_code=snapshot.error_code or None,
    )


def citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        id=citation.citation_id,
        source_name=citation.source_name,
        excerpt=citation.excerpt,
        url=citation.url,
        score=citation.score,
    )


def claim_response(claim: AnswerClaim) -> AnswerClaimResponse:
    return AnswerClaimResponse(text=claim.text, citation_ids=claim.citation_ids)


def answer_response(result: AnswerResult, *, trace_id: str) -> AnswerResponse:
    return AnswerResponse(
        answer=result.answer,
        decision=RouteResponse(
            route=result.decision.route,
            confidence=max(0.0, min(1.0, result.decision.confidence)),
        ),
        claims=tuple(claim_response(item) for item in result.claims),
        citations=tuple(citation_response(item) for item in result.citations),
        trace_id=trace_id,
        latency_ms=max(0.0, result.latency_ms),
    )


__all__ = [
    "AnswerPayload",
    "AnswerResponse",
    "DeleteResponse",
    "DocumentResponse",
    "ErrorDetail",
    "ErrorEnvelope",
    "HealthResponse",
    "JobResponse",
    "KnowledgeBaseListResponse",
    "KnowledgeBaseResponse",
    "KnowledgeBaseSubmissionResponse",
    "RESOURCE_PATTERN",
    "SESSION_PATTERN",
    "answer_response",
    "job_response",
    "knowledge_base_response",
]
