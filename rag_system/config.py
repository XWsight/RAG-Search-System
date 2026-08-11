"""Centralized, validated application configuration."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class SecretValue:
    """Small secret wrapper that never exposes its value through repr/str."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        self._value = (value or "").strip()

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "SecretValue('********')" if self._value else "SecretValue('')"

    __str__ = __repr__


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded once at application bootstrap."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    storage_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "RAG_STORAGE_ROOT",
                str(Path(__file__).resolve().parent.parent / ".rag_data"),
            )
        )
    )
    persist_data: bool = field(default_factory=lambda: _env_bool("RAG_PERSIST_DATA", False))
    api_keys_json: SecretValue = field(
        default_factory=lambda: SecretValue(os.getenv("RAG_API_KEYS_JSON"))
    )
    api_docs_enabled: bool = field(
        default_factory=lambda: _env_bool("RAG_API_DOCS_ENABLED", True)
    )
    api_key: SecretValue = field(default_factory=lambda: SecretValue(os.getenv("ZHIPU_API_KEY")))
    chat_model: str = field(default_factory=lambda: os.getenv("ZHIPU_MODEL", "glm-5.2"))
    chat_url: str = field(
        default_factory=lambda: os.getenv(
            "ZHIPU_CHAT_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        )
    )
    search_url: str = field(
        default_factory=lambda: os.getenv(
            "ZHIPU_SEARCH_URL", "https://open.bigmodel.cn/api/paas/v4/web_search"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    reranker_model: str = field(default_factory=lambda: os.getenv("RAG_RERANKER_MODEL", "").strip())
    reranker_weight: float = field(
        default_factory=lambda: _env_float("RAG_RERANKER_WEIGHT", 0.65)
    )
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 480))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 80))
    dense_candidates: int = field(default_factory=lambda: _env_int("RAG_DENSE_CANDIDATES", 24))
    sparse_candidates: int = field(default_factory=lambda: _env_int("RAG_SPARSE_CANDIDATES", 24))
    fused_candidates: int = field(default_factory=lambda: _env_int("RAG_FUSED_CANDIDATES", 12))
    final_evidence_count: int = field(default_factory=lambda: _env_int("RAG_FINAL_EVIDENCE", 5))
    local_confidence_threshold: float = field(
        default_factory=lambda: _env_float("RAG_LOCAL_CONFIDENCE", 0.590000)
    )
    hybrid_confidence_ratio: float = field(
        default_factory=lambda: _env_float("RAG_HYBRID_CONFIDENCE_RATIO", 0.95)
    )
    routing_lexical_saturation: float = field(
        default_factory=lambda: _env_float("RAG_ROUTING_LEXICAL_SATURATION", 0.30)
    )
    max_file_bytes: int = field(default_factory=lambda: _env_int("RAG_MAX_FILE_BYTES", 5 * 1024 * 1024))
    max_total_bytes: int = field(default_factory=lambda: _env_int("RAG_MAX_TOTAL_BYTES", 20 * 1024 * 1024))
    max_documents: int = field(default_factory=lambda: _env_int("RAG_MAX_DOCUMENTS", 10))
    max_tenant_storage_bytes: int = field(
        default_factory=lambda: _env_int("RAG_MAX_TENANT_STORAGE_BYTES", 1024 * 1024 * 1024)
    )
    max_files_per_tenant: int = field(
        default_factory=lambda: _env_int("RAG_MAX_FILES_PER_TENANT", 1_000)
    )
    max_chunks: int = field(default_factory=lambda: _env_int("RAG_MAX_CHUNKS", 2_000))
    max_document_characters: int = field(
        default_factory=lambda: _env_int("RAG_MAX_DOCUMENT_CHARACTERS", 2_000_000)
    )
    max_pdf_pages: int = field(default_factory=lambda: _env_int("RAG_MAX_PDF_PAGES", 200))
    max_archive_uncompressed_bytes: int = field(
        default_factory=lambda: _env_int(
            "RAG_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 20 * 1024 * 1024
        )
    )
    max_question_characters: int = field(
        default_factory=lambda: _env_int("RAG_MAX_QUESTION_CHARACTERS", 2_000)
    )
    max_context_characters: int = field(
        default_factory=lambda: _env_int("RAG_MAX_CONTEXT_CHARACTERS", 18_000)
    )
    answer_max_tokens: int = field(
        default_factory=lambda: _env_int("RAG_ANSWER_MAX_TOKENS", 4_096)
    )
    query_plan_max_tokens: int = field(
        default_factory=lambda: _env_int("RAG_QUERY_PLAN_MAX_TOKENS", 512)
    )
    connect_timeout_seconds: float = field(
        default_factory=lambda: _env_float("RAG_CONNECT_TIMEOUT", 10.0)
    )
    read_timeout_seconds: float = field(default_factory=lambda: _env_float("RAG_READ_TIMEOUT", 60.0))
    retry_attempts: int = field(default_factory=lambda: _env_int("RAG_RETRY_ATTEMPTS", 2))
    job_workers: int = field(default_factory=lambda: _env_int("RAG_JOB_WORKERS", 4))
    max_jobs: int = field(default_factory=lambda: _env_int("RAG_MAX_JOBS", 128))
    max_jobs_per_tenant: int = field(
        default_factory=lambda: _env_int("RAG_MAX_JOBS_PER_TENANT", 32)
    )
    job_ttl_seconds: int = field(default_factory=lambda: _env_int("RAG_JOB_TTL", 3_600))
    job_history_ttl_seconds: int = field(
        default_factory=lambda: _env_int("RAG_JOB_HISTORY_TTL", 7 * 24 * 60 * 60)
    )
    job_history_max_per_tenant: int = field(
        default_factory=lambda: _env_int("RAG_JOB_HISTORY_MAX_PER_TENANT", 10_000)
    )
    max_concurrent_answers: int = field(
        default_factory=lambda: _env_int("RAG_MAX_CONCURRENT_ANSWERS", 4)
    )
    rate_limit_per_second: float = field(
        default_factory=lambda: _env_float("RAG_RATE_LIMIT_PER_SECOND", 2.0)
    )
    rate_limit_capacity: float = field(
        default_factory=lambda: _env_float("RAG_RATE_LIMIT_CAPACITY", 20.0)
    )
    rate_limit_max_tenants: int = field(
        default_factory=lambda: _env_int("RAG_RATE_LIMIT_MAX_TENANTS", 10_000)
    )
    session_ttl_seconds: int = field(default_factory=lambda: _env_int("RAG_SESSION_TTL", 3_600))
    max_sessions: int = field(default_factory=lambda: _env_int("RAG_MAX_SESSIONS", 32))
    memory_max_rounds: int = field(default_factory=lambda: _env_int("RAG_MEMORY_MAX_ROUNDS", 8))
    memory_max_characters: int = field(
        default_factory=lambda: _env_int("RAG_MEMORY_MAX_CHARACTERS", 6_000)
    )
    retrieval_history_characters: int = field(
        default_factory=lambda: _env_int("RAG_RETRIEVAL_HISTORY_CHARACTERS", 1_000)
    )
    research_max_queries: int = field(
        default_factory=lambda: _env_int("RAG_RESEARCH_MAX_QUERIES", 4)
    )
    research_max_web_queries: int = field(
        default_factory=lambda: _env_int("RAG_RESEARCH_MAX_WEB_QUERIES", 3)
    )
    allow_cloud_default: bool = field(
        default_factory=lambda: _env_bool("RAG_ALLOW_CLOUD_DEFAULT", False)
    )
    allow_web_default: bool = field(default_factory=lambda: _env_bool("RAG_ALLOW_WEB_DEFAULT", False))

    @property
    def default_document(self) -> Path:
        return self.project_root / "data" / "knowledge.txt"

    def validate(self) -> Settings:
        if not str(self.storage_root).strip():
            raise ValueError("storage_root cannot be empty")
        if not 100 <= self.chunk_size <= 4_000:
            raise ValueError("chunk_size must be between 100 and 4000")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 1 <= self.final_evidence_count <= self.fused_candidates:
            raise ValueError("final_evidence_count must be within fused candidate count")
        if self.dense_candidates < self.fused_candidates:
            raise ValueError("dense_candidates cannot be smaller than fused_candidates")
        if self.sparse_candidates < self.fused_candidates:
            raise ValueError("sparse_candidates cannot be smaller than fused_candidates")
        if not 0.0 <= self.local_confidence_threshold <= 1.0:
            raise ValueError("local_confidence_threshold must be between 0 and 1")
        if not 0.5 <= self.hybrid_confidence_ratio < 1.0:
            raise ValueError("hybrid_confidence_ratio must be at least 0.5 and below 1.0")
        if not 0.05 <= self.routing_lexical_saturation <= 1.0:
            raise ValueError("routing_lexical_saturation must be between 0.05 and 1.0")
        if not 0.0 <= self.reranker_weight <= 1.0:
            raise ValueError("reranker_weight must be between 0 and 1")
        if self.max_file_bytes < 1 or self.max_total_bytes < self.max_file_bytes:
            raise ValueError("invalid file size limits")
        if self.max_tenant_storage_bytes < self.max_file_bytes:
            raise ValueError("tenant storage must fit at least one maximum-size file")
        if self.max_files_per_tenant < self.max_documents:
            raise ValueError("tenant file capacity must fit one maximum-size upload")
        if (
            self.max_documents < 1
            or self.max_chunks < 1
            or self.max_document_characters < 1
            or self.max_pdf_pages < 1
            or self.max_archive_uncompressed_bytes < 1
        ):
            raise ValueError("document and chunk limits must be positive")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 300
            for value in (self.connect_timeout_seconds, self.read_timeout_seconds)
        ):
            raise ValueError("timeouts must be finite and between 0 and 300 seconds")
        if not 0 <= self.retry_attempts <= 5:
            raise ValueError("retry_attempts must be between 0 and 5")
        if (
            self.job_workers < 1
            or self.max_jobs < self.job_workers
            or not 1 <= self.max_jobs_per_tenant <= self.max_jobs
            or self.job_ttl_seconds < 1
            or self.job_history_ttl_seconds < max(60, self.job_ttl_seconds)
            or not self.max_jobs_per_tenant
            <= self.job_history_max_per_tenant
            <= 1_000_000
            or not 1 <= self.max_concurrent_answers <= 128
        ):
            raise ValueError("invalid concurrency or background job limits")
        if not 1 <= self.max_question_characters <= 10_000:
            raise ValueError("question character limit must be between 1 and 10000")
        if not 1_000 <= self.max_context_characters <= 200_000:
            raise ValueError("context character limit must be between 1000 and 200000")
        if not 512 <= self.answer_max_tokens <= 32_768:
            raise ValueError("answer token limit must be between 512 and 32768")
        if not 64 <= self.query_plan_max_tokens <= 4_096:
            raise ValueError("query plan token limit must be between 64 and 4096")
        if (
            not math.isfinite(self.rate_limit_per_second)
            or not math.isfinite(self.rate_limit_capacity)
            or self.rate_limit_per_second <= 0
            or self.rate_limit_capacity < 5
        ):
            raise ValueError("invalid rate limit configuration")
        if self.rate_limit_max_tenants < 1:
            raise ValueError("rate limit tenant capacity must be positive")
        if self.session_ttl_seconds < 60 or self.max_sessions < 1:
            raise ValueError("invalid session limits")
        if self.memory_max_rounds < 1 or self.memory_max_characters < 2:
            raise ValueError("invalid conversation memory limits")
        if not 1 <= self.retrieval_history_characters <= self.memory_max_characters:
            raise ValueError("retrieval history limit must fit within conversation memory")
        if not 1 <= self.research_max_queries <= 6:
            raise ValueError("research_max_queries must be between 1 and 6")
        if not 1 <= self.research_max_web_queries <= self.research_max_queries:
            raise ValueError("research web query limit must fit within research query limit")
        self._validate_https_url("chat_url", self.chat_url)
        self._validate_https_url("search_url", self.search_url)
        return self

    @staticmethod
    def _validate_https_url(name: str, value: str) -> None:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"{name} must be an absolute HTTPS URL")


def load_settings(*, dotenv_path: Path | None = None) -> Settings:
    """Load optional dotenv values, then create and validate settings."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return Settings().validate()

    load_dotenv(dotenv_path=dotenv_path)
    return Settings().validate()
