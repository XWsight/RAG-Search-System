"""Optional cross-encoder reranking behind a small, testable boundary."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from rag_system.domain import SearchHit


class RerankerError(RuntimeError):
    """An optional reranker could not produce a valid ranking."""


class CrossEncoderReranker:
    """Blend cross-encoder relevance with first-stage hybrid scores.

    The model is loaded on the first query.  Keeping this component optional
    avoids a large download and slower CPU inference in the default profile.
    """

    def __init__(
        self,
        model_name: str,
        *,
        weight: float = 0.65,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        self.model_name = model_name.strip()
        self.weight = weight
        self._model_factory = model_factory or self._default_factory
        self._model: Any | None = None
        self._lock = threading.RLock()

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        top_k: int,
    ) -> tuple[SearchHit, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        question = (query or "").strip()
        if not question or not hits:
            return ()

        pairs = [(question, hit.chunk.text) for hit in hits]
        try:
            with self._lock:
                model = self._get_model()
                raw_scores = model.predict(
                    pairs,
                    batch_size=min(16, len(pairs)),
                    show_progress_bar=False,
                )
                if hasattr(raw_scores, "tolist"):
                    raw_scores = raw_scores.tolist()
                scores = list(raw_scores)
        except Exception:
            raise RerankerError("reranker inference failed") from None
        if len(scores) != len(hits):
            raise RerankerError("reranker returned an unexpected score count")

        reranked: list[SearchHit] = []
        for hit, raw_score in zip(hits, scores, strict=True):
            try:
                numeric_score = float(raw_score)
            except (TypeError, ValueError):
                raise RerankerError("reranker returned a non-numeric score") from None
            if not math.isfinite(numeric_score):
                raise RerankerError("reranker returned a non-finite score")
            cross_score = _sigmoid(numeric_score)
            score = (1.0 - self.weight) * hit.score + self.weight * cross_score
            reranked.append(
                replace(
                    hit,
                    score=max(0.0, min(1.0, score)),
                    reasons=tuple(dict.fromkeys((*hit.reasons, "rerank"))),
                )
            )
        reranked.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(reranked[:top_k])

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    self._model = self._model_factory(self.model_name)
                except Exception:
                    raise RerankerError("reranker model could not be loaded") from None
            return self._model

    @staticmethod
    def _default_factory(model_name: str) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RerankerError("sentence-transformers is not installed") from None
        return CrossEncoder(model_name)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


__all__ = ["CrossEncoderReranker", "RerankerError"]
