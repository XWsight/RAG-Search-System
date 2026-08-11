"""Compare retrieval stages over one shared local index without cloud calls."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.benchmark_suite import load_retrieval_suite  # noqa: E402
from rag_system.config import Settings, load_settings  # noqa: E402
from rag_system.evaluation import DatasetValidationError  # noqa: E402
from rag_system.index_manager import IndexManager  # noqa: E402
from rag_system.retrieval import (  # noqa: E402
    ChromaIndexRepository,
    FusionWeights,
    HybridRetriever,
    RetrievalProfile,
)
from rag_system.retrieval_experiments import run_retrieval_ablation  # noqa: E402
from rag_system.routing import RoutingPolicy  # noqa: E402


DEFAULT_PROFILES = (
    RetrievalProfile.DENSE.value,
    RetrievalProfile.SPARSE.value,
    RetrievalProfile.FUSION.value,
    RetrievalProfile.FUSION_DIVERSE.value,
)
_VARIANT_NAME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一个真实本地索引上比较 Dense、BM25、融合、多样化与可选重排。"
    )
    parser.add_argument("suite", type=Path, help="严格 retrieval suite JSON")
    parser.add_argument(
        "--split",
        choices=("development", "validation", "test"),
        help="仅运行一个按来源隔离的分段",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(profile.value for profile in RetrievalProfile),
        default=list(DEFAULT_PROFILES),
    )
    parser.add_argument("--baseline", default=RetrievalProfile.FUSION_DIVERSE.value)
    parser.add_argument(
        "--fusion-weight",
        action="append",
        default=[],
        type=_fusion_weight,
        metavar="NAME:DENSE:SPARSE:LEXICAL:RRF",
        help="增加一个使用指定归一化融合权重的 diversified 变体；可重复提供",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        help="显式加载指定 dotenv；默认不读取项目 .env",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manager: IndexManager | None = None
    try:
        settings = (
            load_settings(dotenv_path=arguments.dotenv)
            if arguments.dotenv is not None
            else Settings().validate()
        )
        suite = load_retrieval_suite(arguments.suite)
        cases = suite.cases_for_split(arguments.split) if arguments.split else suite.cases
        documents = (
            suite.documents_for_split(arguments.split)
            if arguments.split
            else suite.documents
        )
        profiles = tuple(dict.fromkeys(RetrievalProfile(item) for item in arguments.profiles))
        if len({name for name, _weights in arguments.fusion_weight}) != len(
            arguments.fusion_weight
        ):
            raise DatasetValidationError("fusion weight variant names must be unique")
        weighted_variants = dict(arguments.fusion_weight)
        variant_names = {profile.value for profile in profiles} | set(weighted_variants)
        if len(variant_names) < 2:
            raise DatasetValidationError("at least two unique profiles are required")
        if arguments.baseline not in variant_names:
            raise DatasetValidationError("baseline must be included in the selected variants")
        if RetrievalProfile.FUSION_DIVERSE_RERANK in profiles and not settings.reranker_model:
            raise DatasetValidationError(
                "fusion-diverse-rerank requires RAG_RERANKER_MODEL"
            )

        manager = IndexManager(settings, ChromaIndexRepository(settings))
        ingestion = manager.prepare(documents)
        build_started = time.perf_counter()
        index_ref = manager.build_prepared(ingestion)
        index_build_ms = (time.perf_counter() - build_started) * 1_000
        production_retriever = manager.get(index_ref.index_id)
        retrievers = {
            profile.value: HybridRetriever(
                production_retriever.vector_index,
                ingestion.chunks,
                settings,
                reranker=(
                    manager.reranker
                    if profile is RetrievalProfile.FUSION_DIVERSE_RERANK
                    else None
                ),
                profile=profile,
            )
            for profile in profiles
        }
        for name, weights in weighted_variants.items():
            if name in retrievers:
                raise DatasetValidationError(f"duplicate retrieval variant {name!r}")
            retrievers[name] = HybridRetriever(
                production_retriever.vector_index,
                ingestion.chunks,
                settings,
                profile=RetrievalProfile.FUSION_DIVERSE,
                fusion_weights=weights,
            )
        variant_configurations = {
            profile.value: _profile_configuration(profile, FusionWeights())
            for profile in profiles
        }
        variant_configurations.update(
            {
                name: _profile_configuration(RetrievalProfile.FUSION_DIVERSE, weights)
                for name, weights in weighted_variants.items()
            }
        )
        report = run_retrieval_ablation(
            cases,
            retrievers,
            RoutingPolicy(settings),
            baseline=arguments.baseline,
            top_k=arguments.top_k,
            repetitions=arguments.repetitions,
            suite_digest=suite.bundle_digest,
            split=arguments.split or "",
            configuration=_configuration(settings),
            variant_configurations=variant_configurations,
            index_build_ms=index_build_ms,
        )
    except Exception as error:
        print(f"消融实验失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        if manager is not None:
            manager.close()

    if arguments.json_output:
        _write(arguments.json_output, report.to_json())
    if arguments.markdown_output:
        _write(arguments.markdown_output, report.to_markdown())
    if not arguments.json_output and not arguments.markdown_output:
        print(report.to_markdown(), end="")
    return 0


def _configuration(settings: Settings) -> dict[str, str | int | float | bool]:
    return {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "reranker_weight": settings.reranker_weight,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "dense_candidates": settings.dense_candidates,
        "sparse_candidates": settings.sparse_candidates,
        "fused_candidates": settings.fused_candidates,
        "final_evidence_count": settings.final_evidence_count,
        "local_confidence_threshold": settings.local_confidence_threshold,
        "hybrid_confidence_ratio": settings.hybrid_confidence_ratio,
        "routing_lexical_saturation": settings.routing_lexical_saturation,
    }


def _profile_configuration(
    profile: RetrievalProfile,
    weights: FusionWeights,
) -> dict[str, str | float | bool]:
    return {
        "profile": profile.value,
        "dense_weight": weights.dense,
        "sparse_weight": weights.sparse,
        "lexical_weight": weights.lexical,
        "rrf_weight": weights.rrf,
        "reranker_enabled": profile is RetrievalProfile.FUSION_DIVERSE_RERANK,
        "source_diversity": profile
        in {
            RetrievalProfile.FUSION_DIVERSE,
            RetrievalProfile.FUSION_DIVERSE_RERANK,
        },
    }


def _fusion_weight(value: str) -> tuple[str, FusionWeights]:
    parts = value.split(":")
    if len(parts) != 5 or not _VARIANT_NAME.fullmatch(parts[0]):
        raise argparse.ArgumentTypeError(
            "fusion weight must be NAME:DENSE:SPARSE:LEXICAL:RRF "
            "with a bounded lowercase name"
        )
    try:
        weights = FusionWeights(*(float(item) for item in parts[1:]))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return parts[0], weights


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
