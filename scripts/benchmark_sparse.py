"""Run a dependency-free BM25 baseline against the annotated retrieval set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.benchmark import load_retrieval_benchmark, run_retrieval_benchmark  # noqa: E402
from rag_system.benchmark_suite import load_retrieval_suite  # noqa: E402
from rag_system.config import Settings, load_settings  # noqa: E402
from rag_system.domain import SearchHit  # noqa: E402
from rag_system.evaluation import DatasetValidationError  # noqa: E402
from rag_system.ingestion import DocumentIngestor  # noqa: E402
from rag_system.quality_gate import evaluate_quality_gate, load_quality_gate  # noqa: E402
from rag_system.retrieval import RoutingPolicy  # noqa: E402
from rag_system.sparse import BM25Index, SparseDocument  # noqa: E402
from rag_system.text import lexical_relevance  # noqa: E402


class SparseBaselineRetriever:
    """Adapt BM25 chunks to the common retriever contract for baselining."""

    def __init__(self, chunks) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._index = BM25Index(
            SparseDocument(chunk.chunk_id, chunk.text) for chunk in chunks
        )

    def search(self, query: str, *, top_k: int):
        sparse_hits = self._index.search(query, top_k=top_k)
        return tuple(
            SearchHit(
                chunk=self._chunks[hit.document_id],
                score=hit.score / (hit.score + 1.0),
                sparse_rank=rank,
                reasons=("sparse",),
                lexical_score=lexical_relevance(query, self._chunks[hit.document_id].text),
            )
            for rank, hit in enumerate(sparse_hits, start=1)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the dependency-free BM25 baseline; no cloud calls are made."
    )
    parser.add_argument("dataset", type=Path, help="JSONL ground truth or suite JSON manifest")
    parser.add_argument(
        "documents",
        nargs="*",
        type=Path,
        help="JSONL corpus documents; suite manifests resolve their own corpus",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "test"),
        help="run only one source-isolated suite split; valid only for suite JSON",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        help="显式加载指定 dotenv；默认不读取项目 .env，避免基准被本地密钥配置污染",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--quality-gate",
        type=Path,
        help="严格质量门禁 JSON；指标回归时以退出码 3 结束",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    settings = (
        load_settings(dotenv_path=arguments.dotenv)
        if arguments.dotenv is not None
        else Settings().validate()
    )
    if arguments.dataset.suffix.lower() == ".json":
        if arguments.documents:
            print("suite manifest cannot be combined with explicit documents", file=sys.stderr)
            return 2
        try:
            suite = load_retrieval_suite(arguments.dataset)
        except (DatasetValidationError, OSError, ValueError) as error:
            print(f"评测套件加载失败：{error}", file=sys.stderr)
            return 2
        cases = suite.cases_for_split(arguments.split) if arguments.split else suite.cases
        documents = (
            suite.documents_for_split(arguments.split) if arguments.split else suite.documents
        )
    else:
        if arguments.split:
            print("--split is valid only for suite manifests", file=sys.stderr)
            return 2
        if not arguments.documents:
            print("JSONL benchmark requires at least one document", file=sys.stderr)
            return 2
        try:
            cases = load_retrieval_benchmark(arguments.dataset)
        except DatasetValidationError as error:
            print(f"评测数据加载失败：{error}", file=sys.stderr)
            return 2
        documents = tuple(arguments.documents)
    ingestion = DocumentIngestor(settings).ingest(documents, namespace="bm25-baseline")
    run = run_retrieval_benchmark(
        cases,
        SparseBaselineRetriever(ingestion.chunks),
        RoutingPolicy(settings),
        top_k=arguments.top_k,
    )
    if arguments.json_output:
        _write(arguments.json_output, run.to_json())
    if arguments.markdown_output:
        _write(arguments.markdown_output, run.to_markdown())
    if not arguments.json_output and not arguments.markdown_output:
        print(run.to_markdown(), end="")
    if arguments.quality_gate:
        try:
            result = evaluate_quality_gate(run, load_quality_gate(arguments.quality_gate))
        except (DatasetValidationError, TypeError, ValueError) as error:
            print(f"质量门禁加载失败：{error}", file=sys.stderr)
            return 2
        if not result.passed:
            print(run.to_markdown(), file=sys.stderr, end="")
            print(result.to_markdown(), file=sys.stderr, end="")
            return 3
    return 0


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
