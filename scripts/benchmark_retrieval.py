"""Run the real local retrieval pipeline against annotated source-level cases."""

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
from rag_system.evaluation import DatasetValidationError  # noqa: E402
from rag_system.index_manager import IndexManager  # noqa: E402
from rag_system.quality_gate import evaluate_quality_gate, load_quality_gate  # noqa: E402
from rag_system.retrieval import ChromaIndexRepository, RoutingPolicy  # noqa: E402
from rag_system.retrieval_analysis import build_retrieval_suite_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用真实本地索引运行检索与路由基准；不会调用云端服务。"
    )
    parser.add_argument("dataset", type=Path, help="检索基准 JSONL 或 suite JSON")
    parser.add_argument(
        "documents",
        nargs="*",
        type=Path,
        help="JSONL 基准语料；suite manifest 会自行解析 corpus",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "test"),
        help="只运行 suite 中一个按来源隔离的数据分段",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        help="显式加载指定 dotenv；默认不读取项目 .env",
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
    manager: IndexManager | None = None
    suite = None
    try:
        settings = (
            load_settings(dotenv_path=arguments.dotenv)
            if arguments.dotenv is not None
            else Settings().validate()
        )
        if arguments.dataset.suffix.lower() == ".json":
            if arguments.documents:
                raise DatasetValidationError(
                    "suite manifest cannot be combined with explicit documents"
                )
            suite = load_retrieval_suite(arguments.dataset)
            cases = suite.cases_for_split(arguments.split) if arguments.split else suite.cases
            documents = (
                suite.documents_for_split(arguments.split)
                if arguments.split
                else suite.documents
            )
        else:
            if arguments.split:
                raise DatasetValidationError("--split is valid only for suite manifests")
            if not arguments.documents:
                raise DatasetValidationError(
                    "JSONL benchmark requires at least one document"
                )
            cases = load_retrieval_benchmark(arguments.dataset)
            documents = tuple(arguments.documents)
        manager = IndexManager(settings, ChromaIndexRepository(settings))
        index_ref = manager.build(documents)
        run = run_retrieval_benchmark(
            cases,
            manager.get(index_ref.index_id),
            RoutingPolicy(settings),
            top_k=arguments.top_k,
        )
    except Exception as error:
        print(f"基准运行失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        if manager is not None:
            manager.close()

    rendered_run = (
        build_retrieval_suite_report(suite, run, split=arguments.split)
        if suite is not None
        else run
    )
    if arguments.json_output:
        _write(arguments.json_output, rendered_run.to_json())
    if arguments.markdown_output:
        _write(arguments.markdown_output, rendered_run.to_markdown())
    if not arguments.json_output and not arguments.markdown_output:
        print(rendered_run.to_markdown(), end="")
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
