"""Run the real local retrieval pipeline against annotated source-level cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.benchmark import load_retrieval_benchmark, run_retrieval_benchmark  # noqa: E402
from rag_system.config import load_settings  # noqa: E402
from rag_system.index_manager import IndexManager  # noqa: E402
from rag_system.retrieval import ChromaIndexRepository, RoutingPolicy  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用真实本地索引运行检索与路由基准；不会调用云端服务。"
    )
    parser.add_argument("dataset", type=Path, help="检索基准 JSONL")
    parser.add_argument("documents", nargs="+", type=Path, help="组成基准语料库的文档")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manager: IndexManager | None = None
    try:
        settings = load_settings()
        manager = IndexManager(settings, ChromaIndexRepository(settings))
        index_ref = manager.build(arguments.documents)
        cases = load_retrieval_benchmark(arguments.dataset)
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

    if arguments.json_output:
        _write(arguments.json_output, run.to_json())
    if arguments.markdown_output:
        _write(arguments.markdown_output, run.to_markdown())
    if not arguments.json_output and not arguments.markdown_output:
        print(run.to_markdown(), end="")
    return 0


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
