"""Run the deterministic offline evaluation and render reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.evaluation import (  # noqa: E402
    DatasetValidationError,
    evaluate_cases,
    load_evaluation_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行完全离线的 RAG 质量评测。")
    parser.add_argument("dataset", type=Path, help="严格 JSONL 格式的评测数据集")
    parser.add_argument("--top-k", type=int, default=5, help="检索指标截断位置，默认 5")
    parser.add_argument("--json-output", type=Path, help="JSON 报告输出路径")
    parser.add_argument("--markdown-output", type=Path, help="Markdown 报告输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        cases = load_evaluation_dataset(arguments.dataset)
        report = evaluate_cases(cases, top_k=arguments.top_k)
    except (DatasetValidationError, TypeError, ValueError) as error:
        print(f"评测失败：{error}", file=sys.stderr)
        return 2

    wrote_report = False
    if arguments.json_output is not None:
        _write(arguments.json_output, report.to_json())
        wrote_report = True
    if arguments.markdown_output is not None:
        _write(arguments.markdown_output, report.to_markdown())
        wrote_report = True
    if not wrote_report:
        print(report.to_markdown(), end="")
    return 0


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
