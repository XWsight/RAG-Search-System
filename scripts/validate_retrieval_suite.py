"""Validate a retrieval suite and print its auditable coverage matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.benchmark_suite import (  # noqa: E402
    load_retrieval_suite,
    validate_suite_contract,
)
from rag_system.evaluation import DatasetValidationError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate suite structure, coverage, source isolation, and duplicate questions."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, help="frozen suite and corpus contract")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        suite = load_retrieval_suite(arguments.manifest)
        if arguments.contract is not None:
            validate_suite_contract(suite, arguments.contract)
    except (DatasetValidationError, OSError, ValueError) as error:
        print(f"检索评测套件无效：{error}", file=sys.stderr)
        return 2

    json_report = json.dumps(
        suite.summary(), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    markdown_report = suite.to_markdown()
    if arguments.json_output:
        _write(arguments.json_output, json_report)
    if arguments.markdown_output:
        _write(arguments.markdown_output, markdown_report)
    if not arguments.json_output and not arguments.markdown_output:
        print(markdown_report, end="")
    return 0


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
