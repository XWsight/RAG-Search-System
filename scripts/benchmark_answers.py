"""Run the structured-answer benchmark against the configured chat provider."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.answer_benchmark import load_answer_benchmark, run_answer_benchmark  # noqa: E402
from rag_system.answer_analysis import build_answer_suite_report  # noqa: E402
from rag_system.answer_quality_gate import (  # noqa: E402
    evaluate_answer_quality_gate,
    load_answer_quality_gate,
)
from rag_system.answer_suite import load_answer_suite  # noqa: E402
from rag_system.config import Settings  # noqa: E402
from rag_system.providers import ZhipuChatModel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Answer JSONL dataset or governed answer-suite JSON manifest.",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        required=True,
        help="Explicit evaluation environment file containing provider credentials.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--quality-gate", type=Path)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "test"),
        help="Run one governed-suite split; unavailable for legacy JSONL datasets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite = None
    if args.dataset.suffix.lower() == ".json":
        suite = load_answer_suite(args.dataset)
        cases = (
            suite.cases_for_split(args.split)
            if args.split is not None
            else suite.benchmark_cases
        )
    else:
        if args.split is not None:
            raise SystemExit("--split requires a governed answer-suite JSON manifest")
        cases = load_answer_benchmark(args.dataset)
    if not args.dotenv.is_file():
        raise SystemExit("evaluation environment file does not exist")
    _load_evaluation_environment(args.dotenv)
    settings = Settings().validate()
    model = ZhipuChatModel(settings)
    if not model.available:
        raise SystemExit("chat provider is unavailable in the explicit evaluation environment")
    try:
        report = run_answer_benchmark(cases, model.answer)
    finally:
        model.close()

    rendered_report = (
        build_answer_suite_report(suite, report, split=args.split)
        if suite is not None
        else report
    )
    json_report = rendered_report.to_json()
    markdown_report = rendered_report.to_markdown()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_report, encoding="utf-8")
    else:
        print(json_report, end="")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_report, encoding="utf-8")
    if args.quality_gate:
        gate_result = evaluate_answer_quality_gate(
            report,
            load_answer_quality_gate(args.quality_gate),
        )
        if not gate_result.passed:
            for violation in gate_result.violations:
                print(
                    f"answer quality regression: {violation.metric}="
                    f"{violation.actual:.4f} < {violation.minimum:.4f}",
                    file=sys.stderr,
                )
            return 3
    return 0


def _load_evaluation_environment(path: Path) -> None:
    """Import the runtime-only dotenv dependency when a live run actually starts."""

    from dotenv import load_dotenv

    load_dotenv(path, override=True)


if __name__ == "__main__":
    raise SystemExit(main())
