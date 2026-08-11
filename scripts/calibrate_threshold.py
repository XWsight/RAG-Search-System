"""Calibrate the local-answer threshold from a retrieval benchmark run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.benchmark import load_retrieval_benchmark  # noqa: E402
from rag_system.calibration import ConfidenceSample, calibrate_threshold  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从真实检索基准结果校准本地回答阈值。")
    parser.add_argument("dataset", type=Path, help="检索基准 JSONL")
    parser.add_argument("run_json", type=Path, help="benchmark_retrieval.py 输出的 JSON")
    parser.add_argument("--false-positive-cost", type=float, default=2.0)
    parser.add_argument("--false-negative-cost", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        cases = {case.case_id: case for case in load_retrieval_benchmark(arguments.dataset)}
        payload = json.loads(arguments.run_json.read_text(encoding="utf-8"))
        predictions = payload["predictions"]
        if not isinstance(predictions, list):
            raise ValueError("predictions must be a list")
        samples = []
        seen: set[str] = set()
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ValueError("prediction must be an object")
            case_id = prediction["case_id"]
            if case_id in seen or case_id not in cases:
                raise ValueError("prediction case IDs must be unique and present in the dataset")
            seen.add(case_id)
            samples.append(
                ConfidenceSample(
                    case_id=case_id,
                    confidence=float(prediction["confidence"]),
                    answerable=bool(cases[case_id].relevance),
                )
            )
        if seen != set(cases):
            raise ValueError("predictions must cover every dataset case")
        report = calibrate_threshold(
            samples,
            false_positive_cost=arguments.false_positive_cost,
            false_negative_cost=arguments.false_negative_cost,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"校准失败：{error}", file=sys.stderr)
        return 2

    markdown = report.to_markdown()
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
