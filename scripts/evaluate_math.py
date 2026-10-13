#!/usr/bin/env python3
"""Evaluate AvoKV-E JSONL generations with MATH-style equivalence checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from avokv.evaluation import extract_answer, math_equal, reference_answer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(rows: list[dict]) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("input contains no rows")

    evaluated = []
    grouped_scores: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        prediction = extract_answer(str(row.get("output", "")))
        reference = reference_answer(row)
        correct = bool(math_equal(prediction, reference, timeout=False))
        key = str(row.get("source_index", row.get("idx", len(grouped_scores))))
        grouped_scores[key].append(correct)
        evaluated.append(
            {
                **row,
                "extracted_answer": prediction,
                "reference_answer": reference,
                "correct": correct,
            }
        )

    all_scores = [score for scores in grouped_scores.values() for score in scores]
    per_problem = [sum(scores) / len(scores) for scores in grouped_scores.values()]
    hit_max = sum(bool(row.get("hit_max_new_tokens", False)) for row in rows)
    metrics = {
        "problems": len(grouped_scores),
        "generations": len(all_scores),
        "correct_generations": sum(all_scores),
        "accuracy": 100.0 * sum(all_scores) / len(all_scores),
        "mean_problem_accuracy": 100.0 * sum(per_problem) / len(per_problem),
        "max_length_hits": hit_max,
    }
    return evaluated, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--details", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics or args.input.with_suffix(".eval.json")
    output_paths = [metrics_path] + ([args.details] if args.details is not None else [])
    resolved = [path.resolve() for path in output_paths]
    if args.input.resolve() in resolved or len(set(resolved)) != len(resolved):
        raise ValueError("input generations, metrics, and details must use different paths")
    evaluated, metrics = evaluate(load_jsonl(args.input))
    print(json.dumps(metrics, indent=2, sort_keys=True))

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with args.details.open("w", encoding="utf-8") as handle:
            for row in evaluated:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
