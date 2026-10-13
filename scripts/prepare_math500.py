#!/usr/bin/env python3
"""Materialize MATH-500 as the JSONL schema consumed by run_reasoning.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/math500.jsonl"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")

    dataset = load_dataset(
        "HuggingFaceH4/MATH-500",
        split="test",
        revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            record = dict(row)
            record.setdefault("source_index", index)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} examples to {args.output}")


if __name__ == "__main__":
    main()
