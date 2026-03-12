#!/usr/bin/env python3
"""Aggregate prepared real-audio runs by dataset."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

RUN_RE = re.compile(
    r"^real_audio_(?P<dataset>[a-z0-9_]+)_seed(?P<seed>\d+)_(?P<family>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def collect_rows(results_root: Path, dataset: str, family: str | None) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_root.glob("real_audio_*/eval_results.json")):
        match = RUN_RE.match(path.parent.name)
        if match is None or match.group("dataset") != dataset:
            continue
        if family is not None and match.group("family") != family:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": path.parent.name,
                "dataset": dataset,
                "family": match.group("family"),
                "seed": int(match.group("seed")),
                "overall_accuracy": float(payload["overall_accuracy"]),
                "n_samples": int(payload["n_samples"]),
                "source_path": str(path),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    accuracies = [row["overall_accuracy"] for row in rows]
    summary = None
    if rows:
        summary = {
            "runs": len(rows),
            "mean_accuracy": statistics.mean(accuracies),
            "std_accuracy": (
                statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
            ),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "seeds": [row["seed"] for row in rows],
        }
    return {"rows": rows, "summary": summary}


def main() -> int:
    args = parse_args()
    payload = summarize(collect_rows(args.results_root, args.dataset, args.family))
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
