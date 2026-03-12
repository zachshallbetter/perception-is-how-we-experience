#!/usr/bin/env python3
"""Aggregate prepared real-sensor routing runs by dataset and routing mode."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(
    r"^real_sensor_(?P<dataset>[a-z0-9_]+)_(?P<routing>canonical|merged)_seed(?P<seed>\d+)_(?P<family>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def collect_rows(results_root: Path, dataset: str, family: str | None) -> list[dict]:
    rows = []
    for path in sorted(results_root.glob("real_sensor_*/eval_results.json")):
        match = RUN_RE.match(path.parent.name)
        if match is None:
            continue
        if match.group("dataset") != dataset:
            continue
        if family is not None and match.group("family") != family:
            continue
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append(
            {
                "run": path.parent.name,
                "dataset": dataset,
                "family": match.group("family"),
                "routing": match.group("routing"),
                "seed": int(match.group("seed")),
                "overall_accuracy": float(payload["overall_accuracy"]),
                "n_samples": int(payload["n_samples"]),
                "source_path": str(path),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["routing"]].append(row)

    summary_rows = []
    for routing, group in sorted(grouped.items()):
        accuracies = [row["overall_accuracy"] for row in group]
        summary_rows.append(
            {
                "routing": routing,
                "runs": len(group),
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "seeds": [row["seed"] for row in group],
            }
        )

    paired = None
    summary_by_routing = {row["routing"]: row for row in summary_rows}
    if {"canonical", "merged"} <= summary_by_routing.keys():
        paired = {
            "canonical_mean_accuracy": summary_by_routing["canonical"]["mean_accuracy"],
            "merged_mean_accuracy": summary_by_routing["merged"]["mean_accuracy"],
            "canonical_minus_merged": (
                summary_by_routing["canonical"]["mean_accuracy"]
                - summary_by_routing["merged"]["mean_accuracy"]
            ),
        }

    return {"rows": rows, "summary": summary_rows, "paired_comparison": paired}


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.results_root, args.dataset, args.family)
    payload = summarize(rows)
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
