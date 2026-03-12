#!/usr/bin/env python3
"""Aggregate sensor-streaming eval runs by variant and routing mode."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(
    r"^sensor_streaming_(?P<variant>.+)_(?P<routing>canonical|merged)_seed(?P<seed>\d+)_(?P<family>\d{8}_\d{6})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def collect_rows(results_root: Path, family: str | None) -> list[dict]:
    rows = []
    for path in sorted(results_root.glob("sensor_streaming_*/eval_results.json")):
        match = RUN_RE.match(path.parent.name)
        if match is None:
            continue
        if family is not None and match.group("family") != family:
            continue
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append(
            {
                "run": path.parent.name,
                "family": match.group("family"),
                "variant": match.group("variant"),
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
        grouped[(row["variant"], row["routing"])].append(row)

    summary_rows = []
    for (variant, routing), group in sorted(grouped.items()):
        accuracies = [row["overall_accuracy"] for row in group]
        summary_rows.append(
            {
                "variant": variant,
                "routing": routing,
                "runs": len(group),
                "mean_accuracy": statistics.mean(accuracies),
                "std_accuracy": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "seeds": [row["seed"] for row in group],
            }
        )

    paired = []
    by_variant = defaultdict(dict)
    for row in summary_rows:
        by_variant[row["variant"]][row["routing"]] = row
    for variant, routing_rows in sorted(by_variant.items()):
        if {"canonical", "merged"} <= routing_rows.keys():
            paired.append(
                {
                    "variant": variant,
                    "canonical_mean_accuracy": routing_rows["canonical"]["mean_accuracy"],
                    "merged_mean_accuracy": routing_rows["merged"]["mean_accuracy"],
                    "canonical_minus_merged": (
                        routing_rows["canonical"]["mean_accuracy"]
                        - routing_rows["merged"]["mean_accuracy"]
                    ),
                }
            )

    return {"rows": rows, "summary": summary_rows, "paired_comparison": paired}


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.results_root, args.family)
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
