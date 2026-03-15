#!/usr/bin/env python3
"""Aggregate eval-only real-sensor distinction probes against baseline sequence runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROBE_RE = re.compile(
    r"^real_sensor_probe_(?P<dataset>[a-z0-9_]+)_(?P<routing>canonical|merged)_(?P<variant>[a-z_]+)_seed(?P<seed>\d+)_(?P<campaign>.+)$"
)
BASELINE_RE = re.compile(
    r"^real_sensor_(?P<dataset>[a-z0-9_]+)_(?P<routing>canonical|merged)_seed(?P<seed>\d+)_(?P<family>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--campaign", type=str, required=True)
    parser.add_argument("--source-sequence-id", type=str, default="real-sensor-seq-002")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_probe_rows(results_root: Path, campaign: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_root.glob("real_sensor_probe_*/eval_results.json")):
        match = PROBE_RE.match(path.parent.name)
        if match is None or match.group("campaign") != campaign:
            continue
        payload = _load_json(path)
        rows.append(
            {
                "run": path.parent.name,
                "dataset": match.group("dataset"),
                "routing": match.group("routing"),
                "variant": match.group("variant"),
                "seed": int(match.group("seed")),
                "campaign": campaign,
                "overall_accuracy": float(payload["overall_accuracy"]),
                "n_samples": int(payload["n_samples"]),
                "source_path": str(path),
            }
        )
    return rows


def collect_baseline_rows(results_root: Path, source_sequence_id: str) -> dict[tuple[str, str, int], float]:
    rows: dict[tuple[str, str, int], float] = {}
    for path in sorted(results_root.glob("real_sensor_*/eval_results.json")):
        match = BASELINE_RE.match(path.parent.name)
        if match is None:
            continue
        dataset = match.group("dataset")
        family = match.group("family")
        expected_family = f"{source_sequence_id}-{dataset}"
        if family != expected_family:
            continue
        payload = _load_json(path)
        rows[(dataset, match.group("routing"), int(match.group("seed")))] = float(
            payload["overall_accuracy"]
        )
    return rows


def summarize(rows: list[dict], baseline_rows: dict[tuple[str, str, int], float]) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["routing"], row["variant"])
        baseline = baseline_rows.get((row["dataset"], row["routing"], row["seed"]))
        row["baseline_accuracy"] = baseline
        row["delta_vs_baseline"] = (
            row["overall_accuracy"] - baseline if baseline is not None else None
        )
        grouped[key].append(row)

    summary_rows = []
    by_dataset_variant: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (dataset, routing, variant), group in sorted(grouped.items()):
        accuracies = [row["overall_accuracy"] for row in group]
        baseline_accuracies = [
            row["baseline_accuracy"] for row in group if row["baseline_accuracy"] is not None
        ]
        deltas = [row["delta_vs_baseline"] for row in group if row["delta_vs_baseline"] is not None]
        summary = {
            "dataset": dataset,
            "routing": routing,
            "variant": variant,
            "runs": len(group),
            "mean_accuracy": statistics.mean(accuracies),
            "std_accuracy": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
            "mean_baseline_accuracy": (
                statistics.mean(baseline_accuracies) if baseline_accuracies else None
            ),
            "mean_delta_vs_baseline": (
                statistics.mean(deltas) if deltas else None
            ),
            "seeds": [row["seed"] for row in group],
        }
        summary_rows.append(summary)
        by_dataset_variant[(dataset, variant)][routing] = summary

    paired_rows = []
    for (dataset, variant), routing_rows in sorted(by_dataset_variant.items()):
        if {"canonical", "merged"} <= routing_rows.keys():
            paired_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "canonical_mean_accuracy": routing_rows["canonical"]["mean_accuracy"],
                    "merged_mean_accuracy": routing_rows["merged"]["mean_accuracy"],
                    "canonical_minus_merged": (
                        routing_rows["canonical"]["mean_accuracy"]
                        - routing_rows["merged"]["mean_accuracy"]
                    ),
                    "canonical_mean_delta_vs_baseline": routing_rows["canonical"][
                        "mean_delta_vs_baseline"
                    ],
                    "merged_mean_delta_vs_baseline": routing_rows["merged"][
                        "mean_delta_vs_baseline"
                    ],
                }
            )

    return {"rows": rows, "summary": summary_rows, "paired_comparison": paired_rows}


def main() -> int:
    args = parse_args()
    rows = collect_probe_rows(args.results_root, args.campaign)
    baseline_rows = collect_baseline_rows(args.results_root, args.source_sequence_id)
    payload = summarize(rows, baseline_rows)
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
