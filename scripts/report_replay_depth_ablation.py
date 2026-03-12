#!/usr/bin/env python3
"""Generate a Markdown report from replay-depth ablation output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_markdown(payload: Dict) -> str:
    lines: List[str] = []
    lines.append("# Replay-Depth Ablation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"- Experiment: `{payload.get('experiment')}`")
    lines.append(f"- Dataset: `{payload.get('dataset')}`")
    lines.append(f"- Seeds: `{payload.get('seeds')}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        "| Condition | Replay Mode | Runs | Acc Mean | Acc Std | Val Acc Mean | Val Acc Std |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload.get("aggregate", []):
        lines.append(
            "| {condition} | {replay_mode} | {n_runs} | {accuracy_mean} | {accuracy_std} | {val_acc_mean} | {val_acc_std} |".format(
                condition=row.get("condition"),
                replay_mode=row.get("replay_mode"),
                n_runs=row.get("n_runs"),
                accuracy_mean=fmt(row.get("accuracy_mean")),
                accuracy_std=fmt(row.get("accuracy_std")),
                val_acc_mean=fmt(row.get("val_acc_mean")),
                val_acc_std=fmt(row.get("val_acc_std")),
            )
        )

    lines.append("")
    lines.append("## Run Records")
    lines.append("")
    lines.append(
        "| Condition | Seed | Train RC | Eval RC | Overall Acc | Final Val Acc | Output Dir |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in payload.get("records", []):
        lines.append(
            "| {condition} | {seed} | {train_returncode} | {eval_returncode} | {overall_accuracy} | {final_val_acc} | `{output_dir}` |".format(
                condition=row.get("condition"),
                seed=row.get("seed"),
                train_returncode=row.get("train_returncode"),
                eval_returncode=row.get("eval_returncode"),
                overall_accuracy=fmt(row.get("overall_accuracy")),
                final_val_acc=fmt(row.get("final_val_acc")),
                output_dir=row.get("output_dir"),
            )
        )

    lines.append("")
    lines.append("## Interpretation Template")
    lines.append("")
    lines.append(
        "1. Compare `feature_only` vs `raw_er` to estimate replay-depth effect through the shared backbone."
    )
    lines.append(
        "2. Compare `raw_er` vs `raw_contrastive` to quantify objective choice impact."
    )
    lines.append(
        "3. Inspect failed runs (`train_rc != 0` or `eval_rc != 0`) before interpreting aggregate means."
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_dir", required=True, help="Directory containing ablation_runs.json"
    )
    parser.add_argument("--output", default="ablation_report.md")
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    runs_json = ablation_dir / "ablation_runs.json"
    if not runs_json.exists():
        raise SystemExit(f"Missing file: {runs_json}")

    payload = json.loads(runs_json.read_text(encoding="utf-8"))
    markdown = build_markdown(payload)
    out = ablation_dir / args.output
    out.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
