#!/usr/bin/env python3
"""Summarize completed run_local BWT sweep outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prefix", default="bwt_")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = []
    for path in sorted(results_root.glob(f"{args.prefix}*/results.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        rows.append(
            {
                "run": path.parent.name,
                "seed": cfg.get("seed"),
                "replay_mode": cfg.get("replay_mode"),
                "replay_weight": cfg.get("replay_weight"),
                "replay_buffer": cfg.get("replay_buffer"),
                "distillation_temp": cfg.get(
                    "distill_temp", cfg.get("distillation_temp")
                ),
                "ewc_lambda": cfg.get("ewc_lambda"),
                "si_lambda": cfg.get("si_lambda"),
                "no_parameter_isolation": cfg.get("no_parameter_isolation"),
                "CIL": data.get("CIL"),
                "BWT": data.get("BWT"),
                "FWT": data.get("FWT"),
            }
        )

    rows.sort(
        key=lambda row: (row["BWT"] if row["BWT"] is not None else -1e9), reverse=True
    )

    for row in rows:
        print(
            f"{row['run']}: "
            f"CIL={row['CIL']:.2f} "
            f"BWT={row['BWT']:.2f} "
            f"FWT={row['FWT']:.2f} "
            f"mode={row['replay_mode']} "
            f"rw={row['replay_weight']} "
            f"buf={row['replay_buffer']} "
            f"dt={row['distillation_temp']} "
            f"ewc={row['ewc_lambda']} "
            f"si={row['si_lambda']} "
            f"no_pi={row['no_parameter_isolation']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
