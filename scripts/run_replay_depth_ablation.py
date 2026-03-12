#!/usr/bin/env python3
"""Run replay-depth ablations and emit machine-readable summaries.

Ablation conditions:
- feature_only: classifier-level replay only (no replay gradient through backbone)
- raw_er: raw-input replay with cross-entropy objective
- raw_contrastive: raw-input replay with NT-Xent objective
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

CONDITIONS = {
    "feature_only": "feature",
    "raw_er": "er",
    "raw_contrastive": "contrastive",
}


@dataclass
class RunRecord:
    condition: str
    replay_mode: str
    seed: int
    output_dir: str
    train_returncode: int
    eval_returncode: int
    duration_sec: float
    overall_accuracy: float | None
    n_samples: int | None
    final_val_acc: float | None


def parse_seed_list(raw: str) -> List[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("At least one seed is required")
    return out


def read_eval_results(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_final_val_acc(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    val = rows[-1].get("val_acc")
    return float(val) if val not in {None, ""} else None


def run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True)


def aggregate(records: List[RunRecord]) -> List[Dict]:
    by_condition: Dict[str, List[RunRecord]] = {}
    for rec in records:
        by_condition.setdefault(rec.condition, []).append(rec)

    out = []
    for condition, rows in sorted(by_condition.items()):
        accs = [r.overall_accuracy for r in rows if r.overall_accuracy is not None]
        val_accs = [r.final_val_acc for r in rows if r.final_val_acc is not None]
        out.append(
            {
                "condition": condition,
                "replay_mode": rows[0].replay_mode,
                "n_runs": len(rows),
                "accuracy_mean": mean(accs) if accs else None,
                "accuracy_std": (
                    pstdev(accs) if len(accs) > 1 else 0.0 if accs else None
                ),
                "val_acc_mean": mean(val_accs) if val_accs else None,
                "val_acc_std": (
                    pstdev(val_accs) if len(val_accs) > 1 else 0.0 if val_accs else None
                ),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="continual")
    parser.add_argument("--dataset", default="split_cifar")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="results/replay_depth_ablation")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / f"run_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    records: List[RunRecord] = []

    print(f"[ablation] output_root={out_root}")
    print(f"[ablation] seeds={seeds}")

    for condition, replay_mode in CONDITIONS.items():
        for seed in seeds:
            run_dir = out_root / f"{condition}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            train_cmd = [
                sys.executable,
                "-m",
                "src.train",
                "--experiment",
                args.experiment,
                "--dataset",
                args.dataset,
                "--replay_mode",
                replay_mode,
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--d_model",
                str(args.d_model),
                "--num_heads",
                str(args.num_heads),
                "--mlp_dim",
                str(args.mlp_dim),
                "--depth",
                str(args.depth),
                "--learning_rate",
                str(args.learning_rate),
                "--output_dir",
                str(run_dir),
            ]

            t0 = time.perf_counter()
            print("[ablation] train:", " ".join(train_cmd))
            train_res = run_cmd(train_cmd, cwd=Path(__file__).resolve().parents[1])

            eval_rc = 0
            if not args.skip_eval and train_res.returncode == 0:
                ckpt = run_dir / "best_model.pt"
                eval_out = run_dir / "eval_results.json"
                eval_cmd = [
                    sys.executable,
                    "-m",
                    "src.eval",
                    "--checkpoint",
                    str(ckpt),
                    "--experiment",
                    args.experiment,
                    "--dataset",
                    args.dataset,
                    "--batch_size",
                    str(args.batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--output_file",
                    str(eval_out),
                ]
                print("[ablation] eval:", " ".join(eval_cmd))
                eval_res = run_cmd(eval_cmd, cwd=Path(__file__).resolve().parents[1])
                eval_rc = eval_res.returncode
            elif train_res.returncode != 0:
                eval_rc = -1

            duration_sec = time.perf_counter() - t0
            eval_data = read_eval_results(run_dir / "eval_results.json")
            final_val_acc = read_final_val_acc(run_dir / "train_log.csv")

            rec = RunRecord(
                condition=condition,
                replay_mode=replay_mode,
                seed=seed,
                output_dir=str(run_dir),
                train_returncode=train_res.returncode,
                eval_returncode=eval_rc,
                duration_sec=duration_sec,
                overall_accuracy=eval_data.get("overall_accuracy"),
                n_samples=eval_data.get("n_samples"),
                final_val_acc=final_val_acc,
            )
            records.append(rec)
            print(
                f"[ablation] done condition={condition} seed={seed} train_rc={train_res.returncode} eval_rc={eval_rc}"
            )

    aggregate_rows = aggregate(records)

    runs_json = out_root / "ablation_runs.json"
    runs_json.write_text(
        json.dumps(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "experiment": args.experiment,
                "dataset": args.dataset,
                "seeds": seeds,
                "records": [asdict(r) for r in records],
                "aggregate": aggregate_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(out_root / "ablation_runs.csv", [asdict(r) for r in records])
    write_csv(out_root / "ablation_summary.csv", aggregate_rows)

    print(f"[ablation] wrote {runs_json}")
    print(f"[ablation] wrote {out_root / 'ablation_runs.csv'}")
    print(f"[ablation] wrote {out_root / 'ablation_summary.csv'}")

    failed = [
        r
        for r in records
        if r.train_returncode != 0 or (not args.skip_eval and r.eval_returncode != 0)
    ]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
