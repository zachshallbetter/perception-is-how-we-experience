#!/usr/bin/env python3
"""Render a live Titan experiment dashboard from processes, logs, and results."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_RE = re.compile(r"Task\s+(\d+)/(\d+)", re.IGNORECASE)
EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)", re.IGNORECASE)


@dataclass
class GPUStat:
    index: int
    utilization_gpu: float
    memory_used_mb: int
    memory_total_mb: int
    temperature_c: int


@dataclass
class ActiveRun:
    pid: int
    elapsed_seconds: int
    cpu_percent: float
    mem_percent: float
    script: str
    label: str
    output_dir: str | None
    log_path: str | None
    task_current: int | None
    task_total: int | None
    epoch_current: int | None
    epoch_total: int | None
    progress_fraction: float | None
    eta_seconds: int | None


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def parse_etime(raw: str) -> int:
    raw = raw.strip()
    if "-" in raw:
        days_raw, time_raw = raw.split("-", 1)
        days = int(days_raw)
    else:
        days = 0
        time_raw = raw

    parts = [int(part) for part in time_raw.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours = 0
        minutes = 0
        seconds = parts[0]

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_seconds(value: int | None) -> str:
    if value is None:
        return "-"
    if value < 0:
        return "-"
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def read_log_tail(path: Path, max_lines: int = 400) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.rstrip() for line in text.splitlines()]
    return lines[-max_lines:]


def parse_progress_from_lines(lines: list[str]) -> tuple[int | None, int | None, int | None, int | None]:
    task_current = task_total = epoch_current = epoch_total = None
    for line in lines:
        task_match = TASK_RE.search(line)
        if task_match:
            task_current = int(task_match.group(1))
            task_total = int(task_match.group(2))
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            epoch_current = int(epoch_match.group(1))
            epoch_total = int(epoch_match.group(2))
    return task_current, task_total, epoch_current, epoch_total


def progress_fraction(
    task_current: int | None,
    task_total: int | None,
    epoch_current: int | None,
    epoch_total: int | None,
) -> float | None:
    if not task_current or not task_total:
        return None
    completed_tasks = task_current - 1
    epoch_component = 0.0
    if epoch_current and epoch_total:
        epoch_component = epoch_current / epoch_total
    return min(0.999, max(0.0, (completed_tasks + epoch_component) / task_total))


def infer_log_path(log_dir: Path, output_dir: str | None) -> Path | None:
    if not output_dir:
        return None
    output_name = Path(output_dir).name
    direct = log_dir / f"{output_name}.log"
    if direct.exists():
        return direct
    matches = sorted(log_dir.glob(f"{output_name}*.log"))
    return matches[-1] if matches else None


def parse_active_runs(log_dir: Path) -> list[ActiveRun]:
    ps_output = run_command(
        [
            "ps",
            "-eo",
            "pid=,etime=,pcpu=,pmem=,args=",
        ]
    )
    rows: list[ActiveRun] = []
    for raw_line in ps_output.splitlines():
        parts = raw_line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid_raw, etime_raw, cpu_raw, mem_raw, args_raw = parts
        if "scripts/run_local.py" not in args_raw and "scripts/run_baselines_cl.py" not in args_raw:
            continue
        try:
            argv = shlex.split(args_raw)
        except ValueError:
            continue
        if not argv:
            continue
        executable = Path(argv[0]).name
        if "python" not in executable:
            continue

        script = "run_local.py" if "scripts/run_local.py" in args_raw else "run_baselines_cl.py"
        output_dir = None
        label = script
        if "--output_dir" in argv:
            output_dir = argv[argv.index("--output_dir") + 1]
            label = Path(output_dir).name
        elif "--model" in argv:
            label = f"baseline-{argv[argv.index('--model') + 1]}"

        log_path = infer_log_path(log_dir, output_dir)
        lines = read_log_tail(log_path) if log_path else []
        task_current, task_total, epoch_current, epoch_total = parse_progress_from_lines(lines)
        fraction = progress_fraction(task_current, task_total, epoch_current, epoch_total)
        elapsed_seconds = parse_etime(etime_raw)
        eta_seconds = None
        if fraction and fraction > 0:
            eta_seconds = int((elapsed_seconds / fraction) - elapsed_seconds)

        rows.append(
            ActiveRun(
                pid=int(pid_raw),
                elapsed_seconds=elapsed_seconds,
                cpu_percent=float(cpu_raw),
                mem_percent=float(mem_raw),
                script=script,
                label=label,
                output_dir=output_dir,
                log_path=str(log_path) if log_path else None,
                task_current=task_current,
                task_total=task_total,
                epoch_current=epoch_current,
                epoch_total=epoch_total,
                progress_fraction=fraction,
                eta_seconds=eta_seconds,
            )
        )

    rows.sort(key=lambda row: (row.label, row.pid))
    return rows


def query_gpus() -> list[GPUStat]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[GPUStat] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        rows.append(
            GPUStat(
                index=int(fields[0]),
                utilization_gpu=float(fields[1]),
                memory_used_mb=int(fields[2]),
                memory_total_mb=int(fields[3]),
                temperature_c=int(fields[4]),
            )
        )
    return rows


def load_completed_results(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*/results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cfg = data.get("config", {})
        rows.append(
            {
                "run": path.parent.name,
                "path": str(path.parent),
                "seed": cfg.get("seed"),
                "replay_mode": cfg.get("replay_mode"),
                "replay_weight": cfg.get("replay_weight"),
                "replay_buffer": cfg.get("replay_buffer"),
                "ewc_lambda": cfg.get("ewc_lambda"),
                "si_lambda": cfg.get("si_lambda"),
                "no_parameter_isolation": cfg.get("no_parameter_isolation"),
                "CIL": data.get("CIL"),
                "BWT": data.get("BWT"),
                "FWT": data.get("FWT"),
                "elapsed_seconds": data.get("elapsed_seconds"),
            }
        )
    return rows


def summarize_completed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        run = row["run"]
        if run.startswith("bwt_"):
            by_family["bwt"].append(row)
        elif run.startswith("paper_runlocal_tuned_20t"):
            by_family["paper_core"].append(row)
        elif run.startswith("paper_baseline_perceiver_io_20t"):
            by_family["paper_perceiver_io"].append(row)
        elif run.startswith("paper_baseline_perceiver_20t"):
            by_family["paper_perceiver"].append(row)
        else:
            by_family["other"].append(row)

    summary: dict[str, Any] = {}
    for key, family_rows in by_family.items():
        metrics = [row for row in family_rows if row.get("CIL") is not None]
        if not metrics:
            summary[key] = {"count": len(family_rows)}
            continue
        summary[key] = {
            "count": len(family_rows),
            "cil_mean": sum(row["CIL"] for row in metrics) / len(metrics),
            "bwt_mean": sum(row["BWT"] for row in metrics) / len(metrics),
            "fwt_mean": sum(row["FWT"] for row in metrics) / len(metrics),
            "best_bwt_run": max(metrics, key=lambda row: row["BWT"])["run"],
        }
    return summary


def build_markdown(gpus: list[GPUStat], active_runs: list[ActiveRun], completed_rows: list[dict[str, Any]]) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# Titan Dashboard")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")

    lines.append("## GPUs")
    lines.append("")
    lines.append("| GPU | Util | Memory | Temp |")
    lines.append("| --- | ---: | ---: | ---: |")
    for gpu in gpus:
        lines.append(
            f"| {gpu.index} | {gpu.utilization_gpu:.0f}% | "
            f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MiB | {gpu.temperature_c} C |"
        )
    lines.append("")

    lines.append("## Active Runs")
    lines.append("")
    if not active_runs:
        lines.append("_No active training processes found._")
    else:
        lines.append("| Run | Script | Stage | Elapsed | ETA | CPU | MEM |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for run in active_runs:
            stage = "starting"
            if run.task_current and run.task_total:
                stage = f"T{run.task_current}/{run.task_total}"
                if run.epoch_current and run.epoch_total:
                    stage += f" E{run.epoch_current}/{run.epoch_total}"
            lines.append(
                f"| {run.label} | `{run.script}` | {stage} | "
                f"{format_seconds(run.elapsed_seconds)} | {format_seconds(run.eta_seconds)} | "
                f"{run.cpu_percent:.0f}% | {run.mem_percent:.1f}% |"
            )
    lines.append("")

    bwt_rows = sorted(
        [row for row in completed_rows if row["run"].startswith("bwt_") and row.get("BWT") is not None],
        key=lambda row: row["BWT"],
        reverse=True,
    )
    lines.append("## BWT Leaderboard")
    lines.append("")
    if not bwt_rows:
        lines.append("_No completed BWT runs yet._")
    else:
        lines.append("| Run | CIL | BWT | FWT | Replay | Buffer | EWC | SI | No PI |")
        lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |")
        for row in bwt_rows[:10]:
            lines.append(
                f"| {row['run']} | {row['CIL']:.2f} | {row['BWT']:.2f} | {row['FWT']:.2f} | "
                f"{row['replay_mode']} | {row['replay_buffer']} | {row['ewc_lambda']} | "
                f"{row['si_lambda']} | {row['no_parameter_isolation']} |"
            )
    lines.append("")

    paper_core = [
        row for row in completed_rows if row["run"].startswith("paper_runlocal_tuned_20t") and row.get("BWT") is not None
    ]
    lines.append("## Paper Core Summary")
    lines.append("")
    if not paper_core:
        lines.append("_No completed paper core runs yet._")
    else:
        cil_mean = sum(row["CIL"] for row in paper_core) / len(paper_core)
        bwt_mean = sum(row["BWT"] for row in paper_core) / len(paper_core)
        fwt_mean = sum(row["FWT"] for row in paper_core) / len(paper_core)
        lines.append(f"- runs: `{len(paper_core)}`")
        lines.append(f"- mean CIL: `{cil_mean:.2f}`")
        lines.append(f"- mean BWT: `{bwt_mean:.2f}`")
        lines.append(f"- mean FWT: `{fwt_mean:.2f}`")
    lines.append("")

    recent_rows = sorted(
        [row for row in completed_rows if row.get("elapsed_seconds")],
        key=lambda row: row["path"],
        reverse=True,
    )
    lines.append("## Recent Completed Runs")
    lines.append("")
    if not recent_rows:
        lines.append("_No completed runs yet._")
    else:
        for row in recent_rows[:8]:
            lines.append(
                f"- `{row['run']}`  CIL={row['CIL']:.2f}  BWT={row['BWT']:.2f}  elapsed={format_seconds(int(row['elapsed_seconds']))}"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--log-dir", default="results/titan_logs")
    parser.add_argument("--md-out")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    log_dir = Path(args.log_dir)

    gpus = query_gpus()
    active_runs = parse_active_runs(log_dir)
    completed_rows = load_completed_results(results_root)
    completed_summary = summarize_completed(completed_rows)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": [asdict(gpu) for gpu in gpus],
        "active_runs": [asdict(run) for run in active_runs],
        "completed_summary": completed_summary,
        "completed_runs": completed_rows,
    }
    markdown = build_markdown(gpus, active_runs, completed_rows)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
