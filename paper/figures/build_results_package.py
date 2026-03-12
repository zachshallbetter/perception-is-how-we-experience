#!/usr/bin/env python3
"""Build a paper-ready results package from validated Sensory Perceiver runs."""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "paper" / "figures" / "generated" / "results_package"
SOURCE_MATRIX_DIR = OUTPUT_DIR / "source_matrices"
REAL_DATA_DIR = OUTPUT_DIR / "real_data"
REAL_AUDIO_SUMMARY_DIR = RESULTS_DIR / "real_audio_sequence_summaries"
REAL_SENSOR_SUMMARY_DIR = RESULTS_DIR / "real_sensor_sequence_summaries"

LEGACY_PHASE_1 = {
    "phase": "1",
    "label": "Feature-only replay",
    "source_kind": "paper_canonical_section",
    "source_path": "paper/sections/06-results.md",
    "CIL": 4.86,
    "BWT": -53.80,
    "FWT": 0.00,
    "task0_retention": 0.6,
}

PHASE_2_PATH = RESULTS_DIR / "cl_baseline_reservoir_sampling" / "results.json"
BWT_SWEEP_GLOB = "bwt_*/results.json"
TUNED_RUN_RE = re.compile(r"^paper_runlocal_tuned_20t_seed(?P<seed>\d+)_(?P<family>.+)$")


@dataclass(frozen=True)
class ResultRow:
    run: str
    source_path: Path
    CIL: float
    BWT: float
    FWT: float
    task0_retention: float | None
    replay_mode: str | None = None
    replay_weight: float | None = None
    replay_buffer: int | None = None
    ewc_lambda: float | None = None
    si_lambda: float | None = None
    seed: int | None = None


def discover_tuned_paths() -> tuple[str, list[Path]]:
    families: dict[str, list[Path]] = {}
    for path in RESULTS_DIR.glob("paper_runlocal_tuned_20t_seed*/results.json"):
        match = TUNED_RUN_RE.match(path.parent.name)
        if match is None:
            continue
        family = match.group("family")
        families.setdefault(family, []).append(path)

    if not families:
        raise FileNotFoundError(
            "No tuned-core results found via glob: paper_runlocal_tuned_20t_seed*/results.json"
        )

    best_family = max(
        families.items(),
        key=lambda item: (len(item[1]), item[0]),
    )[0]
    tuned_paths = sorted(
        families[best_family],
        key=lambda path: int(TUNED_RUN_RE.match(path.parent.name).group("seed")),
    )
    return best_family, tuned_paths


def load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required results file is missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def task0_retention_from_result(result: dict) -> float | None:
    matrix = result.get("R_matrix")
    if not matrix:
        return None
    return float(matrix[-1][0])


def row_from_result(path: Path) -> ResultRow:
    result = load_result(path)
    config = result.get("config", {})
    return ResultRow(
        run=path.parent.name,
        source_path=path,
        CIL=float(result["CIL"]),
        BWT=float(result["BWT"]),
        FWT=float(result.get("FWT", 0.0)),
        task0_retention=task0_retention_from_result(result),
        replay_mode=config.get("replay_mode"),
        replay_weight=config.get("replay_weight"),
        replay_buffer=config.get("replay_buffer"),
        ewc_lambda=config.get("ewc_lambda"),
        si_lambda=config.get("si_lambda"),
        seed=config.get("seed"),
    )


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows provided for CSV output: {path}")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_markdown(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content)


def save_text(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_phase_progression_figure(phases: list[dict]) -> None:
    labels = [row["phase_label"] for row in phases]
    cil = [row["CIL"] for row in phases]
    bwt = [row["BWT"] for row in phases]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#C73E1D", "#F18F01", "#2E86AB"]

    axes[0].bar(labels, cil, color=colors)
    axes[0].set_title("CIL Progression")
    axes[0].set_ylabel("CIL (%)")
    axes[0].set_ylim(0, max(cil) + 8)

    axes[1].bar(labels, bwt, color=colors)
    axes[1].set_title("BWT Progression")
    axes[1].set_ylabel("BWT (%)")
    axes[1].axhline(0, color="#444444", linewidth=1, linestyle="--")
    axes[1].set_ylim(min(bwt) - 8, 5)

    for ax, values in zip(axes, [cil, bwt]):
        for idx, value in enumerate(values):
            ax.text(idx, value + (0.8 if value >= 0 else -2.5), f"{value:.2f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Split-CIFAR-100 Results Progression", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure_phase_progression")


def build_tuned_multiseed_figure(tuned_rows: list[ResultRow], tuned_mean: dict) -> None:
    labels = [f"seed {row.seed}" for row in tuned_rows]
    cil = [row.CIL for row in tuned_rows]
    bwt = [row.BWT for row in tuned_rows]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(x, cil, color="#2E86AB", s=80, zorder=3)
    axes[0].axhline(tuned_mean["CIL"], color="#2E86AB", linestyle="--", linewidth=1.5)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("CIL (%)")
    axes[0].set_title("Tuned Core CIL by Seed")

    axes[1].scatter(x, bwt, color="#A23B72", s=80, zorder=3)
    axes[1].axhline(tuned_mean["BWT"], color="#A23B72", linestyle="--", linewidth=1.5)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("BWT (%)")
    axes[1].set_title("Tuned Core BWT by Seed")

    for ax, values in zip(axes, [cil, bwt]):
        for idx, value in enumerate(values):
            ax.text(idx, value + (0.35 if value >= 0 else -0.75), f"{value:.2f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Tuned-Core Multiseed Stability", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure_tuned_core_multiseed")


def build_bwt_sweep_figure(bwt_rows: list[ResultRow], tuned_mean_bwt: float) -> None:
    sorted_rows = sorted(bwt_rows, key=lambda row: row.BWT, reverse=True)
    labels = [row.run.replace("_20260311_002625", "").replace("_20260311_0033_followup", "").replace("_20260311_0030_direct", "") for row in sorted_rows]
    values = [row.BWT for row in sorted_rows]
    colors = ["#6A994E" if value >= tuned_mean_bwt else "#C73E1D" for value in values]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, values, color=colors)
    ax.axvline(tuned_mean_bwt, color="#2E86AB", linestyle="--", linewidth=2, label=f"Tuned core mean ({tuned_mean_bwt:.2f})")
    ax.set_xlabel("BWT (%)")
    ax.set_title("Exploratory BWT Sweep Ranked by Final BWT")
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure_bwt_sweep_ranked")


def build_results_table_figure(table_rows: list[dict]) -> None:
    headers = ["Configuration", "Source", "CIL", "BWT", "Task 0 retention"]
    data = [
        [
            row["configuration"],
            row["source_kind"],
            f"{row['CIL']:.2f}",
            f"{row['BWT']:.2f}",
            f"{row['task0_retention']:.1f}" if row["task0_retention"] is not None else "N/A",
        ]
        for row in table_rows
    ]

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    table = ax.table(cellText=[headers] + data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for col in range(len(headers)):
        table[(0, col)].set_facecolor("#2E86AB")
        table[(0, col)].set_text_props(weight="bold", color="white")

    for row_idx in range(1, len(data) + 1):
        fill = "#EAF3F8" if row_idx == len(data) else ("#F7F7F7" if row_idx % 2 == 0 else "white")
        for col in range(len(headers)):
            table[(row_idx, col)].set_facecolor(fill)
            table[(row_idx, col)].set_edgecolor("#CCCCCC")

    fig.suptitle("Results Overview Table", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "table_results_overview")


def build_distillation_comparison_figure(
    distill_rows: list[ResultRow], control_rows: list[ResultRow]
) -> None:
    labels = [row.run for row in distill_rows + control_rows]
    cil = [row.CIL for row in distill_rows + control_rows]
    bwt = [row.BWT for row in distill_rows + control_rows]
    colors = ["#A23B72"] * len(distill_rows) + ["#2E86AB"] * len(control_rows)
    x = list(range(len(labels)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, cil, color=colors)
    axes[0].set_title("Distillation Campaign CIL")
    axes[0].set_ylabel("CIL (%)")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")

    axes[1].bar(x, bwt, color=colors)
    axes[1].set_title("Distillation Campaign BWT")
    axes[1].set_ylabel("BWT (%)")
    axes[1].axhline(0, color="#444444", linewidth=1, linestyle="--")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Distillation Replay vs Tuned ER Controls", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure_distillation_vs_control")


def copy_source_matrix(result_path: Path, label: str) -> str:
    source_png = result_path.parent / "accuracy_matrix.png"
    if not source_png.exists():
        raise FileNotFoundError(f"Expected accuracy matrix is missing: {source_png}")
    destination = SOURCE_MATRIX_DIR / f"{label}.png"
    shutil.copy2(source_png, destination)
    return str(destination.relative_to(ROOT))


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_completed_audio_entry(report: dict[str, Any] | None, dataset: str) -> dict[str, Any] | None:
    if not report:
        return None
    for entry in report.get("completed", []):
        if entry.get("dataset") == dataset:
            return entry
    return None


def extract_completed_sensor_entry(report: dict[str, Any] | None, dataset: str) -> dict[str, Any] | None:
    if not report:
        return None
    for entry in report.get("completed", []):
        if entry.get("dataset") == dataset:
            return entry
    return None


def real_audio_rows(report: dict[str, Any] | None, *, stage: str, canonical: bool) -> list[dict[str, Any]]:
    if not report:
        return []
    rows: list[dict[str, Any]] = []
    for entry in report.get("completed", []):
        summary = entry.get("summary") or {}
        rows.append(
            {
                "stage": stage,
                "canonical": canonical,
                "dataset": entry.get("dataset"),
                "family": entry.get("family"),
                "runs": summary.get("runs"),
                "mean_accuracy": summary.get("mean_accuracy"),
                "std_accuracy": summary.get("std_accuracy"),
                "min_accuracy": summary.get("min_accuracy"),
                "max_accuracy": summary.get("max_accuracy"),
                "status": report.get("status"),
            }
        )
    return rows


def real_sensor_rows(report: dict[str, Any] | None, *, stage: str, canonical: bool) -> list[dict[str, Any]]:
    if not report:
        return []
    rows: list[dict[str, Any]] = []
    for entry in report.get("completed", []):
        paired = entry.get("paired_comparison") or {}
        rows.append(
            {
                "stage": stage,
                "canonical": canonical,
                "dataset": entry.get("dataset"),
                "family": entry.get("family"),
                "canonical_mean_accuracy": paired.get("canonical_mean_accuracy"),
                "merged_mean_accuracy": paired.get("merged_mean_accuracy"),
                "canonical_minus_merged": paired.get("canonical_minus_merged"),
                "status": report.get("status"),
            }
        )
    return rows


def build_real_data_overview_figure(audio_rows: list[dict[str, Any]], sensor_rows: list[dict[str, Any]]) -> None:
    completed_audio = [row for row in audio_rows if row.get("mean_accuracy") is not None]
    completed_sensor = [row for row in sensor_rows if row.get("canonical_minus_merged") is not None]
    if not completed_audio and not completed_sensor:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    if completed_audio:
        labels = [f"{row['dataset']}\n({row['stage']})" for row in completed_audio]
        values = [row["mean_accuracy"] * 100.0 for row in completed_audio]
        colors = ["#2E86AB" if row["canonical"] else "#A23B72" for row in completed_audio]
        axes[0].bar(labels, values, color=colors)
        axes[0].set_title("Real Audio Accuracy")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(0, 100)
        for idx, value in enumerate(values):
            axes[0].text(idx, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "No completed real-audio runs yet", ha="center", va="center")
        axes[0].set_axis_off()

    if completed_sensor:
        labels = [f"{row['dataset']}\n({row['stage']})" for row in completed_sensor]
        values = [row["canonical_minus_merged"] * 100.0 for row in completed_sensor]
        colors = ["#6A994E" if value >= 0 else "#C73E1D" for value in values]
        axes[1].bar(labels, values, color=colors)
        axes[1].axhline(0, color="#444444", linewidth=1, linestyle="--")
        axes[1].set_title("Real Sensor Routing Gap")
        axes[1].set_ylabel("Canonical - Merged (pp)")
        for idx, value in enumerate(values):
            offset = 1.0 if value >= 0 else -1.0
            axes[1].text(idx, value + offset, f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No completed real-sensor runs yet", ha="center", va="center")
        axes[1].set_axis_off()

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Real-Data Evidence Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure_real_data_evidence")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    phase_2 = row_from_result(PHASE_2_PATH)
    tuned_family, tuned_paths = discover_tuned_paths()
    tuned_rows = [row_from_result(path) for path in tuned_paths]
    bwt_rows = [row_from_result(path) for path in sorted(RESULTS_DIR.glob(BWT_SWEEP_GLOB))]
    if not bwt_rows:
        raise FileNotFoundError(f"No BWT sweep results found via glob: {BWT_SWEEP_GLOB}")

    tuned_mean = {
        "CIL": statistics.mean(row.CIL for row in tuned_rows),
        "BWT": statistics.mean(row.BWT for row in tuned_rows),
        "FWT": statistics.mean(row.FWT for row in tuned_rows),
        "task0_retention": statistics.mean(
            row.task0_retention for row in tuned_rows if row.task0_retention is not None
        ),
        "CIL_std": statistics.pstdev(row.CIL for row in tuned_rows),
        "BWT_std": statistics.pstdev(row.BWT for row in tuned_rows),
    }

    phase_rows = [
        {
            "phase": LEGACY_PHASE_1["phase"],
            "phase_label": "Phase 1",
            "configuration": LEGACY_PHASE_1["label"],
            "source_kind": LEGACY_PHASE_1["source_kind"],
            "source_path": LEGACY_PHASE_1["source_path"],
            "CIL": LEGACY_PHASE_1["CIL"],
            "BWT": LEGACY_PHASE_1["BWT"],
            "FWT": LEGACY_PHASE_1["FWT"],
            "task0_retention": LEGACY_PHASE_1["task0_retention"],
        },
        {
            "phase": "2",
            "phase_label": "Phase 2",
            "configuration": "Raw-input replay + reservoir sampling",
            "source_kind": "artifact_results_json",
            "source_path": str(phase_2.source_path.relative_to(ROOT)),
            "CIL": phase_2.CIL,
            "BWT": phase_2.BWT,
            "FWT": phase_2.FWT,
            "task0_retention": phase_2.task0_retention,
        },
        {
            "phase": "3",
            "phase_label": "Phase 3",
            "configuration": f"Tuned core continual path ({len(tuned_rows)}-seed mean)",
            "source_kind": "artifact_results_json_mean",
            "source_path": "; ".join(str(path.relative_to(ROOT)) for path in tuned_paths),
            "CIL": tuned_mean["CIL"],
            "BWT": tuned_mean["BWT"],
            "FWT": tuned_mean["FWT"],
            "task0_retention": tuned_mean["task0_retention"],
        },
    ]

    tuned_seed_rows = [
        {
            "run": row.run,
            "seed": row.seed,
            "source_path": str(row.source_path.relative_to(ROOT)),
            "CIL": row.CIL,
            "BWT": row.BWT,
            "FWT": row.FWT,
            "task0_retention": row.task0_retention,
        }
        for row in tuned_rows
    ]

    real_audio_seq001_report = load_json_if_exists(REAL_AUDIO_SUMMARY_DIR / "real-audio-seq-001-report.json")
    real_audio_seq002_report = load_json_if_exists(REAL_AUDIO_SUMMARY_DIR / "real-audio-seq-002-report.json")
    real_sensor_seq004_report = load_json_if_exists(REAL_SENSOR_SUMMARY_DIR / "real-seq-004-report.json")
    real_sensor_seq001_report = load_json_if_exists(REAL_SENSOR_SUMMARY_DIR / "real-sensor-seq-001-report.json")

    real_audio_table_rows = (
        real_audio_rows(real_audio_seq001_report, stage="bootstrap", canonical=False)
        + real_audio_rows(real_audio_seq002_report, stage="canonical", canonical=True)
    )
    real_sensor_table_rows = (
        real_sensor_rows(real_sensor_seq004_report, stage="bootstrap", canonical=False)
        + real_sensor_rows(real_sensor_seq001_report, stage="canonical", canonical=True)
    )

    bootstrap_audio_entry = extract_completed_audio_entry(real_audio_seq001_report, "speech_commands")
    bootstrap_sensor_entry = extract_completed_sensor_entry(real_sensor_seq004_report, "wisdm")

    bwt_table_rows = [
        {
            "run": row.run,
            "source_path": str(row.source_path.relative_to(ROOT)),
            "seed": row.seed,
            "replay_mode": row.replay_mode,
            "replay_weight": row.replay_weight,
            "replay_buffer": row.replay_buffer,
            "ewc_lambda": row.ewc_lambda,
            "si_lambda": row.si_lambda,
            "CIL": row.CIL,
            "BWT": row.BWT,
            "FWT": row.FWT,
            "task0_retention": row.task0_retention,
        }
        for row in sorted(bwt_rows, key=lambda item: item.BWT, reverse=True)
    ]

    save_csv(OUTPUT_DIR / "phase_progression.csv", phase_rows)
    save_csv(OUTPUT_DIR / "tuned_core_multiseed.csv", tuned_seed_rows)
    save_csv(OUTPUT_DIR / "bwt_sweep_ranked.csv", bwt_table_rows)
    if real_audio_table_rows:
        save_csv(REAL_DATA_DIR / "real_audio_evidence.csv", real_audio_table_rows)
    if real_sensor_table_rows:
        save_csv(REAL_DATA_DIR / "real_sensor_evidence.csv", real_sensor_table_rows)
    save_json(
        REAL_DATA_DIR / "real_data_status.json",
        {
            "bootstrap_audio_sequence": {
                "sequence_id": "real-audio-seq-001",
                "status": None if real_audio_seq001_report is None else real_audio_seq001_report.get("status"),
                "speech_commands": None if bootstrap_audio_entry is None else bootstrap_audio_entry.get("summary"),
                "notes": "Bootstrap audio evidence; historical report may still list esc50 and urbansound8k as candidates.",
            },
            "canonical_audio_sequence": {
                "sequence_id": "real-audio-seq-002",
                "status": None if real_audio_seq002_report is None else real_audio_seq002_report.get("status"),
                "completed": [] if real_audio_seq002_report is None else real_audio_seq002_report.get("completed", []),
                "blocked": [] if real_audio_seq002_report is None else real_audio_seq002_report.get("blocked", []),
                "expected_integrated_datasets": ["speech_commands", "esc50", "urbansound8k"],
            },
            "bootstrap_sensor_sequence": {
                "sequence_id": "real-seq-004",
                "status": None if real_sensor_seq004_report is None else real_sensor_seq004_report.get("status"),
                "wisdm": None if bootstrap_sensor_entry is None else bootstrap_sensor_entry.get("paired_comparison"),
                "notes": "Bootstrap sensor evidence from the first real WISDM routing sequence.",
            },
            "canonical_sensor_sequence": {
                "sequence_id": "real-sensor-seq-001",
                "status": None if real_sensor_seq001_report is None else real_sensor_seq001_report.get("status"),
                "completed": [] if real_sensor_seq001_report is None else real_sensor_seq001_report.get("completed", []),
                "blocked": [] if real_sensor_seq001_report is None else real_sensor_seq001_report.get("blocked", []),
                "expected_integrated_datasets": [
                    "wisdm",
                    "pamap2",
                    "opportunity",
                    "uci_har",
                    "hhar",
                    "mhealth",
                    "selfback",
                    "geomagnetic_wlan",
                ],
            },
            "blocked_datasets": {
                "audio": {"audioset": "blocked-audio-media-missing"},
                "sensor": {"ubiqlog": "blocked-task-definition-missing"},
            },
        },
    )

    phase_matrix = copy_source_matrix(PHASE_2_PATH, "phase2_raw_input_replay_accuracy_matrix")
    tuned_matrices = [
        copy_source_matrix(path, f"tuned_core_seed{row.seed}_accuracy_matrix")
        for row, path in zip(tuned_rows, tuned_paths)
    ]

    build_phase_progression_figure(phase_rows)
    build_tuned_multiseed_figure(tuned_rows, tuned_mean)
    build_bwt_sweep_figure(bwt_rows, tuned_mean["BWT"])
    build_results_table_figure(phase_rows)
    build_real_data_overview_figure(real_audio_table_rows, real_sensor_table_rows)

    save_markdown(
        OUTPUT_DIR / "README.md",
        "\n".join(
            [
                "# Paper Results Package",
                "",
                "This package is generated from validated Split-CIFAR-100 results used by the canonical paper sections.",
                "",
                "## Sources",
                "",
                "- Phase 1 uses the canonical legacy metrics from `paper/sections/06-results.md` because the original feature-only run directory is not currently tracked as a local results artifact.",
                "- Phase 2 uses `results/cl_baseline_reservoir_sampling/results.json`.",
                f"- Phase 3 uses the discovered tuned-core family `{tuned_family}` with {len(tuned_rows)} synchronized Titan runs.",
                "- The BWT sweep uses the completed `results/bwt_*/results.json` runs synchronized from Titan.",
                "",
                "## Included artifacts",
                "",
                "- `phase_progression.csv`",
                "- `tuned_core_multiseed.csv`",
                "- `bwt_sweep_ranked.csv`",
                "- `table_results_overview.{png,pdf,svg}`",
                "- `figure_phase_progression.{png,pdf,svg}`",
                "- `figure_tuned_core_multiseed.{png,pdf,svg}`",
                "- `figure_bwt_sweep_ranked.{png,pdf,svg}`",
                "- `figure_real_data_evidence.{png,pdf,svg}`",
                "- `package_manifest.json`",
                "- `real_data/real_data_status.json`",
                "- `real_data/real_audio_evidence.csv` (when local reports exist)",
                "- `real_data/real_sensor_evidence.csv` (when local reports exist)",
                "- `source_matrices/*.png`",
                "",
                "## Build",
                "",
                "```bash",
                "python3 paper/figures/build_results_package.py",
                "```",
            ]
        )
        + "\n",
    )

    save_markdown(
        OUTPUT_DIR / "captions.md",
        "\n".join(
            [
                "# Figure And Table Captions",
                "",
                "## Table 1",
                f"Split-CIFAR-100 continual-learning progression. Phase 1 is a legacy feature-only replay result sourced from the canonical paper section because the original run directory is not currently tracked as a local artifact. Phase 2 and Phase 3 are artifact-backed, with Phase 3 reporting the mean over {len(tuned_rows)} tuned-core Titan runs from family `{tuned_family}`.",
                "",
                "## Figure 1",
                "CIL and BWT progression from feature-only replay to raw-input replay with reservoir sampling and then the tuned-core continual configuration. The figure emphasizes that replay-path design, not just stronger regularization, drove the major retention gains.",
                "",
                "## Figure 2",
                "Tuned-core multiseed stability on Titan Node. Each point is one full 20-task Split-CIFAR-100 run, with dashed lines indicating the multiseed means. Variance is modest relative to the mean effect size.",
                "",
                "## Figure 3",
                "Exploratory BWT sweep ranked by final BWT. The dashed reference line marks the tuned-core multiseed mean. None of the completed EWC, SI, larger-buffer ER, or contrastive replay runs surpass the tuned-core baseline.",
                "",
                "## Figure 4",
                "Real-data evidence overview. Bootstrap evidence comes from `real-audio-seq-001` on Speech Commands and `real-seq-004` on WISDM. Canonical evidence is promoted from `real-audio-seq-002` and `real-sensor-seq-001` as those bounded integrated sequences complete. Historical bootstrap reports may still list newly integrated datasets as candidates, so the package treats sequence stage and current supervisor defaults as the canonical source of truth.",
                "",
                "## Source Matrices",
                "The copied accuracy-matrix images provide the underlying per-task retention structure for the reported Phase 2 raw-input replay run and the three tuned-core Titan runs.",
            ]
        )
        + "\n",
    )

    save_text(
        OUTPUT_DIR / "latex_results_package.tex",
        "\n".join(
            [
                "% Auto-generated by paper/figures/build_results_package.py",
                "% Drop-in snippet for the Split-CIFAR-100 results section.",
                "",
                "\\begin{table}[t]",
                "  \\centering",
                f"  \\includegraphics[width=\\linewidth]{{../figures/generated/results_package/table_results_overview.pdf}}",
                f"  \\caption{{Split-CIFAR-100 continual-learning progression. Phase 1 is a legacy feature-only replay result sourced from the canonical paper section because the original run directory is not currently tracked as a local artifact. Phase 2 and Phase 3 are artifact-backed, with Phase 3 reporting the mean over {len(tuned_rows)} tuned-core Titan runs from family `{tuned_family}`.}}",
                "  \\label{tab:split-cifar-results-overview}",
                "\\end{table}",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                "  \\includegraphics[width=\\linewidth]{../figures/generated/results_package/figure_phase_progression.pdf}",
                "  \\caption{CIL and BWT progression from feature-only replay to raw-input replay with reservoir sampling and then the tuned-core continual configuration. The figure emphasizes that replay-path design, not just stronger regularization, drove the major retention gains.}",
                "  \\label{fig:split-cifar-phase-progression}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                "  \\includegraphics[width=\\linewidth]{../figures/generated/results_package/figure_tuned_core_multiseed.pdf}",
                "  \\caption{Tuned-core multiseed stability on Titan Node. Each point is one full 20-task Split-CIFAR-100 run, with dashed lines indicating the multiseed means. Variance is modest relative to the mean effect size.}",
                "  \\label{fig:tuned-core-multiseed}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                "  \\includegraphics[width=\\linewidth]{../figures/generated/results_package/figure_bwt_sweep_ranked.pdf}",
                "  \\caption{Exploratory BWT sweep ranked by final BWT. The dashed reference line marks the tuned-core multiseed mean. None of the completed EWC, SI, larger-buffer ER, or contrastive replay runs surpass the tuned-core baseline.}",
                "  \\label{fig:bwt-sweep-ranked}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                "  \\includegraphics[width=\\linewidth]{../figures/generated/results_package/figure_real_data_evidence.pdf}",
                "  \\caption{Real-data evidence overview. Bootstrap evidence comes from `real-audio-seq-001` on Speech Commands and `real-seq-004` on WISDM. Canonical evidence is promoted from `real-audio-seq-002` and `real-sensor-seq-001` as those bounded integrated sequences complete. Historical bootstrap reports may still list newly integrated datasets as candidates, so the package treats sequence stage and current supervisor defaults as the canonical source of truth.}",
                "  \\label{fig:real-data-evidence-overview}",
                "\\end{figure}",
                "",
            ]
        ),
    )

    distill_paths = sorted(RESULTS_DIR.glob("distill_t*_seed*/results.json"))
    control_paths = sorted(RESULTS_DIR.glob("tuned_er_seed*/results.json"))
    distill_rows = [row_from_result(path) for path in distill_paths]
    control_rows = [row_from_result(path) for path in control_paths]
    if distill_rows:
        save_csv(
            OUTPUT_DIR / "distillation_rows.csv",
            [
                {
                    "run": row.run,
                    "source_path": str(row.source_path.relative_to(ROOT)),
                    "seed": row.seed,
                    "replay_mode": row.replay_mode,
                    "CIL": row.CIL,
                    "BWT": row.BWT,
                    "FWT": row.FWT,
                    "task0_retention": row.task0_retention,
                }
                for row in distill_rows
            ],
        )
    if control_rows:
        save_csv(
            OUTPUT_DIR / "distillation_control_rows.csv",
            [
                {
                    "run": row.run,
                    "source_path": str(row.source_path.relative_to(ROOT)),
                    "seed": row.seed,
                    "replay_mode": row.replay_mode,
                    "CIL": row.CIL,
                    "BWT": row.BWT,
                    "FWT": row.FWT,
                    "task0_retention": row.task0_retention,
                }
                for row in control_rows
            ],
        )
    if distill_rows and control_rows:
        build_distillation_comparison_figure(distill_rows, control_rows)
    if not distill_rows and not control_rows:
        distill_package_status = "pending"
    elif distill_rows and control_rows:
        distill_package_status = "partial-both-present"
    else:
        distill_package_status = "partial-one-side-only"

    distill_status = {
        "completed_distill_runs": [
            {
                "run": row.run,
                "source_path": str(row.source_path.relative_to(ROOT)),
                "seed": row.seed,
                "replay_mode": row.replay_mode,
                "CIL": row.CIL,
                "BWT": row.BWT,
                "FWT": row.FWT,
                "task0_retention": row.task0_retention,
            }
            for row in distill_rows
        ],
        "completed_control_runs": [
            {
                "run": row.run,
                "source_path": str(row.source_path.relative_to(ROOT)),
                "seed": row.seed,
                "replay_mode": row.replay_mode,
                "CIL": row.CIL,
                "BWT": row.BWT,
                "FWT": row.FWT,
                "task0_retention": row.task0_retention,
            }
            for row in control_rows
        ],
        "status": distill_package_status,
        "note": (
            "Distillation replay results will be folded into the package automatically once both "
            "`results/distill_t*_seed*/results.json` and `results/tuned_er_seed*/results.json` exist."
        ),
    }
    save_json(OUTPUT_DIR / "distillation_campaign_status.json", distill_status)

    generated_files = sorted(
        str(path.relative_to(ROOT))
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file()
    )
    generated_files.append(str((OUTPUT_DIR / "package_manifest.json").relative_to(ROOT)))
    manifest = {
        "package_version": 1,
        "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
        "tuned_core_family": tuned_family,
        "phase_progression": phase_rows,
        "tuned_core_summary": tuned_mean,
        "tuned_seed_rows": tuned_seed_rows,
        "bwt_sweep_top_ranked": bwt_table_rows,
        "real_data_status_json": str((REAL_DATA_DIR / "real_data_status.json").relative_to(ROOT)),
        "real_audio_rows": real_audio_table_rows,
        "real_sensor_rows": real_sensor_table_rows,
        "source_matrices": {
            "phase_2": phase_matrix,
            "tuned_core": tuned_matrices,
        },
        "generated_files": generated_files,
    }
    save_json(OUTPUT_DIR / "package_manifest.json", manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
