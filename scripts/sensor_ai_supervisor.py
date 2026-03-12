#!/usr/bin/env python3
"""Titan-local bounded supervisor for autonomous sensor-routing experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FamilyResult:
    family: str
    summary_file: str
    paired: list[dict]


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path(args.root).resolve()
        self.results_root = self.root / args.results_root
        self.log_dir = self.root / args.log_dir
        self.summary_dir = self.root / args.summary_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(args.log_file).resolve()
        self.report_file = self.summary_dir / f"{args.current_family}-autonomy-report.json"
        self.python_bin = args.python_bin
        self.current_family = args.current_family
        self.poll_seconds = args.poll_seconds
        self.expected_results = args.expected_results
        self.competitive_gap_threshold = args.competitive_gap_threshold
        self.dominance_gap_threshold = args.dominance_gap_threshold
        self.require_capacity_gate = args.require_capacity_gate
        self.history: list[dict] = []
        self.active_family: str | None = None

    def write_report_payload(self, payload: dict) -> None:
        self.report_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def write_progress_report(
        self,
        *,
        status: str,
        completed: list[FamilyResult],
        baseline_family: FamilyResult | None = None,
        active_family: str | None = None,
        active_results: int | None = None,
        active_sessions: int | None = None,
        error: str | None = None,
    ) -> None:
        variant_gaps = self.summarize_completed(completed)
        payload = {
            "status": status,
            "current_family": self.current_family,
            "active_family": active_family,
            "active_results": active_results,
            "active_sessions": active_sessions,
            "baseline_gap": variant_gaps.get("baseline"),
            "variant_gaps": variant_gaps,
            "competitive_gap_threshold": self.competitive_gap_threshold,
            "dominance_gap_threshold": self.dominance_gap_threshold,
            "require_capacity_gate": self.require_capacity_gate,
            "completed_families": [asdict(item) for item in completed],
            "history_tail": self.history[-50:],
        }
        if baseline_family is not None:
            payload["baseline_family"] = asdict(baseline_family)
        if error is not None:
            payload["error"] = error
        self.write_report_payload(payload)

    def log(self, message: str, **fields: object) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "message": message,
            **fields,
        }
        self.history.append(payload)
        line = json.dumps(payload, sort_keys=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def run(self, *cmd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        return subprocess.run(
            cmd,
            cwd=self.root,
            env=merged_env,
            capture_output=True,
            text=True,
            check=True,
        )

    def count_results(self, family: str) -> int:
        pattern = f"sensor_streaming_*_{family}/eval_results.json"
        return sum(1 for _ in self.results_root.glob(pattern))

    def count_sessions(self, family: str) -> int:
        result = subprocess.run(
            ["tmux", "ls"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return 0
        return sum(1 for line in result.stdout.splitlines() if f"sensor-mime-{family}-gpu" in line)

    def wait_for_family(
        self,
        family: str,
        *,
        completed: list[FamilyResult],
        baseline_family: FamilyResult | None = None,
    ) -> None:
        self.active_family = family
        while True:
            sessions = self.count_sessions(family)
            results = self.count_results(family)
            self.log("wait-family", family=family, sessions=sessions, results=results, expected=self.expected_results)
            self.write_progress_report(
                status="running",
                completed=completed,
                baseline_family=baseline_family,
                active_family=family,
                active_results=results,
                active_sessions=sessions,
            )
            if sessions == 0:
                if results < self.expected_results:
                    raise RuntimeError(
                        f"Family {family} ended with only {results}/{self.expected_results} eval_results.json artifacts"
                    )
                self.active_family = None
                return
            time.sleep(self.poll_seconds)

    def family_complete(self, family: str) -> bool:
        return self.count_results(family) >= self.expected_results and self.count_sessions(family) == 0

    def family_has_activity(self, family: str) -> bool:
        return self.count_results(family) > 0 or self.count_sessions(family) > 0

    def summarize_family(self, family: str) -> FamilyResult:
        output_file = self.summary_dir / f"{family}.json"
        self.run(
            self.python_bin,
            "scripts/summarize_sensor_routing_results.py",
            "--results-root",
            str(self.results_root),
            "--family",
            family,
            "--output",
            str(output_file),
        )
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        self.log("summary-written", family=family, output=str(output_file))
        return FamilyResult(family=family, summary_file=str(output_file), paired=payload.get("paired_comparison", []))

    def baseline_gap(self, family_result: FamilyResult) -> float:
        return self.variant_gap(family_result, "baseline")

    def variant_gap(self, family_result: FamilyResult, variant: str) -> float:
        for row in family_result.paired:
            if row["variant"] == variant:
                return float(row["canonical_minus_merged"])
        raise RuntimeError(
            f"Variant '{variant}' paired comparison missing for family {family_result.family}"
        )

    def launch(self, family: str, launcher: str, *, extra_env: dict[str, str] | None = None) -> None:
        env = {
            "PYTHON_BIN": self.python_bin,
            "CAMPAIGN_ID": f"sensor-mime-{family}",
            "RUN_FAMILY": family,
        }
        if extra_env:
            env.update(extra_env)
        self.log("launch", family=family, launcher=launcher, env=env)
        self.run("bash", launcher, env=env)

    def summarize_completed(self, completed: list[FamilyResult]) -> dict[str, float]:
        summary: dict[str, float] = {}
        for item in completed:
            for row in item.paired:
                key = row["variant"]
                if key == "baseline" and item.family.endswith("capacity-match"):
                    key = "baseline_capacity_match"
                summary[key] = float(row["canonical_minus_merged"])
        return summary

    def write_report(self, status: str, baseline_family: FamilyResult, completed: list[FamilyResult]) -> None:
        self.write_progress_report(
            status=status,
            completed=completed,
            baseline_family=baseline_family,
            active_family=None,
            active_results=None,
            active_sessions=None,
        )
        self.log("report-written", status=status, report=str(self.report_file))

    def determine_stop_state(self, baseline_family: FamilyResult, completed: list[FamilyResult]) -> str:
        baseline_gap = self.baseline_gap(baseline_family)
        if baseline_gap < 0:
            return "merged-favored"

        variant_gaps = self.summarize_completed(completed)
        capacity_gap = variant_gaps.get("baseline_capacity_match")
        if self.require_capacity_gate and capacity_gap is None:
            return "capacity-unverified"
        if capacity_gap is not None and capacity_gap < 0:
            return "capacity-confounded"

        stress_variants = [
            "drop_smartwatch",
            "drop_room_env",
            "drop_robot_arm",
            "corrupt_bindings",
            "noise_smartwatch",
            "noise_room_env",
            "noise_robot_arm",
        ]
        available_stress = [variant_gaps[name] for name in stress_variants if name in variant_gaps]
        supported_stress = sum(gap > 0 for gap in available_stress)

        if (
            baseline_gap >= self.dominance_gap_threshold
            and (capacity_gap is None or capacity_gap >= self.competitive_gap_threshold)
            and available_stress
            and supported_stress >= max(1, len(available_stress) - 1)
        ):
            return "canonical-dominant"
        return "competitive"

    def execute(self) -> int:
        self.log("supervisor-start", family=self.current_family)

        completed: list[FamilyResult] = []
        self.wait_for_family(self.current_family, completed=completed)
        baseline_family = self.summarize_family(self.current_family)
        completed.append(baseline_family)
        baseline_gap = self.baseline_gap(baseline_family)

        if baseline_gap < 0:
            self.log("baseline-merged-favored", gap=baseline_gap)
            status = self.determine_stop_state(baseline_family, completed)
            self.write_report(status, baseline_family, completed)
            self.log("supervisor-complete", status=status)
            return 0

        sequence = []
        if self.require_capacity_gate or baseline_gap >= 0:
            sequence.append(
                (
                    f"{self.current_family}-capacity-match",
                    "scripts/launch_sensor_capacity_matched_control_titan.sh",
                    None,
                )
            )

        sequence.extend(
            [
                (f"{self.current_family}-drop-smartwatch", "scripts/launch_sensor_mime_ablation_titan.sh", {"SENSOR_VARIANT": "drop_smartwatch"}),
                (f"{self.current_family}-drop-room-env", "scripts/launch_sensor_mime_ablation_titan.sh", {"SENSOR_VARIANT": "drop_room_env"}),
                (f"{self.current_family}-drop-robot-arm", "scripts/launch_sensor_mime_ablation_titan.sh", {"SENSOR_VARIANT": "drop_robot_arm"}),
                (f"{self.current_family}-corrupt-bindings", "scripts/launch_sensor_mime_corruption_titan.sh", {"SENSOR_VARIANT": "corrupt_bindings"}),
                (f"{self.current_family}-noise-smartwatch", "scripts/launch_sensor_mime_ablation_titan.sh", {"SENSOR_VARIANT": "noise_smartwatch"}),
                (f"{self.current_family}-noise-room-env", "scripts/launch_sensor_mime_ablation_titan.sh", {"SENSOR_VARIANT": "noise_room_env"}),
                (f"{self.current_family}-noise-robot-arm", "scripts/launch_sensor_mime_ablation_titan.sh", {"SENSOR_VARIANT": "noise_robot_arm"}),
            ]
        )

        active_sequence = [item for item in sequence if self.count_sessions(item[0]) > 0 and not self.family_complete(item[0])]
        pending_sequence = [item for item in sequence if item not in active_sequence]
        ordered_launches = active_sequence + pending_sequence
        if active_sequence:
            self.log(
                "resume-active-families",
                families=[item[0] for item in active_sequence],
            )

        for family, launcher, env in ordered_launches:
            if self.family_complete(family):
                completed.append(self.summarize_family(family))
                continue
            if not self.family_has_activity(family):
                if env is None:
                    self.launch(family, launcher)
                else:
                    self.launch(family, launcher, extra_env=env)
            self.wait_for_family(family, completed=completed, baseline_family=baseline_family)
            completed.append(self.summarize_family(family))

        status = self.determine_stop_state(baseline_family, completed)
        self.write_report(status, baseline_family, completed)
        self.log("supervisor-complete", status=status)
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("current_family")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--log-dir", default="results/titan_logs")
    parser.add_argument("--summary-dir", default="results/sensor_sequence_summaries")
    parser.add_argument("--log-file", default="results/titan_logs/sensor-sequence-supervisor.log")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--expected-results", type=int, default=6)
    parser.add_argument("--competitive-gap-threshold", type=float, default=0.03)
    parser.add_argument("--dominance-gap-threshold", type=float, default=0.08)
    parser.add_argument(
        "--require-capacity-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    supervisor = Supervisor(parse_args())
    try:
        return supervisor.execute()
    except Exception as exc:
        supervisor.log("supervisor-failed", family=supervisor.active_family or supervisor.current_family, error=str(exc))
        supervisor.write_progress_report(
            status="failed",
            completed=[],
            active_family=supervisor.active_family,
            active_results=supervisor.count_results(supervisor.active_family or supervisor.current_family),
            active_sessions=supervisor.count_sessions(supervisor.active_family or supervisor.current_family),
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
