#!/usr/bin/env python3
"""Titan-local bounded supervisor for prepared real-audio experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.real_audio import REAL_AUDIO_DATASET_SPECS, real_audio_dataset_is_prepared


CANDIDATE_DATASETS = {
    "audioset": {
        "source_name": "AudioSet",
        "status": "blocked-audio-media-missing",
        "notes": "Only metadata CSVs are present locally; audio clips have not been acquired into the prepared-audio path.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_id")
    parser.add_argument("--root", default=ROOT_DIR)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--datasets", default="speech_commands,esc50,urbansound8k")
    parser.add_argument("--candidate-datasets", default="audioset")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--prepared-root", default="data/real-audio")
    parser.add_argument("--raw-root", default="data/raw/audio")
    parser.add_argument("--log-file", default="results/titan_logs/real-audio-sequence-supervisor.log")
    parser.add_argument("--summary-dir", default="results/real_audio_sequence_summaries")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--expected-results", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path(args.root).resolve()
        self.python_bin = args.python_bin
        self.datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
        self.candidate_datasets = [item.strip() for item in args.candidate_datasets.split(",") if item.strip()]
        self.results_root = self.root / args.results_root
        self.prepared_root = self.root / args.prepared_root
        self.raw_root = self.root / args.raw_root
        self.poll_seconds = args.poll_seconds
        self.expected_results = args.expected_results
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.log_file = self.root / args.log_file
        self.summary_dir = self.root / args.summary_dir
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.report_file = self.summary_dir / f"{args.sequence_id}-report.json"
        self.sequence_id = args.sequence_id
        self.history: list[dict] = []
        self.active_family: str | None = None

    def log(self, message: str, **fields: object) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "message": message, **fields}
        self.history.append(payload)
        line = json.dumps(payload, sort_keys=True)
        self.log_file.open("a", encoding="utf-8").write(line + "\n")
        print(line, flush=True)

    def run(self, *cmd: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        return subprocess.run(cmd, cwd=self.root, env=merged_env, capture_output=True, text=True, check=True)

    def count_results(self, family: str) -> int:
        pattern = f"real_audio_*_seed*_{family}/eval_results.json"
        return sum(1 for _ in self.results_root.glob(pattern))

    def count_sessions(self, family: str) -> int:
        result = subprocess.run(["tmux", "ls"], cwd=self.root, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return 0
        return sum(1 for line in result.stdout.splitlines() if "real-audio-" in line and family in line)

    def wait_for_family(self, family: str) -> None:
        self.active_family = family
        while True:
            sessions = self.count_sessions(family)
            results = self.count_results(family)
            self.log("wait-family", family=family, sessions=sessions, results=results, expected=self.expected_results)
            self.write_report(status="running", active_family=family, active_results=results, active_sessions=sessions)
            if sessions == 0:
                if results < self.expected_results:
                    raise RuntimeError(f"Family {family} ended with only {results}/{self.expected_results} eval_results.json artifacts")
                self.active_family = None
                return
            time.sleep(self.poll_seconds)

    def maybe_prepare_dataset(self, dataset: str) -> dict:
        prepared = real_audio_dataset_is_prepared(dataset, self.prepared_root)
        raw_dir = self.resolve_raw_dir(dataset)
        if prepared:
            return {"dataset": dataset, "status": "prepared"}
        if dataset in {"speech_commands", "esc50", "urbansound8k"} and raw_dir.exists():
            self.log("prepare-dataset", dataset=dataset, raw_dir=str(raw_dir))
            self.run(
                self.python_bin,
                "scripts/prepare_real_audio_dataset.py",
                "--dataset",
                dataset,
                "--raw-dir",
                str(raw_dir),
                "--output-root",
                str(self.prepared_root),
            )
            if real_audio_dataset_is_prepared(dataset, self.prepared_root):
                return {"dataset": dataset, "status": "prepared"}
            raise RuntimeError(f"Dataset {dataset} preparation ran but prepared files are still missing")
        if raw_dir.exists():
            return {"dataset": dataset, "status": "raw-present-parser-missing", "raw_dir": str(raw_dir)}
        return {"dataset": dataset, "status": "awaiting-raw-data", "raw_dir": str(raw_dir)}

    def resolve_raw_dir(self, dataset: str) -> Path:
        if dataset == "speech_commands":
            candidates = [
                self.raw_root / "speech_commands_hf",
                self.raw_root / "speech_commands_enriched_and_annotated",
                self.raw_root / "speech_commands_enriched",
                self.raw_root / "speech_commands",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        if dataset == "esc50":
            candidates = [self.raw_root / "esc50", self.raw_root / "ESC-50-master"]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        if dataset == "urbansound8k":
            candidates = [self.raw_root / "urbansound8k", self.raw_root / "UrbanSound8K"]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return self.raw_root / dataset

    def summarize_family(self, dataset: str, family: str) -> dict:
        output_file = self.summary_dir / f"{family}.json"
        self.run(
            self.python_bin,
            "scripts/summarize_real_audio_results.py",
            "--results-root",
            str(self.results_root),
            "--dataset",
            dataset,
            "--family",
            family,
            "--output",
            str(output_file),
        )
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        self.log("summary-written", dataset=dataset, family=family, output=str(output_file))
        return {"dataset": dataset, "family": family, "summary_file": str(output_file), "summary": payload.get("summary")}

    def launch_dataset(self, dataset: str, family: str) -> None:
        env = {
            "PYTHON_BIN": self.python_bin,
            "DATASET": dataset,
            "RUN_FAMILY": family,
            "CAMPAIGN_ID": f"real-audio-{dataset}-{family}",
            "BATCH_SIZE": str(self.batch_size),
            "EPOCHS": str(self.epochs),
        }
        self.log("launch", dataset=dataset, family=family, env=env)
        self.run("bash", "scripts/launch_real_audio_routing_titan.sh", env=env)

    def write_report(
        self,
        *,
        status: str,
        completed: list[dict] | None = None,
        blocked: list[dict] | None = None,
        candidate_queue: list[dict] | None = None,
        active_family: str | None = None,
        active_results: int | None = None,
        active_sessions: int | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "sequence_id": self.sequence_id,
            "active_family": active_family,
            "active_results": active_results,
            "active_sessions": active_sessions,
            "completed": completed or [],
            "blocked": blocked or [],
            "candidate_queue": candidate_queue or [],
            "history_tail": self.history[-50:],
        }
        if error is not None:
            payload["error"] = error
        self.report_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def execute(self) -> int:
        self.log("real-audio-supervisor-start", sequence=self.sequence_id, datasets=self.datasets)
        completed: list[dict] = []
        blocked: list[dict] = []

        for dataset in self.datasets:
            prep_state = self.maybe_prepare_dataset(dataset)
            if prep_state["status"] != "prepared":
                blocked.append(prep_state)
                self.log("dataset-blocked", **prep_state)
                continue

            family = f"{self.sequence_id}-{dataset}"
            self.launch_dataset(dataset, family)
            self.wait_for_family(family)
            completed.append(self.summarize_family(dataset, family))

        candidate_queue = []
        for dataset in self.candidate_datasets:
            spec = REAL_AUDIO_DATASET_SPECS.get(dataset)
            info = CANDIDATE_DATASETS.get(
                dataset,
                {
                    "source_name": spec.source_name if spec else dataset,
                    "status": "candidate-not-integrated",
                    "notes": "Tracked audio candidate; parser or prepared-data integration still pending.",
                },
            )
            candidate_queue.append({"dataset": dataset, **info})

        final_status = "completed" if completed else "blocked"
        self.write_report(
            status=final_status,
            completed=completed,
            blocked=blocked,
            candidate_queue=candidate_queue,
            active_family=None,
            active_results=None,
            active_sessions=None,
        )
        self.log("real-audio-supervisor-complete", status=final_status)
        return 0


def main() -> int:
    supervisor = Supervisor(parse_args())
    try:
        return supervisor.execute()
    except Exception as exc:
        supervisor.log("real-audio-supervisor-failed", family=supervisor.active_family, error=str(exc))
        supervisor.write_report(
            status="failed",
            completed=[],
            blocked=[],
            candidate_queue=[],
            active_family=supervisor.active_family,
            active_results=supervisor.count_results(supervisor.active_family) if supervisor.active_family else None,
            active_sessions=supervisor.count_sessions(supervisor.active_family) if supervisor.active_family else None,
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
