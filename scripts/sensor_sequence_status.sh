#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
LOG_FILE="${LOG_FILE:-${RESULTS_ROOT}/titan_logs/sensor-sequence-supervisor.log}"

if [ "$#" -lt 1 ]; then
  echo "Usage: sensor_sequence_status.sh <baseline-family-id>" >&2
  exit 1
fi

FAMILY="$1"

cd "${ROOT_DIR}"

python3 - "$FAMILY" "$RESULTS_ROOT" "$LOG_FILE" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

family = sys.argv[1]
results_root = Path(sys.argv[2])
log_file = Path(sys.argv[3])

statuses = ["baseline", "capacity-match", "drop-smartwatch", "drop-room-env", "drop-robot-arm", "corrupt-bindings", "noise-smartwatch", "noise-room-env", "noise-robot-arm"]

def count_eval_results(name: str) -> int:
    return len(list(results_root.glob(f"sensor_streaming_*_{name}/eval_results.json")))

tmux = subprocess.run(["tmux", "ls"], capture_output=True, text=True, check=False)
session_lines = tmux.stdout.splitlines() if tmux.returncode == 0 else []
active_sessions = [line.split(":", 1)[0] for line in session_lines if f"sensor-mime-{family}" in line]

gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
    check=True,
)

summary_path = results_root / "sensor_sequence_summaries" / f"{family}-autonomy-report.json"
summary = None
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

log_tail = []
if log_file.exists():
    lines = log_file.read_text(encoding="utf-8").splitlines()
    log_tail = [line for line in lines if family in line][-5:]

print(f"sensor-sequence family={family}")
print(f"active_sessions={','.join(active_sessions) if active_sessions else 'none'}")

baseline_results = count_eval_results(family)
print(f"baseline_progress={baseline_results}/6")

for suffix in statuses[1:]:
    branch_family = f"{family}-{suffix}"
    branch_results = count_eval_results(branch_family)
    if branch_results or any(branch_family in line for line in active_sessions):
        print(f"{suffix}_progress={branch_results}/6")

print("gpu_snapshot:")
for row in gpu.stdout.strip().splitlines():
    idx, util, used, total, temp = [part.strip() for part in row.split(",")]
    print(f"  gpu{idx}: util={util}% mem={used}/{total}MiB temp={temp}C")

if summary is not None:
    print(f"autonomy_status={summary.get('status')}")
    variant_gaps = summary.get("variant_gaps", {})
    if variant_gaps:
        for key in sorted(variant_gaps):
            print(f"  gap[{key}]={variant_gaps[key]:.4f}")
else:
    print("autonomy_status=pending")

if log_tail:
    print("recent_supervisor_log:")
    for line in log_tail:
        print(f"  {line}")
PY
