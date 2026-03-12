#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
LOG_FILE="${LOG_FILE:-${RESULTS_ROOT}/titan_logs/real-sensor-sequence-supervisor.log}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: real_sensor_sequence_status.sh <sequence-id>" >&2
  exit 2
fi

SEQUENCE_ID="$1"

python3 - <<'PY' "${RESULTS_ROOT}" "${SEQUENCE_ID}" "${LOG_FILE}"
import json
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
sequence_id = sys.argv[2]
log_file = Path(sys.argv[3])
report_path = results_root / "real_sensor_sequence_summaries" / f"{sequence_id}-report.json"

if not report_path.exists():
    print(f"report_missing={report_path}")
    sys.exit(0)

payload = json.loads(report_path.read_text(encoding="utf-8"))
print(f"status={payload.get('status')}")
print(f"sequence_id={payload.get('sequence_id')}")
print(f"active_family={payload.get('active_family')}")
print(f"active_results={payload.get('active_results')}")
print(f"active_sessions={payload.get('active_sessions')}")

completed = payload.get("completed", [])
blocked = payload.get("blocked", [])
candidate_queue = payload.get("candidate_queue", [])
print(f"completed_count={len(completed)}")
for item in completed:
    paired = item.get("paired_comparison") or {}
    gap = paired.get("canonical_minus_merged")
    print(f"completed[{item['dataset']}] family={item['family']} gap={gap}")

print(f"blocked_count={len(blocked)}")
for item in blocked:
    print(f"blocked[{item['dataset']}] status={item['status']}")

print(f"candidate_count={len(candidate_queue)}")
for item in candidate_queue:
    print(f"candidate[{item['dataset']}] status={item['status']}")

if log_file.exists():
    print("recent_supervisor_log:")
    for line in log_file.read_text(encoding="utf-8").splitlines()[-10:]:
        print(line)
PY
