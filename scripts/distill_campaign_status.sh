#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: distill_campaign_status.sh <campaign-id>"
  exit 1
fi

CAMPAIGN_ID="$1"
TS="${CAMPAIGN_ID#distill-campaign-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

extract_stage() {
  local session="$1"
  if ! session_exists "${session}"; then
    echo "not-running"
    return
  fi
  local pane
  pane="$(tmux capture-pane -pt "${session}" | tr -d '\r' | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -n 250)"
  local task epoch
  task="$(echo "${pane}" | grep -Eo "Task [0-9]+/[0-9]+" | tail -n 1 || true)"
  epoch="$(echo "${pane}" | grep -Eo "[Ee]poch [0-9]+/[0-9]+" | tail -n 1 || true)"
  if [ -n "${task}" ] && [ -n "${epoch}" ]; then
    echo "${task}, ${epoch}"
  elif [ -n "${task}" ]; then
    echo "${task}"
  elif echo "${pane}" | grep -q "worker complete"; then
    echo "complete"
  else
    echo "starting"
  fi
}

count_results() {
  local pattern="$1"
  find "${ROOT_DIR}/results" -maxdepth 2 -type f -path "${ROOT_DIR}/results/${pattern}/results.json" | wc -l | tr -d ' '
}

echo "Campaign: ${CAMPAIGN_ID}"
echo "Timestamp key: ${TS}"
echo
echo "Sessions:"
for s in "${CAMPAIGN_ID}-gpu0" "${CAMPAIGN_ID}-gpu1" "${CAMPAIGN_ID}-gpu2" "results-package-watcher"; do
  if session_exists "${s}"; then
    echo "  [up]   ${s}"
  else
    echo "  [down] ${s}"
  fi
done

echo
echo "GPU utilization:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo
echo "Worker stage:"
echo "  gpu0 (distill T=2.0): $(extract_stage "${CAMPAIGN_ID}-gpu0")"
echo "  gpu1 (distill T=4.0): $(extract_stage "${CAMPAIGN_ID}-gpu1")"
echo "  gpu2 (tuned ER ctrl): $(extract_stage "${CAMPAIGN_ID}-gpu2")"

echo
echo "Completed outputs (this campaign timestamp):"
echo "  distill T=2.0 runs done: $(count_results "distill_t2_seed*_${TS}")"
echo "  distill T=4.0 runs done: $(count_results "distill_t4_seed*_${TS}")"
echo "  tuned ER control runs done: $(count_results "tuned_er_seed*_${TS}")"

echo
echo "Watcher package status:"
if [ -f "${ROOT_DIR}/paper/figures/generated/results_package/distillation_campaign_status.json" ]; then
  python3 - <<'PY' "${ROOT_DIR}/paper/figures/generated/results_package/distillation_campaign_status.json"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(f"  status: {data.get('status')}")
print(f"  completed distill runs: {len(data.get('completed_distill_runs', []))}")
print(f"  completed control runs: {len(data.get('completed_control_runs', []))}")
PY
else
  echo "  status: no-package-yet"
fi
