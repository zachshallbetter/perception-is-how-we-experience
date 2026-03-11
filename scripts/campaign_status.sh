#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: campaign_status.sh <campaign-id>"
  exit 1
fi

CAMPAIGN_ID="$1"
TS="${CAMPAIGN_ID#paper-campaign-}"
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
  pane="$(tmux capture-pane -pt "${session}" | tr -d '\r' | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -n 200)"
  local task epoch
  task="$(echo "${pane}" | grep -Eo "Task [0-9]+/[0-9]+" | tail -n 1 || true)"
  epoch="$(echo "${pane}" | grep -Eo "[Ee]poch [0-9]+/[0-9]+" | tail -n 1 || true)"
  if [ -n "${task}" ] && [ -n "${epoch}" ]; then
    echo "${task}, ${epoch}"
  elif [ -n "${task}" ]; then
    echo "${task}"
  elif echo "${pane}" | grep -q "Worker complete"; then
    echo "complete"
  else
    echo "starting"
  fi
}

count_dirs() {
  local pattern="$1"
  find "${ROOT_DIR}/results" -maxdepth 2 -type d -name "${pattern}" | wc -l | tr -d ' '
}

core_done="$(count_dirs "paper_runlocal_tuned_20t_seed*_${TS}_r*")"
perceiver_done="$(count_dirs "paper_baseline_perceiver_20t_seed*_${TS}_r*")"
perceiver_io_done="$(count_dirs "paper_baseline_perceiver_io_20t_seed*_${TS}_r*")"

echo "Campaign: ${CAMPAIGN_ID}"
echo "Timestamp key: ${TS}"
echo
echo "Sessions:"
for s in "${CAMPAIGN_ID}-gpu0" "${CAMPAIGN_ID}-gpu1" "${CAMPAIGN_ID}-gpu2" "${CAMPAIGN_ID}-monitor"; do
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
echo "  gpu0: $(extract_stage "${CAMPAIGN_ID}-gpu0")"
echo "  gpu1: $(extract_stage "${CAMPAIGN_ID}-gpu1")"
echo "  gpu2: $(extract_stage "${CAMPAIGN_ID}-gpu2")"

echo
echo "Completed outputs (this campaign timestamp):"
echo "  core run_local seeds done:        ${core_done}"
echo "  baseline perceiver seeds done:    ${perceiver_done}"
echo "  baseline perceiver_io seeds done: ${perceiver_io_done}"
