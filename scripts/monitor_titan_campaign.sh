#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: monitor_titan_campaign.sh <campaign-id> [interval-seconds] [idle-threshold]"
  exit 1
fi

CAMPAIGN_ID="$1"
INTERVAL="${2:-60}"
IDLE_THRESHOLD="${3:-3}"
LOG_DIR="${LOG_DIR:-results/titan_logs}"
LOG_FILE="${LOG_DIR}/${CAMPAIGN_ID}-monitor.log"

mkdir -p "${LOG_DIR}"

idle_streak_0=0
idle_streak_1=0
idle_streak_2=0

echo "[$(date -Iseconds)] monitor-start campaign=${CAMPAIGN_ID} interval=${INTERVAL}s idle_threshold=${IDLE_THRESHOLD}" | tee -a "${LOG_FILE}"

while true; do
  ts="$(date -Iseconds)"

  if ! gpu_rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "[${ts}] error unable-to-query-nvidia-smi" | tee -a "${LOG_FILE}"
    sleep "${INTERVAL}"
    continue
  fi

  line0="$(echo "${gpu_rows}" | sed -n '1p')"
  line1="$(echo "${gpu_rows}" | sed -n '2p')"
  line2="$(echo "${gpu_rows}" | sed -n '3p')"

  util0="$(echo "${line0}" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
  util1="$(echo "${line1}" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
  util2="$(echo "${line2}" | awk -F',' '{gsub(/ /,"",$3); print $3}')"

  mem0="$(echo "${line0}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
  mem1="$(echo "${line1}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
  mem2="$(echo "${line2}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"

  if [ "${util0}" -lt 20 ] && [ "${mem0}" -lt 1000 ]; then idle_streak_0=$((idle_streak_0 + 1)); else idle_streak_0=0; fi
  if [ "${util1}" -lt 20 ] && [ "${mem1}" -lt 1000 ]; then idle_streak_1=$((idle_streak_1 + 1)); else idle_streak_1=0; fi
  if [ "${util2}" -lt 20 ] && [ "${mem2}" -lt 1000 ]; then idle_streak_2=$((idle_streak_2 + 1)); else idle_streak_2=0; fi

  tmux_count="$(tmux ls 2>/dev/null | grep -c "${CAMPAIGN_ID}" || true)"
  echo "[${ts}] gpu0(util=${util0} mem=${mem0}) gpu1(util=${util1} mem=${mem1}) gpu2(util=${util2} mem=${mem2}) tmux_sessions=${tmux_count}" | tee -a "${LOG_FILE}"

  if [ "${idle_streak_0}" -ge "${IDLE_THRESHOLD}" ]; then
    echo "[${ts}] alert gpu0-idle streak=${idle_streak_0}" | tee -a "${LOG_FILE}"
    idle_streak_0=0
  fi
  if [ "${idle_streak_1}" -ge "${IDLE_THRESHOLD}" ]; then
    echo "[${ts}] alert gpu1-idle streak=${idle_streak_1}" | tee -a "${LOG_FILE}"
    idle_streak_1=0
  fi
  if [ "${idle_streak_2}" -ge "${IDLE_THRESHOLD}" ]; then
    echo "[${ts}] alert gpu2-idle streak=${idle_streak_2}" | tee -a "${LOG_FILE}"
    idle_streak_2=0
  fi

  sleep "${INTERVAL}"
done
