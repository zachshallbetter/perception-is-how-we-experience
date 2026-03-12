#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-titan}"
REMOTE_ROOT="${REMOTE_ROOT:-/ai/projects/Perception}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-${ROOT_DIR}/results}"
LOG_DIR="${LOCAL_RESULTS_DIR}/titan_logs"

mkdir -p "${LOCAL_RESULTS_DIR}" "${LOG_DIR}" \
  "${LOCAL_RESULTS_DIR}/real_audio_sequence_summaries" \
  "${LOCAL_RESULTS_DIR}/real_sensor_sequence_summaries"

rsync -az \
  --include 'real_audio_sequence_summaries/' \
  --include 'real_audio_sequence_summaries/*.json' \
  --include 'real_sensor_sequence_summaries/' \
  --include 'real_sensor_sequence_summaries/*.json' \
  --include 'real_audio_*_seed*/' \
  --include 'real_audio_*_seed*/eval_results.json' \
  --include 'real_sensor_*_seed*/' \
  --include 'real_sensor_*_seed*/eval_results.json' \
  --exclude '*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/results/" \
  "${LOCAL_RESULTS_DIR}/"

echo "Synced Titan real-data evidence into ${LOCAL_RESULTS_DIR}"
