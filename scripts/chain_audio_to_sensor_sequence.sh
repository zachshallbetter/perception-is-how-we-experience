#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/ai/envs/perception/bin/python}"
export PYTHON_BIN
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ "$#" -lt 2 ]]; then
  echo "Usage: chain_audio_to_sensor_sequence.sh <audio-sequence-id> <sensor-sequence-id> [sensor supervisor args...]" >&2
  exit 2
fi

AUDIO_SEQUENCE_ID="$1"
SENSOR_SEQUENCE_ID="$2"
shift 2

REPORT_PATH="${ROOT_DIR}/results/real_audio_sequence_summaries/${AUDIO_SEQUENCE_ID}-report.json"
CHAIN_LOG="${ROOT_DIR}/results/titan_logs/real-audio-to-sensor-chain.log"
mkdir -p "$(dirname "${CHAIN_LOG}")"

cd "${ROOT_DIR}"

echo "{\"message\":\"chain-start\",\"audio_sequence\":\"${AUDIO_SEQUENCE_ID}\",\"sensor_sequence\":\"${SENSOR_SEQUENCE_ID}\"}" >> "${CHAIN_LOG}"

while true; do
  if [[ -f "${REPORT_PATH}" ]]; then
    STATUS="$("${PYTHON_BIN}" - <<'PY' "${REPORT_PATH}"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status", "missing"))
PY
)"
    case "${STATUS}" in
      completed)
        echo "{\"message\":\"chain-launch\",\"audio_sequence\":\"${AUDIO_SEQUENCE_ID}\",\"sensor_sequence\":\"${SENSOR_SEQUENCE_ID}\"}" >> "${CHAIN_LOG}"
        exec "${PYTHON_BIN}" scripts/real_sensor_ai_supervisor.py "${SENSOR_SEQUENCE_ID}" "$@"
        ;;
      blocked|failed)
        echo "{\"message\":\"chain-abort\",\"audio_sequence\":\"${AUDIO_SEQUENCE_ID}\",\"status\":\"${STATUS}\"}" >> "${CHAIN_LOG}"
        exit 1
        ;;
    esac
  fi
  sleep "${POLL_SECONDS}"
done
