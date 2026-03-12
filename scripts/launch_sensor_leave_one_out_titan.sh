#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-20}"

for variant in drop_smartwatch drop_room_env drop_robot_arm; do
  echo "Launching leave-one-out variant: ${variant}"
  (
    cd "${ROOT_DIR}"
    SENSOR_VARIANT="${variant}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/launch_sensor_mime_ablation_titan.sh
  )
done
