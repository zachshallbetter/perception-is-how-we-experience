#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
SENSOR_VARIANT="${SENSOR_VARIANT:-corrupt_bindings}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
EPOCHS="${EPOCHS:-20}" \
PYTHON_BIN="${PYTHON_BIN:-python}" \
bash scripts/launch_sensor_mime_ablation_titan.sh
