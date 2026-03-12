#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-20}"
cd "${ROOT_DIR}"
CAPACITY_JSON="$(PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ${PYTHON_BIN} "${ROOT_DIR}/scripts/find_sensor_capacity_match.py")"
MERGED_D_MODEL="$(printf '%s' "${CAPACITY_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["merged_d_model"])')"
MERGED_MLP_DIM="$(( MERGED_D_MODEL * 2 ))"

cat <<EOF
Capacity-matched control setup
  merged_d_model: ${MERGED_D_MODEL}
  merged_mlp_dim: ${MERGED_MLP_DIM}
  batch_size: ${BATCH_SIZE}
  epochs: ${EPOCHS}
EOF

SENSOR_VARIANT="${SENSOR_VARIANT:-baseline}" \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCHS="${EPOCHS}" \
PYTHON_BIN="${PYTHON_BIN}" \
MERGED_D_MODEL="${MERGED_D_MODEL}" \
MERGED_MLP_DIM="${MERGED_MLP_DIM}" \
bash scripts/launch_sensor_mime_ablation_titan.sh
