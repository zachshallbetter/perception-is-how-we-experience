#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-sensor-mime-${TIMESTAMP}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-20}"
SENSOR_VARIANT="${SENSOR_VARIANT:-baseline}"
CANONICAL_D_MODEL="${CANONICAL_D_MODEL:-768}"
MERGED_D_MODEL="${MERGED_D_MODEL:-768}"
CANONICAL_MLP_DIM="${CANONICAL_MLP_DIM:-2048}"
MERGED_MLP_DIM="${MERGED_MLP_DIM:-2048}"
MODEL_DEPTH="${MODEL_DEPTH:-6}"
NUM_HEADS="${NUM_HEADS:-8}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

ensure_results_package_watcher() {
  local watcher_session="${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}"
  local watcher_interval="${RESULTS_PACKAGE_WATCHER_INTERVAL:-60}"

  if tmux has-session -t "${watcher_session}" 2>/dev/null; then
    echo "Results package watcher already running: ${watcher_session}"
    return
  fi

  tmux new-session -d -s "${watcher_session}" \
    "cd '${ROOT_DIR}' && LOG_DIR='${LOG_DIR}' PYTHON_BIN='${PYTHON_BIN}' bash scripts/watch_results_package.sh '${watcher_interval}'"
  echo "Started results package watcher: ${watcher_session}"
}

start_tmux_job() {
  local session="$1"
  local gpu="$2"
  local script_path="$3"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to reuse existing tmux session: ${session}" >&2
    exit 1
  fi
  tmux new-session -d -s "${session}" "cd '${ROOT_DIR}' && export CUDA_VISIBLE_DEVICES='${gpu}' && bash '${script_path}'"
}

make_worker_script() {
  local worker_name="$1"
  local script_path="${ROOT_DIR}/results/${CAMPAIGN_ID}-${worker_name}.sh"
  : > "${script_path}"
  chmod +x "${script_path}"
  echo "${script_path}"
}

gpu0_script="$(make_worker_script "gpu0")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "'${PYTHON_BIN}' -m src.train --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode canonical --sensor_variant '${SENSOR_VARIANT}' --d_model '${CANONICAL_D_MODEL}' --mlp_dim '${CANONICAL_MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed 42 --output_dir '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed42_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed42_${TIMESTAMP}/best_model.pt' --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode canonical --sensor_variant '${SENSOR_VARIANT}' --seed 42 --output_file '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed42_${TIMESTAMP}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed42_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.train --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode canonical --sensor_variant '${SENSOR_VARIANT}' --d_model '${CANONICAL_D_MODEL}' --mlp_dim '${CANONICAL_MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed 43 --output_dir '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed43_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed43_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed43_${TIMESTAMP}/best_model.pt' --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode canonical --sensor_variant '${SENSOR_VARIANT}' --seed 43 --output_file '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed43_${TIMESTAMP}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed43_${TIMESTAMP}.log'"
  echo "echo '[gpu0] worker complete'"
} >> "${gpu0_script}"

gpu1_script="$(make_worker_script "gpu1")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "'${PYTHON_BIN}' -m src.train --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode merged --sensor_variant '${SENSOR_VARIANT}' --d_model '${MERGED_D_MODEL}' --mlp_dim '${MERGED_MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed 42 --output_dir '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_merged_seed42_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed42_${TIMESTAMP}/best_model.pt' --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode merged --sensor_variant '${SENSOR_VARIANT}' --seed 42 --output_file '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed42_${TIMESTAMP}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_merged_seed42_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.train --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode merged --sensor_variant '${SENSOR_VARIANT}' --d_model '${MERGED_D_MODEL}' --mlp_dim '${MERGED_MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed 43 --output_dir '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed43_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_merged_seed43_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed43_${TIMESTAMP}/best_model.pt' --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode merged --sensor_variant '${SENSOR_VARIANT}' --seed 43 --output_file '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed43_${TIMESTAMP}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_merged_seed43_${TIMESTAMP}.log'"
  echo "echo '[gpu1] worker complete'"
} >> "${gpu1_script}"

gpu2_script="$(make_worker_script "gpu2")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "'${PYTHON_BIN}' -m src.train --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode canonical --sensor_variant '${SENSOR_VARIANT}' --d_model '${CANONICAL_D_MODEL}' --mlp_dim '${CANONICAL_MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed 44 --output_dir '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed44_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed44_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed44_${TIMESTAMP}/best_model.pt' --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode canonical --sensor_variant '${SENSOR_VARIANT}' --seed 44 --output_file '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed44_${TIMESTAMP}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_canonical_seed44_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.train --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode merged --sensor_variant '${SENSOR_VARIANT}' --d_model '${MERGED_D_MODEL}' --mlp_dim '${MERGED_MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed 44 --output_dir '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed44_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_merged_seed44_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed44_${TIMESTAMP}/best_model.pt' --experiment sensor_streaming --dataset sensor_streaming --modality_routing_mode merged --sensor_variant '${SENSOR_VARIANT}' --seed 44 --output_file '${RESULTS_ROOT}/sensor_streaming_${SENSOR_VARIANT}_merged_seed44_${TIMESTAMP}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/sensor_streaming_${SENSOR_VARIANT}_merged_seed44_${TIMESTAMP}.log'"
  echo "echo '[gpu2] worker complete'"
} >> "${gpu2_script}"

start_tmux_job "${CAMPAIGN_ID}-gpu0" 0 "${gpu0_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu1" 1 "${gpu1_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu2" 2 "${gpu2_script}"
ensure_results_package_watcher

cat <<EOF
Launched sensor/MIME ablation campaign: ${CAMPAIGN_ID}
Variant:
  ${SENSOR_VARIANT}

Sessions:
  ${CAMPAIGN_ID}-gpu0
  ${CAMPAIGN_ID}-gpu1
  ${CAMPAIGN_ID}-gpu2
  ${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}

This compares:
  canonical = typed modality towers from the MIME registry
  merged    = one generic 20-feature stream without typed modality routing
EOF
