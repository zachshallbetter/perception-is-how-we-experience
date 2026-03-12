#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:?DATASET is required}"
CAMPAIGN_ID="${CAMPAIGN_ID:-real-audio-${DATASET}-${TIMESTAMP}}"
RUN_FAMILY="${RUN_FAMILY:-${CAMPAIGN_ID#real-audio-${DATASET}-}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
D_MODEL="${D_MODEL:-768}"
MLP_DIM="${MLP_DIM:-2048}"
MODEL_DEPTH="${MODEL_DEPTH:-6}"
NUM_HEADS="${NUM_HEADS:-8}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

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

write_worker() {
  local script_path="$1"
  local seed="$2"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '${ROOT_DIR}'"
    echo "'${PYTHON_BIN}' -m src.train --experiment multimodal --dataset '${DATASET}' --d_model '${D_MODEL}' --mlp_dim '${MLP_DIM}' --depth '${MODEL_DEPTH}' --num_heads '${NUM_HEADS}' --batch_size '${BATCH_SIZE}' --epochs '${EPOCHS}' --seed '${seed}' --output_dir '${RESULTS_ROOT}/real_audio_${DATASET}_seed${seed}_${RUN_FAMILY}' 2>&1 | tee '${LOG_DIR}/real_audio_${DATASET}_seed${seed}_${RUN_FAMILY}.log'"
    echo "'${PYTHON_BIN}' -m src.eval --checkpoint '${RESULTS_ROOT}/real_audio_${DATASET}_seed${seed}_${RUN_FAMILY}/best_model.pt' --experiment multimodal --dataset '${DATASET}' --seed '${seed}' --output_file '${RESULTS_ROOT}/real_audio_${DATASET}_seed${seed}_${RUN_FAMILY}/eval_results.json' 2>&1 | tee -a '${LOG_DIR}/real_audio_${DATASET}_seed${seed}_${RUN_FAMILY}.log'"
    echo "echo '[worker] complete seed=${seed}'"
  } >> "${script_path}"
}

gpu0_script="$(make_worker_script "gpu0")"
gpu1_script="$(make_worker_script "gpu1")"
gpu2_script="$(make_worker_script "gpu2")"
write_worker "${gpu0_script}" 42
write_worker "${gpu1_script}" 43
write_worker "${gpu2_script}" 44

start_tmux_job "${CAMPAIGN_ID}-gpu0" 0 "${gpu0_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu1" 1 "${gpu1_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu2" 2 "${gpu2_script}"

cat <<EOF
Launched real-audio campaign: ${CAMPAIGN_ID}
Dataset:
  ${DATASET}
Run family:
  ${RUN_FAMILY}

Sessions:
  ${CAMPAIGN_ID}-gpu0
  ${CAMPAIGN_ID}-gpu1
  ${CAMPAIGN_ID}-gpu2
EOF
