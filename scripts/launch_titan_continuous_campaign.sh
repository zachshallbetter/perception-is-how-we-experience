#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

CORE_PYTHON="${CORE_PYTHON:-python}"
BASELINE_PYTHON="${BASELINE_PYTHON:-python}"

CORE_GPU="${CORE_GPU:-0}"
BASELINE_GPU_1="${BASELINE_GPU_1:-1}"
BASELINE_GPU_2="${BASELINE_GPU_2:-2}"

TASKS="${TASKS:-20}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-4}"

CORE_SEEDS="${CORE_SEEDS:-42,43,44}"
BASELINE_SEEDS="${BASELINE_SEEDS:-42,43,44}"

CAMPAIGN_ID="${CAMPAIGN_ID:-paper-campaign-${TIMESTAMP}}"
SESSION_CORE="${SESSION_CORE:-${CAMPAIGN_ID}-gpu0}"
SESSION_BASELINE_1="${SESSION_BASELINE_1:-${CAMPAIGN_ID}-gpu1}"
SESSION_BASELINE_2="${SESSION_BASELINE_2:-${CAMPAIGN_ID}-gpu2}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tmux

ensure_results_package_watcher() {
  local watcher_session="${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}"
  local watcher_interval="${RESULTS_PACKAGE_WATCHER_INTERVAL:-60}"

  if tmux has-session -t "${watcher_session}" 2>/dev/null; then
    echo "Results package watcher already running: ${watcher_session}"
    return
  fi

  tmux new-session -d -s "${watcher_session}" \
    "cd '${ROOT_DIR}' && LOG_DIR='${LOG_DIR}' PYTHON_BIN='${CORE_PYTHON}' bash scripts/watch_results_package.sh '${watcher_interval}'"
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

queue_core_worker() {
  local script_path
  script_path="$(make_worker_script "gpu0")"
  local run_idx=0
  IFS=',' read -r -a seeds <<< "${CORE_SEEDS}"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '${ROOT_DIR}'"
    for seed in "${seeds[@]}"; do
      run_idx=$((run_idx + 1))
      echo "echo '[gpu0] Starting tuned run seed=${seed}'"
      echo "'${CORE_PYTHON}' scripts/run_local.py --tuned --n_tasks '${TASKS}' --batch_size '${TRAIN_BATCH_SIZE}' --seed '${seed}' --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_${TASKS}t_seed${seed}_${TIMESTAMP}_r${run_idx}' 2>&1 | tee '${LOG_DIR}/${CAMPAIGN_ID}-gpu0-seed${seed}-r${run_idx}.log'"
    done
    echo "echo '[gpu0] Starting replay-depth ablation'"
    echo "'${CORE_PYTHON}' scripts/run_replay_depth_ablation.py --experiment continual --dataset split_cifar --seeds '${CORE_SEEDS}' --epochs '${EPOCHS_PER_TASK}' --device cuda 2>&1 | tee '${LOG_DIR}/${CAMPAIGN_ID}-gpu0-replay-depth.log'"
    echo "echo '[gpu0] Worker complete'"
  } >> "${script_path}"
  echo "${script_path}"
}

queue_baseline_worker() {
  local worker_name="$1"
  local model="$2"
  local script_path
  script_path="$(make_worker_script "${worker_name}")"
  local run_idx=0
  IFS=',' read -r -a seeds <<< "${BASELINE_SEEDS}"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '${ROOT_DIR}'"
    for seed in "${seeds[@]}"; do
      run_idx=$((run_idx + 1))
      echo "echo '[${worker_name}] Starting ${model} run seed=${seed}'"
      echo "'${BASELINE_PYTHON}' scripts/run_baselines_cl.py --model '${model}' --n_tasks '${TASKS}' --epochs_per_task '${EPOCHS_PER_TASK}' --batch_size '${BASELINE_BATCH_SIZE}' --seed '${seed}' --output_dir '${RESULTS_ROOT}/paper_baseline_${model}_${TASKS}t_seed${seed}_${TIMESTAMP}_r${run_idx}' 2>&1 | tee '${LOG_DIR}/${CAMPAIGN_ID}-${worker_name}-${model}-seed${seed}-r${run_idx}.log'"
    done
    echo "echo '[${worker_name}] Worker complete'"
  } >> "${script_path}"
  echo "${script_path}"
}

worker_core_script="$(queue_core_worker)"
worker_gpu1_script="$(queue_baseline_worker "gpu1" "perceiver")"
worker_gpu2_script="$(queue_baseline_worker "gpu2" "perceiver_io")"

start_tmux_job "${SESSION_CORE}" "${CORE_GPU}" "${worker_core_script}"
start_tmux_job "${SESSION_BASELINE_1}" "${BASELINE_GPU_1}" "${worker_gpu1_script}"
start_tmux_job "${SESSION_BASELINE_2}" "${BASELINE_GPU_2}" "${worker_gpu2_script}"
ensure_results_package_watcher

cat <<EOF
Launched continuous campaign: ${CAMPAIGN_ID}

Sessions:
  ${SESSION_CORE}
  ${SESSION_BASELINE_1}
  ${SESSION_BASELINE_2}
  ${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}

Workers:
  ${worker_core_script}
  ${worker_gpu1_script}
  ${worker_gpu2_script}

Monitor:
  tmux ls
  nvidia-smi
  tail -f ${LOG_DIR}/${CAMPAIGN_ID}-gpu0-seed42-r1.log
EOF
