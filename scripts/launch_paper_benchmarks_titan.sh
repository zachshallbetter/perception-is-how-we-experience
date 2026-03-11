#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CORE_PYTHON="${CORE_PYTHON:-python}"
BASELINE_PYTHON="${BASELINE_PYTHON:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"

CORE_GPU="${CORE_GPU:-0}"
BASELINE_GPU_1="${BASELINE_GPU_1:-1}"
BASELINE_GPU_2="${BASELINE_GPU_2:-2}"

TASKS="${TASKS:-20}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-4}"

TRAIN_SESSION="${TRAIN_SESSION:-paper-train-${TIMESTAMP}}"
PERCEIVER_SESSION="${PERCEIVER_SESSION:-paper-perceiver-${TIMESTAMP}}"
PERCEIVER_IO_SESSION="${PERCEIVER_IO_SESSION:-paper-perceiver-io-${TIMESTAMP}}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

train_output="${RESULTS_ROOT}/paper_runlocal_tuned_${TASKS}t_${TIMESTAMP}"
perceiver_output="${RESULTS_ROOT}/paper_baseline_perceiver_${TASKS}t_${TIMESTAMP}"
perceiver_io_output="${RESULTS_ROOT}/paper_baseline_perceiver_io_${TASKS}t_${TIMESTAMP}"

train_log="${LOG_DIR}/${TRAIN_SESSION}.log"
perceiver_log="${LOG_DIR}/${PERCEIVER_SESSION}.log"
perceiver_io_log="${LOG_DIR}/${PERCEIVER_IO_SESSION}.log"

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
  local command="$3"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to reuse existing tmux session: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" "cd '${ROOT_DIR}' && export CUDA_VISIBLE_DEVICES='${gpu}' && ${command}"
}

echo "Launching paper benchmark sessions from ${ROOT_DIR}"
echo "  Train session:        ${TRAIN_SESSION}"
echo "  Perceiver session:    ${PERCEIVER_SESSION}"
echo "  Perceiver IO session: ${PERCEIVER_IO_SESSION}"

start_tmux_job \
  "${TRAIN_SESSION}" \
  "${CORE_GPU}" \
  "'${CORE_PYTHON}' scripts/run_local.py --tuned --n_tasks '${TASKS}' --batch_size '${TRAIN_BATCH_SIZE}' --output_dir '${train_output}' 2>&1 | tee '${train_log}'"

start_tmux_job \
  "${PERCEIVER_SESSION}" \
  "${BASELINE_GPU_1}" \
  "'${BASELINE_PYTHON}' scripts/run_baselines_cl.py --model perceiver --n_tasks '${TASKS}' --epochs_per_task '${EPOCHS_PER_TASK}' --batch_size '${BASELINE_BATCH_SIZE}' --output_dir '${perceiver_output}' 2>&1 | tee '${perceiver_log}'"

start_tmux_job \
  "${PERCEIVER_IO_SESSION}" \
  "${BASELINE_GPU_2}" \
  "'${BASELINE_PYTHON}' scripts/run_baselines_cl.py --model perceiver_io --n_tasks '${TASKS}' --epochs_per_task '${EPOCHS_PER_TASK}' --batch_size '${BASELINE_BATCH_SIZE}' --output_dir '${perceiver_io_output}' 2>&1 | tee '${perceiver_io_log}'"

ensure_results_package_watcher

cat <<EOF

Launched detached tmux sessions:
  tmux attach -t ${TRAIN_SESSION}
  tmux attach -t ${PERCEIVER_SESSION}
  tmux attach -t ${PERCEIVER_IO_SESSION}
  tmux attach -t ${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}

Logs:
  ${train_log}
  ${perceiver_log}
  ${perceiver_io_log}

Results:
  ${train_output}
  ${perceiver_output}
  ${perceiver_io_output}
EOF
