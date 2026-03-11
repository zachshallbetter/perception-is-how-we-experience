#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CAMPAIGN_ID="${CAMPAIGN_ID:-tuned-core-expand-${TIMESTAMP}}"

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
  echo "'${PYTHON_BIN}' scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed 47 --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_20t_seed47_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/paper_runlocal_tuned_20t_seed47_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed 48 --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_20t_seed48_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/paper_runlocal_tuned_20t_seed48_${TIMESTAMP}.log'"
  echo "echo '[gpu0] worker complete'"
} >> "${gpu0_script}"

gpu1_script="$(make_worker_script "gpu1")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed 49 --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_20t_seed49_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/paper_runlocal_tuned_20t_seed49_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed 50 --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_20t_seed50_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/paper_runlocal_tuned_20t_seed50_${TIMESTAMP}.log'"
  echo "echo '[gpu1] worker complete'"
} >> "${gpu1_script}"

gpu2_script="$(make_worker_script "gpu2")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed 51 --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_20t_seed51_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/paper_runlocal_tuned_20t_seed51_${TIMESTAMP}.log'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed 52 --output_dir '${RESULTS_ROOT}/paper_runlocal_tuned_20t_seed52_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/paper_runlocal_tuned_20t_seed52_${TIMESTAMP}.log'"
  echo "echo '[gpu2] worker complete'"
} >> "${gpu2_script}"

start_tmux_job "${CAMPAIGN_ID}-gpu0" 0 "${gpu0_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu1" 1 "${gpu1_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu2" 2 "${gpu2_script}"
ensure_results_package_watcher

cat <<EOF
Launched tuned-core expansion campaign: ${CAMPAIGN_ID}

Sessions:
  ${CAMPAIGN_ID}-gpu0
  ${CAMPAIGN_ID}-gpu1
  ${CAMPAIGN_ID}-gpu2
  ${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}

Follow-up use:
  Run this if distillation replay does not beat tuned ER, and you want stronger paper confidence intervals around the current main result.
EOF
