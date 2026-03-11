#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CAMPAIGN_ID="${CAMPAIGN_ID:-distill-campaign-${TIMESTAMP}}"
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
  echo "echo '[gpu0] starting distill_t2_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode distill --replay_weight 0.8 --distill_temp 2.0 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/distill_t2_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/distill_t2_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu0] starting distill_t2_seed43'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 43 --replay_mode distill --replay_weight 0.8 --distill_temp 2.0 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/distill_t2_seed43_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/distill_t2_seed43_${TIMESTAMP}.log'"
  echo "echo '[gpu0] worker complete'"
} >> "${gpu0_script}"

gpu1_script="$(make_worker_script "gpu1")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "echo '[gpu1] starting distill_t4_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode distill --replay_weight 0.8 --distill_temp 4.0 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/distill_t4_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/distill_t4_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu1] starting distill_t4_seed43'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 43 --replay_mode distill --replay_weight 0.8 --distill_temp 4.0 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/distill_t4_seed43_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/distill_t4_seed43_${TIMESTAMP}.log'"
  echo "echo '[gpu1] worker complete'"
} >> "${gpu1_script}"

gpu2_script="$(make_worker_script "gpu2")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "echo '[gpu2] starting tuned_er_seed45'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 45 --replay_mode er --replay_weight 0.8 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/tuned_er_seed45_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/tuned_er_seed45_${TIMESTAMP}.log'"
  echo "echo '[gpu2] starting tuned_er_seed46'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 46 --replay_mode er --replay_weight 0.8 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/tuned_er_seed46_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/tuned_er_seed46_${TIMESTAMP}.log'"
  echo "echo '[gpu2] worker complete'"
} >> "${gpu2_script}"

start_tmux_job "${CAMPAIGN_ID}-gpu0" 0 "${gpu0_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu1" 1 "${gpu1_script}"
start_tmux_job "${CAMPAIGN_ID}-gpu2" 2 "${gpu2_script}"
ensure_results_package_watcher

cat <<EOF
Launched distillation replay campaign: ${CAMPAIGN_ID}

Sessions:
  ${CAMPAIGN_ID}-gpu0
  ${CAMPAIGN_ID}-gpu1
  ${CAMPAIGN_ID}-gpu2
  ${RESULTS_PACKAGE_WATCHER_SESSION:-results-package-watcher}

Logs are under:
  ${LOG_DIR}

Results are under:
  ${RESULTS_ROOT}/distill_* and ${RESULTS_ROOT}/tuned_er_*
EOF
