#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CAMPAIGN_ID="${CAMPAIGN_ID:-bwt-campaign-${TIMESTAMP}}"
SESSION_GPU0="${CAMPAIGN_ID}-gpu0"
SESSION_GPU1="${CAMPAIGN_ID}-gpu1"
SESSION_GPU2="${CAMPAIGN_ID}-gpu2"

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

gpu0_script="$(make_worker_script "gpu0")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "echo '[gpu0] starting bwt_replay_er_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 0.8 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/bwt_replay_er_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_replay_er_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu0] starting bwt_replay_er_high_buffer_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/bwt_replay_er_high_buffer_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_replay_er_high_buffer_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu0] starting bwt_replay_contrastive_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode contrastive --replay_weight 0.8 --replay_buffer 8000 --ewc_lambda 0.6 --output_dir '${RESULTS_ROOT}/bwt_replay_contrastive_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_replay_contrastive_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu0] worker complete'"
} >> "${gpu0_script}"

gpu1_script="$(make_worker_script "gpu1")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "echo '[gpu1] starting bwt_ewc_1_0_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 1.0 --output_dir '${RESULTS_ROOT}/bwt_ewc_1_0_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_ewc_1_0_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu1] starting bwt_ewc_2_0_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 2.0 --output_dir '${RESULTS_ROOT}/bwt_ewc_2_0_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_ewc_2_0_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu1] starting bwt_ewc_3_0_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 3.0 --output_dir '${RESULTS_ROOT}/bwt_ewc_3_0_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_ewc_3_0_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu1] worker complete'"
} >> "${gpu1_script}"

gpu2_script="$(make_worker_script "gpu2")"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd '${ROOT_DIR}'"
  echo "echo '[gpu2] starting bwt_si_0_1_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 1.0 --si_lambda 0.1 --output_dir '${RESULTS_ROOT}/bwt_si_0_1_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_si_0_1_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu2] starting bwt_si_0_5_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 1.0 --si_lambda 0.5 --output_dir '${RESULTS_ROOT}/bwt_si_0_5_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_si_0_5_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu2] starting bwt_no_parameter_isolation_seed42'"
  echo "'${PYTHON_BIN}' scripts/run_local.py --n_tasks 20 --epochs_per_task 5 --batch_size 32 --seed 42 --replay_mode er --replay_weight 1.0 --replay_buffer 12000 --ewc_lambda 1.0 --no_parameter_isolation --output_dir '${RESULTS_ROOT}/bwt_no_parameter_isolation_seed42_${TIMESTAMP}' 2>&1 | tee '${LOG_DIR}/bwt_no_parameter_isolation_seed42_${TIMESTAMP}.log'"
  echo "echo '[gpu2] worker complete'"
} >> "${gpu2_script}"

start_tmux_job "${SESSION_GPU0}" 0 "${gpu0_script}"
start_tmux_job "${SESSION_GPU1}" 1 "${gpu1_script}"
start_tmux_job "${SESSION_GPU2}" 2 "${gpu2_script}"

cat <<EOF
Launched BWT improvement campaign: ${CAMPAIGN_ID}

Sessions:
  ${SESSION_GPU0}
  ${SESSION_GPU1}
  ${SESSION_GPU2}

Logs are under:
  ${LOG_DIR}

Results are under:
  ${RESULTS_ROOT}/bwt_*
EOF
