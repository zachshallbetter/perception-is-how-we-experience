#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/results/titan_logs}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_SEQUENCE_ID="${SOURCE_SEQUENCE_ID:-real-sensor-seq-002}"
CAMPAIGN_ID="${CAMPAIGN_ID:-real-sensor-distinction-${TIMESTAMP}}"
DATASETS="${DATASETS:-pamap2,mhealth,opportunity,uci_har,hhar}"
ROUTINGS="${ROUTINGS:-canonical,merged}"
VARIANTS="${VARIANTS:-drop_smartwatch,drop_room_env,drop_robot_arm,noise_smartwatch,noise_room_env,noise_robot_arm}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
IFS=',' read -r -a ROUTING_LIST <<< "${ROUTINGS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

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
    echo "DATASETS='${DATASETS}'"
    echo "ROUTINGS='${ROUTINGS}'"
    echo "VARIANTS='${VARIANTS}'"
    echo "IFS=',' read -r -a DATASET_LIST <<< \"\${DATASETS}\""
    echo "IFS=',' read -r -a ROUTING_LIST <<< \"\${ROUTINGS}\""
    echo "IFS=',' read -r -a VARIANT_LIST <<< \"\${VARIANTS}\""
    echo "for dataset in \"\${DATASET_LIST[@]}\"; do"
    echo "  for routing in \"\${ROUTING_LIST[@]}\"; do"
    echo "    checkpoint='${RESULTS_ROOT}/real_sensor_'\"\${dataset}\"'_'\"\${routing}\"'_seed${seed}_${SOURCE_SEQUENCE_ID}-'\"\${dataset}\"'/best_model.pt'"
    echo "    test -f \"\${checkpoint}\""
    echo "    for variant in \"\${VARIANT_LIST[@]}\"; do"
    echo "      output_dir='${RESULTS_ROOT}/real_sensor_probe_'\"\${dataset}\"'_'\"\${routing}\"'_'\"\${variant}\"'_seed${seed}_${CAMPAIGN_ID}'"
    echo "      output_file=\"\${output_dir}/eval_results.json\""
    echo "      log_file='${LOG_DIR}/real_sensor_probe_'\"\${dataset}\"'_'\"\${routing}\"'_'\"\${variant}\"'_seed${seed}_${CAMPAIGN_ID}.log'"
    echo "      if [ -f \"\${output_file}\" ]; then"
    echo "        echo \"[probe] skip existing dataset=\${dataset} routing=\${routing} variant=\${variant} seed=${seed}\""
    echo "        continue"
    echo "      fi"
    echo "      mkdir -p \"\${output_dir}\""
    echo "      '${PYTHON_BIN}' -m src.eval --checkpoint \"\${checkpoint}\" --experiment sensor_streaming --dataset \"\${dataset}\" --modality_routing_mode \"\${routing}\" --sensor_variant \"\${variant}\" --batch_size '${BATCH_SIZE}' --num_workers '${NUM_WORKERS}' --seed '${seed}' --output_file \"\${output_file}\" 2>&1 | tee \"\${log_file}\""
    echo "    done"
    echo "  done"
    echo "done"
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
Launched real-sensor distinction probe campaign: ${CAMPAIGN_ID}
Source sequence:
  ${SOURCE_SEQUENCE_ID}
Datasets:
  ${DATASETS}
Routings:
  ${ROUTINGS}
Variants:
  ${VARIANTS}

Sessions:
  ${CAMPAIGN_ID}-gpu0
  ${CAMPAIGN_ID}-gpu1
  ${CAMPAIGN_ID}-gpu2
EOF
