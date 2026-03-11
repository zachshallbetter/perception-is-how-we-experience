#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: restart_campaign_gpu_baseline.sh <campaign-id> <gpu-index> <model> [timestamp-key]"
  exit 1
fi

CAMPAIGN_ID="$1"
GPU_INDEX="$2"
MODEL="$3"
TS_KEY="${4:-${CAMPAIGN_ID#paper-campaign-}}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${MODEL}" != "perceiver" ] && [ "${MODEL}" != "perceiver_io" ]; then
  echo "Model must be one of: perceiver, perceiver_io"
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p results/titan_logs

session="${CAMPAIGN_ID}-gpu${GPU_INDEX}"

if tmux has-session -t "${session}" 2>/dev/null; then
  tmux kill-session -t "${session}"
fi

tmux new-session -d -s "${session}" \
  "cd '${ROOT_DIR}' && export CUDA_VISIBLE_DEVICES='${GPU_INDEX}' && \
   for seed in 42 43 44; do \
     /ai/envs/torch-cu118-smoke/bin/python scripts/run_baselines_cl.py --model '${MODEL}' --n_tasks 20 --epochs_per_task 5 --batch_size 4 --seed \"\$seed\" \
       --output_dir \"results/paper_baseline_${MODEL}_20t_seed\${seed}_${TS_KEY}_recover\" 2>&1 | \
       tee \"results/titan_logs/${CAMPAIGN_ID}-gpu${GPU_INDEX}-${MODEL}-seed\${seed}-recover.log\"; \
   done"

echo "Restarted ${session} (${MODEL})"
