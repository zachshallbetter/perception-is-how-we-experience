#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: restart_campaign_gpu0.sh <campaign-id> [timestamp-key]"
  exit 1
fi

CAMPAIGN_ID="$1"
TS_KEY="${2:-${CAMPAIGN_ID#paper-campaign-}}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
mkdir -p results/titan_logs

session="${CAMPAIGN_ID}-gpu0"

if tmux has-session -t "${session}" 2>/dev/null; then
  tmux kill-session -t "${session}"
fi

tmux new-session -d -s "${session}" \
  "cd '${ROOT_DIR}' && export CUDA_VISIBLE_DEVICES=0 && \
   for seed in 42 43 44; do \
     /ai/envs/perception/bin/python scripts/run_local.py --tuned --n_tasks 20 --batch_size 32 --seed \"\$seed\" \
       --output_dir \"results/paper_runlocal_tuned_20t_seed\${seed}_${TS_KEY}_recover\" 2>&1 | \
       tee \"results/titan_logs/${CAMPAIGN_ID}-gpu0-seed\${seed}-recover.log\"; \
   done && \
   /ai/envs/perception/bin/python scripts/run_replay_depth_ablation.py --experiment continual --dataset split_cifar --seeds 42,43,44 --epochs 5 --device cuda \
      2>&1 | tee \"results/titan_logs/${CAMPAIGN_ID}-gpu0-replay-depth-recover.log\""

echo "Restarted ${session}"
