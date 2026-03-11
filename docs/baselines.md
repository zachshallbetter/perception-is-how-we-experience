# Sensory Perceiver Baselines

This document defines the paper-primary baseline workflow for continual learning comparisons.

## 1. Canonical Baselines

The authoritative baselines for the current paper runs are:

- `perceiver`
- `perceiver_io`

Both execute through:

```bash
python scripts/run_baselines_cl.py --model <perceiver|perceiver_io>
```

## 2. Environments and Dependencies

Use a separate baseline environment:

```bash
pip install -r requirements.txt
pip install -r requirements-baselines.txt
```

`requirements-baselines.txt` pins baseline-only dependencies (including `transformers<5`) to avoid breaking the core training environment.

## 3. Titan-Optimized Defaults

For 12 GB Titan X GPUs, use:

- `n_tasks=20`
- `epochs_per_task=5`
- `batch_size=4`

Example:

```bash
python scripts/run_baselines_cl.py \
  --model perceiver \
  --n_tasks 20 \
  --epochs_per_task 5 \
  --batch_size 4 \
  --seed 42 \
  --output_dir results/paper_baseline_perceiver_20t_seed42
```

Perceiver IO:

```bash
python scripts/run_baselines_cl.py \
  --model perceiver_io \
  --n_tasks 20 \
  --epochs_per_task 5 \
  --batch_size 4 \
  --seed 42 \
  --output_dir results/paper_baseline_perceiver_io_20t_seed42
```

## 4. Campaign Launcher

For unattended multi-seed runs on Titan:

```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
CORE_SEEDS=42,43,44 \
BASELINE_SEEDS=42,43,44 \
bash scripts/launch_titan_continuous_campaign.sh
```

This launcher queues baseline seeds on GPU 1 and GPU 2 while GPU 0 runs core-model seeds and replay-depth ablation.
