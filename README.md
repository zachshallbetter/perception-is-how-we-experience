# Sensory Perceiver

Sensory Perceiver is a research codebase for machine perception under continual adaptation. The repository combines:
- modality-specific towers,
- shared cross-modal attention,
- replay-aware continual-learning workflows,
- typed runtime contracts and service-layer entrypoints,
- Titan-oriented experiment automation and paper packaging.

## What This Project Is

Sensory Perceiver is a research system for machine perception. It is designed around the idea that a perceptual model should:

- accept heterogeneous sensory streams as explicitly typed inputs,
- preserve modality identity through modality-specific towers,
- integrate those streams through a shared perceptual core,
- and continue adapting over time without catastrophic forgetting.

In this repository, that broader idea appears in two coupled layers:

1. **Perception architecture**
   - modality-specific towers plus shared cross-modal attention in [core.py](/Users/zachshallbetter/Projects/Perception/src/models/core.py)
   - canonical MIME-style modality registry for `sensor_streaming` in [mime_cases.py](/Users/zachshallbetter/Projects/Perception/src/runtime/mime_cases.py)
   - sensor-streaming modality definitions for `smartwatch`, `room_env`, and `robot_arm`

2. **Continual-learning evidence path**
   - replay-centric continual-learning benchmarks on Split-CIFAR-100
   - task-adapter isolation primitives and opt-in sequential rollover
   - paper-primary empirical results currently centered on replay-path design and retention

The important distinction is:

- **Conceptually**, this project is about multimodal perception with explicit modality semantics and temporal adaptation.
- **Empirically**, the strongest current validated evidence is still continual-learning behavior, especially how replay placement affects forgetting.

So this is not just a CIFAR experiment repo, and it is not yet a finished large-scale multimodal foundation model. It is a perception research platform whose current best-proven result is replay-centric continual learning.

## What This Project Is Not

To keep the claims honest:

- it is not yet a comprehensively validated large-scale multimodal perception benchmark suite
- it is not yet a paper where the broadest sensor/MIME claims have more evidence than the continual-learning core
- it is not a claim that MIME strings are parsed directly inside model `forward()`

The strongest current evidence is still replay-centric continual learning. The next most important evidence path is the sensor/MIME canonical-vs-merged ablation now exposed through `scripts/launch_sensor_mime_ablation_titan.sh`, which uses a harder synthetic sensor task with compositional class codes, shared nuisance structure, and a shifted validation regime instead of random labels. The follow-on sequence is also wired: leave-one-out, MIME-binding corruption, single-modality noise robustness, and capacity-matched controls, with a bounded Titan-local supervisor to drive the sequence autonomously.

## Repository Layout

- `src/models/` — canonical model package (`core.py` + exports)
- `src/train.py` — training CLI entrypoint
- `src/eval.py` — evaluation CLI entrypoint
- `src/experiments/experiment_orchestrator.py` — multi-run orchestration CLI
- `src/services/` — training, evaluation, orchestration service layer
- `src/runtime/` — runtime config, validation, typed errors
- `src/observability/` — structured events + metrics collection
- `scripts/` — utility scripts for setup, orchestration, result processing
- `docs/` — operator guides, benchmark plans, and review/rebuttal notes
- `paper/sections/` — canonical paper source of truth
- `paper/figures/generated/results_package/` — current paper-ready tables/figures package

## Canonical Reading Order

If you are orienting yourself to the project, read in this order:

1. `README.md` for project scope and operating model.
2. `docs/README.md` for the documentation map.
3. `docs/training-guide.md` for the canonical Titan workflow.
4. `docs/sensor-mime-experiment-plan.md` for the current perception-side evidence ladder.
5. `paper/sections/README.md` plus `paper/sections/06-results.md` and `paper/sections/07-discussion.md` for the current paper-aligned claim surface.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Baseline wrappers should stay on `transformers` 4.x, but that dependency should live in a separate baseline environment. The core training environment does not require `transformers`.

Canonical 3-GPU paper launch:
```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
bash scripts/launch_paper_benchmarks_titan.sh
```

Set-and-forget multi-seed campaign:
```bash
CORE_PYTHON=/ai/envs/perception/bin/python \
BASELINE_PYTHON=/ai/envs/torch-cu118-smoke/bin/python \
CORE_SEEDS=42,43,44 \
BASELINE_SEEDS=42,43,44 \
bash scripts/launch_titan_continuous_campaign.sh
```

Exploratory distillation replay campaign:
```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_distill_bwt_campaign.sh
```

Canonical sensor/MIME baseline routing comparison:
```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_mime_ablation_titan.sh
```

Autonomous bounded sensor evidence ladder:
```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/run_sensor_sequence_autonomous.sh <baseline-family-id>
```

Periodic campaign monitor:
```bash
bash scripts/monitor_titan_campaign.sh <campaign-id> 60 3
```

Canonical remote dashboard from your Mac:
```bash
bash scripts/titan_remote_dashboard.sh
```

Live watch mode:
```bash
bash scripts/titan_remote_dashboard.sh --watch 60
```

2. Run a short training job:
```bash
python -m src.train --experiment continual --dataset split_cifar --epochs 1 --output_dir results/smoke_train
```

3. Evaluate a checkpoint:
```bash
python -m src.eval --checkpoint results/smoke_train/best_model.pt --experiment continual --dataset split_cifar --output_file results/smoke_train/eval_results.json
```

4. Run orchestrator quick mode:
```bash
python -m src.experiments.experiment_orchestrator --mode quick --n-gpus 1
```

## Validation

Run runtime/service checks before longer jobs:
```bash
python3 -m unittest test_runtime_contracts.py test_service_layer_smoke.py
```

Run environment preflight checks:
```bash
python scripts/validate_setup.py --verbose
```

## Current Canonical Results

The current paper-primary continual result is the tuned-core Split-CIFAR-100 path on Titan Node:

- `scripts/run_local.py --tuned`
- 6-seed mean: `CIL 23.96%`
- 6-seed mean: `BWT -30.39%`
- Task 0 retention mean: `25.1%`

The current paper package is under:

- `paper/figures/generated/results_package/`
- `paper/sections/06-results.md`

The strongest validated conclusion remains replay-centric:

- feature-only replay fails badly,
- raw-input replay with reservoir sampling materially improves retention,
- the tuned-core replay path remains better than the current exploratory EWC/SI/distillation variants.

## Notes

- Use module entrypoints (`python -m src.*`) as the canonical execution path.
- The paper-primary continual benchmark path is `scripts/run_local.py --tuned` on Split-CIFAR-100 with 20 tasks.
- A genuinely different replay mechanism class is available via `scripts/run_local.py --replay_mode distill`, which replays raw samples through a frozen previous-task teacher and optimizes KL distillation rather than labels or feature contrast alone.
- The paper-primary baselines are `scripts/run_baselines_cl.py --model perceiver` and `--model perceiver_io`.
- Results are written under `results/` with per-run configs, metrics, checkpoints, and eval outputs.
- The canonical live status surface is `scripts/render_titan_dashboard.py`, typically via `scripts/titan_remote_dashboard.sh` from the Mac.
- Baseline workflows are documented in `docs/baselines.md` and scripts under `scripts/`.
- Sensor-streaming MIME input cases are defined in `docs/sensor-streaming-mime-cases.md` and `src/runtime/mime_cases.py`.
- Sensor/MIME follow-on experiments can now be driven by a bounded Titan-local supervisor rather than a manual queue, with results recorded under `results/sensor_sequence_summaries/`.
- Continual `src.train` jobs follow the dataset benchmark task count by default. Split-CIFAR-100 and Permuted MNIST therefore run 20 sequential tasks unless `--continual_task_count` is overridden.
- Use a separate environment for baselines and install [requirements-baselines.txt](/Users/zachshallbetter/Projects/Perception/requirements-baselines.txt) there.
