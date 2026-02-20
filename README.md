# Sensory Perceiver

Sensory Perceiver is a multimodal training/evaluation codebase centered on:
- modality-specific towers,
- cross-modal attention fusion,
- replay-aware continual learning workflows,
- service-oriented runtime entrypoints.

## Repository Layout

- `src/models/` — canonical model package (`core.py` + exports)
- `src/train.py` — training CLI entrypoint
- `src/eval.py` — evaluation CLI entrypoint
- `src/experiments/experiment_orchestrator.py` — multi-run orchestration CLI
- `src/services/` — training, evaluation, orchestration service layer
- `src/runtime/` — runtime config, validation, typed errors
- `src/observability/` — structured events + metrics collection
- `scripts/` — utility scripts for setup, orchestration, result processing
- `docs/` — usage and benchmark documentation

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
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

## Notes

- Use module entrypoints (`python -m src.*`) as the canonical execution path.
- Results are written under `results/` with per-run configs, metrics, checkpoints, and eval outputs.
- Baseline workflows are documented in `docs/baselines.md` and scripts under `scripts/`.
