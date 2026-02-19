# Quick Start Guide

Get Sensory Perceiver running in 5 minutes.

## Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU experiments)
- 16GB+ GPU memory (for 3-GPU parallel experiments)

## Setup (2 minutes)

```bash
# Clone/navigate to project
cd perception-is-how-we-experience

# Install dependencies
pip install -r config/requirements.txt

# Verify setup
python scripts/validate_setup.py
```

## Run a Quick Test (3 minutes)

```bash
# Single GPU test (CIFAR-100, 1 epoch)
python src/train.py \
  --dataset cifar100 \
  --num_epochs 1 \
  --batch_size 32 \
  --use_gpu

# Output: Results saved to results/
```

## Run Full Experiments (18 hours)

```bash
# Multi-GPU orchestration
python src/experiments/experiment_orchestrator.py \
  --num_gpus 3 \
  --experiment_set multimodal \
  --full_validation

# Experiments run in parallel
# - Split-CIFAR-100 continual learning
# - Synthetic robotics (4 modalities)
# - Synthetic medical (4 modalities)
# - Synthetic smart home (5 modalities)
# - Ablation studies (3 variants)
```

## View Results

```bash
# Aggregate results
python scripts/aggregate_results.py

# Generate plots
python scripts/plot_results.py

# Results in: results/
```

## Project Structure

- **Code**: `src/` (models, training, evaluation)
- **Paper**: `main-paper/` (main manuscript, figures)
- **Planning**: `planning/` (execution plans, timelines)
- **Research**: `research/` (frameworks, analysis, outlines)

## Key Commands

| Task | Command |
|------|---------|
| Setup environment | `config/SETUP_GUIDE.md` |
| Run experiments | `python src/experiments/experiment_orchestrator.py` |
| View paper | `main-paper/Perception Is How We Experience.md` |
| Check status | `PROGRESS.md` |
| See execution plan | `planning/EXECUTION_READY.md` |

## Next Steps

1. Run validation setup: `python scripts/validate_setup.py`
2. Execute M1 experiments: `python src/experiments/experiment_orchestrator.py`
3. Check progress: See `PROGRESS.md`

For full details, see `README.md` and `planning/EXECUTION_READY.md`.

---

**Questions?** See `docs/TROUBLESHOOTING.md` or check `docs/FAQ.md`
