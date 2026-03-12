# Sensor/MIME Experiment Plan

## Goal

Turn the broader perception/MIME claim into a measured result instead of leaving it mostly architectural.

## Core Comparison

We compare two routing modes on the same generated `sensor_streaming` benchmark:

1. `canonical`
   - three typed modalities:
     - `smartwatch`
     - `room_env`
     - `robot_arm`
   - separate modality-specific towers
   - runtime modality setup derived from the MIME registry

2. `merged`
   - one generic `sensor_stream` input
   - all 20 features concatenated into a single stream
   - no typed modality separation inside the model

This is the cleanest practical approximation of:

- with explicit typed modality routing
- without explicit typed modality routing

## What This Experiment Can Support

If `canonical` outperforms `merged`, especially with similar or lower latency, that supports:

- typed modality structure matters empirically
- the MIME registry is not only documentation or protocol ceremony
- the broader perception framing has an actual benchmark foothold outside Split-CIFAR

If the result is neutral or negative, that is also useful:

- the broader claim should be narrowed
- MIME semantics may still be valuable for governance/runtime correctness, but not yet demonstrated as a performance lever

## Launcher

Canonical Titan entrypoint:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_mime_ablation_titan.sh
```

## Campaign Layout

- GPU 0:
  - canonical seed 42
  - canonical seed 43
- GPU 1:
  - merged seed 42
  - merged seed 43
- GPU 2:
  - canonical seed 44
  - merged seed 44

Each run includes:

- training via `src.train`
- evaluation via `src.eval`
- separate output directory
- per-run log in `results/titan_logs/`

## Metrics To Report

Primary:

- `overall_accuracy` from `eval_results.json`
- multiseed mean and standard deviation

Secondary:

- latency summary from `eval_results.json`
- train/validation curves from `train_log.csv`

## Interpretation Rules

- If canonical mean accuracy is higher than merged with comparable latency, promote this as the key perception-side result.
- If canonical and merged are effectively tied, treat MIME semantics as a runtime/architecture contribution rather than a demonstrated model-performance contribution.
- If merged wins, narrow the paper’s modality-routing claim sharply.

## Why This Matters

This experiment is the shortest path from:

- "interesting architecture story"

to:

- "architecture story with a directly measured perception-side result"
