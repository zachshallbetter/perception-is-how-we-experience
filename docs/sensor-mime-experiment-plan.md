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

The current generator is intentionally harder than the earlier random-label scaffold:

- labels map to compositional modality codes rather than one globally unique per-modality signature
- each modality is individually ambiguous for some classes
- samples include shared nuisance structure
- validation uses a shifted generator regime rather than the exact same train distribution

## What This Experiment Can Support

If `canonical` outperforms `merged`, especially with similar or lower latency, that supports:

- typed modality structure matters empirically
- the MIME registry is not only documentation or protocol ceremony
- the broader perception framing has an actual benchmark foothold outside Split-CIFAR

If the result is neutral or negative, that is also useful:

- the broader claim should be narrowed
- MIME semantics may still be valuable for governance/runtime correctness, but not yet demonstrated as a performance lever

Even with these hardenings, this remains synthetic evidence. It should be interpreted as controlled perception-side support, not as a substitute for a real large-scale multimodal benchmark suite.

## Launcher

Canonical Titan entrypoint:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
bash scripts/launch_sensor_mime_ablation_titan.sh
```

## Campaign Layout

- GPU 0:
  - canonical seed 42
  - merged seed 42
- GPU 1:
  - canonical seed 43
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

Current operational policy:

- run the baseline canonical-vs-merged family first
- run a capacity-matched control before promoting any positive routing claim
- aggregate before interpretation
- then run the bounded follow-on sequence:
  - leave-one-out
  - corruption
  - noise robustness
  - capacity-matched control as a required gate for positive results

This sequence is now automatable through `scripts/run_sensor_sequence_autonomous.sh` and `scripts/sensor_ai_supervisor.py`.

## Interpretation Rules

- If canonical mean accuracy is higher than merged with comparable latency, promote this as the key perception-side result.
- If canonical and merged are effectively tied, treat MIME semantics as a runtime/architecture contribution rather than a demonstrated model-performance contribution.
- If merged wins, narrow the paper’s modality-routing claim sharply.

## Current Completed Reading

Completed ladder branches show:

- baseline gap: `+0.4583`
- capacity-matched baseline gap: `+0.3767`
- corruption gap: `+0.2717`
- leave-one-out gaps:
  - `drop_smartwatch`: `+0.3317`
  - `drop_room_env`: `+0.7950`
  - `drop_robot_arm`: `+0.6333`
- noise gaps:
  - `noise_smartwatch`: `+0.0633`
  - `noise_room_env`: `+0.0617`
  - `noise_robot_arm`: `-0.0933`

So the current honest reading is:

- typed canonical routing is leading on the overall ladder and on every branch except `noise_robot_arm`,
- that lead survives a capacity-matched control,
- but the margin becomes much smaller under single-modality noise stress,
- and merged routing can win under specific stress (`noise_robot_arm`),
- which means the result is strong synthetic support, but still not a license to overclaim robustness.

## Why This Matters

This experiment is the shortest path from:

- "interesting architecture story"

to:

- "architecture story with a directly measured perception-side result"

## Decision States

The bounded Titan-local supervisor reduces the baseline comparison to one of four states:

1. `canonical-dominant`
   - typed routing wins clearly
   - capacity-matched control remains positive
   - most stress variants still favor canonical
2. `capacity-confounded`
   - baseline canonical win does not survive the capacity-matched control
   - architectural claim should be narrowed before promotion
3. `competitive`
   - canonical and merged are close
   - or stress results are mixed enough that the claim remains provisional
4. `merged-favored`
   - generic merged routing is stronger
   - the paper claim must narrow toward runtime/governance value rather than model-performance advantage
