# Real Sensor Dataset Scaffold

This document defines the first real-data ingestion scaffold for:

- `wisdm`
- `pamap2`
- `opportunity`

The goal is to plug real datasets into the existing `sensor_streaming` experiment family without inventing a second runtime or a parallel experiment contract.

## Canonical Direction

All three real datasets are being staged through one prepared-data contract:

- experiment: `sensor_streaming`
- dataset ids:
  - `wisdm`
  - `pamap2`
  - `opportunity`

Prepared datasets live under:

- `data/real-sensor/<dataset>/`

Each prepared dataset must provide:

- `manifest.json`
- `train.pt`
- `val.pt`

## Prepared Split Contract

Each split file must be a `torch.save()` payload with:

- `labels`: `LongTensor[N]`
- `smartwatch`: `FloatTensor[N, T, 7]`
- `room_env`: `FloatTensor[N, T, 4]`
- `robot_arm`: `FloatTensor[N, T, 9]`

This keeps the real-data path aligned with the same canonical modality registry used by the synthetic sensor ladder.

## Important Boundary

The current scaffold does **not** claim that WISDM, PAMAP2, or OPPORTUNITY naturally map one-to-one into the canonical `smartwatch`, `room_env`, and `robot_arm` modalities.

Instead, the current contract says:

1. choose and document an explicit projection from the source dataset into the canonical modality shape
2. record that choice in the dataset card / manifest
3. use the same projection consistently across train and validation

If a source dataset lacks one of the canonical modalities, that missing-modality strategy must be explicit in the prepared dataset card before the data is used for paper claims.

## Bootstrap Command

Create the scaffold for one dataset:

```bash
python3 scripts/prepare_real_sensor_dataset.py \
  --dataset wisdm \
  --raw-dir data/raw/wisdm
```

Swap `wisdm` for `pamap2` or `opportunity` as needed.

This command creates:

- `data/real-sensor/<dataset>/manifest.json`
- `data/real-sensor/<dataset>/README.md`

It does **not** parse raw vendor files yet. That is the next implementation step.

## Runtime Integration

Once prepared files exist, the standard train/eval entrypoints can use them:

```bash
python -m src.train \
  --experiment sensor_streaming \
  --dataset wisdm \
  --modality_routing_mode canonical \
  --output_dir results/wisdm_smoke
```

```bash
python -m src.eval \
  --checkpoint results/wisdm_smoke/best_model.pt \
  --experiment sensor_streaming \
  --dataset wisdm \
  --output_file results/wisdm_smoke/eval_results.json
```

The same runtime path also works for:

- `pamap2`
- `opportunity`

provided their prepared manifests and split files are present.
