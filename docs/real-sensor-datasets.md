# Real Sensor Dataset Scaffold

This document defines the real-data path for:

- `wisdm`
- `pamap2`
- `opportunity`

and the next queued candidate datasets:

- `uci_har`
- `hhar`
- `mhealth`
- `ubiqlog`
- `selfback`
- `geomagnetic_wlan`

The goal is to plug real datasets into the existing `sensor_streaming` experiment family without inventing a second runtime or a parallel experiment contract.

Current implementation status:

- `wisdm`: prepared raw-to-canonical parser implemented
- `pamap2`: scaffold only
- `opportunity`: scaffold only
- `uci_har`: candidate only
- `hhar`: candidate only
- `mhealth`: candidate only
- `ubiqlog`: candidate only
- `selfback`: candidate only
- `geomagnetic_wlan`: candidate only

## Dataset Matrix

| Dataset | Domain | Current status | Notes |
| --- | --- | --- | --- |
| `wisdm` | phone + smartwatch activity sensing | runnable once raw files are present | first working raw-to-canonical real-data path |
| `pamap2` | body-worn IMU + heart rate | scaffold only | canonical prepared contract exists, parser pending |
| `opportunity` | wearable + ambient + object sensing | scaffold only | canonical prepared contract exists, parser pending |
| `uci_har` | smartphone inertial activity sensing | candidate only | likely useful control dataset, but single-device and not yet mapped into the tri-modal contract |
| `hhar` | heterogeneous phone/watch sensing | candidate only | strong next wearable candidate once WISDM path is stable |
| `mhealth` | multimodal body sensing | candidate only | good body-sensing follow-on after PAMAP2 |
| `ubiqlog` | smartphone context logging | candidate only | requires a different event/context projection strategy |
| `selfback` | activity / coaching sensing | candidate only | parser and canonical projection not implemented |
| `geomagnetic_wlan` | indoor localization | candidate only | useful localization benchmark, but not yet integrated into the current modality contract |

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

## Bootstrap / Prepare Command

Create the scaffold for one dataset:

```bash
python3 scripts/prepare_real_sensor_dataset.py \
  --dataset wisdm \
  --raw-dir data/raw/wisdm
```

Behavior:

- `wisdm`: parses the raw inertial files and writes prepared `train.pt` / `val.pt`
- `pamap2` and `opportunity`: create scaffold only

To force scaffold-only behavior even for WISDM:

```bash
python3 scripts/prepare_real_sensor_dataset.py \
  --dataset wisdm \
  --raw-dir data/raw/wisdm \
  --bootstrap-only
```

The scaffold path creates:

- `data/real-sensor/<dataset>/manifest.json`
- `data/real-sensor/<dataset>/README.md`

## WISDM Projection Strategy

WISDM does not naturally provide `room_env` or `robot_arm` streams. The current prepared path therefore uses an explicit proxy projection:

- `smartwatch`: watch accelerometer xyz + watch gyroscope xyz + watch accelerometer norm
- `room_env`: phone accelerometer xyz + phone accelerometer norm
- `robot_arm`: phone accelerometer xyz + phone gyroscope xyz + cross-device delta features

This is a usable real-data routing benchmark, but it must be described honestly as a proxy projection into the canonical modality contract, not as literal ambient-room or robot-arm semantics.

## Real-Data Autonomous Path

The Titan-local real-data supervisor is:

- `scripts/run_real_sensor_sequence_autonomous.sh`
- backed by `scripts/real_sensor_ai_supervisor.py`

It is bounded and honest:

1. it only launches datasets that are actually prepared
2. it will auto-prepare `wisdm` if `data/raw/wisdm` exists
3. it marks `pamap2` and `opportunity` as blocked until real parsers or prepared splits exist
4. it records candidate-only datasets as queued, not runnable

Status command:

```bash
bash scripts/real_sensor_sequence_status.sh <sequence-id>
```

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
