# Real Dataset Scaffold

This document defines the real-data path for:

- `wisdm`
- `pamap2`
- `opportunity`
- `uci_har`
- `hhar`
- `mhealth`
- `selfback`
- `geomagnetic_wlan`

the still-blocked sensor/body candidate:

- `ubiqlog`

and the audio lineup:

- `speech_commands`
- `esc50`
- `urbansound8k`
- `audioset`

The goal is to plug real datasets into the existing `sensor_streaming` experiment family without inventing a second runtime or a parallel experiment contract.

Current implementation status:

- `wisdm`: prepared raw-to-canonical parser implemented
- `pamap2`: prepared raw-to-canonical parser implemented
- `opportunity`: prepared raw-to-canonical parser implemented
- `uci_har`: prepared raw-to-canonical parser implemented
- `hhar`: prepared raw-to-canonical parser implemented
- `mhealth`: prepared raw-to-canonical parser implemented
- `selfback`: prepared raw-to-canonical parser implemented
- `geomagnetic_wlan`: prepared raw-to-canonical parser implemented
- `ubiqlog`: blocked until a canonical supervised task is defined
- audio datasets:
  - `speech_commands`: prepared raw-to-canonical parser implemented
  - `esc50`: prepared raw-to-canonical parser implemented
  - `urbansound8k`: prepared raw-to-canonical parser implemented (requires `ffmpeg` fallback for non-PCM WAVs)
  - `audioset`: metadata-only scaffold, blocked until audio media is acquired

## Dataset Matrix

| Dataset | Domain | Current status | Notes |
| --- | --- | --- | --- |
| `wisdm` | phone + smartwatch activity sensing | runnable once raw files are present | first working raw-to-canonical real-data path |
| `pamap2` | body-worn IMU + heart rate | runnable once raw files are present | body-sensing projection into canonical sensor contract |
| `opportunity` | wearable + ambient + object sensing | runnable once raw files are present | locomotion target with canonical proxy routing |
| `uci_har` | smartphone inertial activity sensing | runnable once raw files are present | single-device benchmark projected into the tri-modal contract |
| `hhar` | heterogeneous phone/watch sensing | runnable once raw files are present | strong wearable routing benchmark with user-holdout splits |
| `mhealth` | multimodal body sensing | runnable once raw files are present | chest/wrist/ankle sensors projected into canonical modalities |
| `ubiqlog` | smartphone context logging | blocked | requires an explicit supervised task definition |
| `selfback` | activity / coaching sensing | runnable once raw files are present | phone/watch accelerometer routing benchmark |
| `geomagnetic_wlan` | indoor localization | runnable once raw files are present | phone/watch/WLAN localization benchmark |
| `speech_commands` | spoken command recognition | runnable once raw files are present | first working prepared-audio path in the `multimodal` experiment family |
| `esc50` | environmental audio classification | runnable once raw files are present | non-speech semantic audio follow-on after Speech Commands |
| `urbansound8k` | urban event audio classification | runnable once raw files are present | ambient audio benchmark with `ffmpeg` fallback for non-PCM WAVs |
| `audioset` | large-scale weakly labeled audio events | blocked | metadata staged, but no local audio media |

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

Audio datasets are tracked in the same bounded real-data lineup, but they are not forced through the `smartwatch` / `room_env` / `robot_arm` contract. The audio contract is separate:

- experiment: `multimodal`
- modality contract: `audio`
- prepared dataset ids:
  - `speech_commands`
  - `esc50`
  - `urbansound8k`
  - `audioset`
- prepared datasets live under `data/real-audio/<dataset>/`
- each prepared split provides:
  - `labels`: `LongTensor[N]`
  - `audio`: `FloatTensor[N, T, 128]`

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

- integrated sensor datasets parse raw files and write prepared `train.pt` / `val.pt`
- `ubiqlog` remains blocked until a supervised task is defined

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
2. it auto-prepares any integrated sensor dataset whose raw root exists locally
3. it leaves `ubiqlog` explicitly blocked until a canonical supervised task exists
4. it records blocked audio candidates separately from runnable sensor datasets

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
- `uci_har`
- `hhar`
- `mhealth`
- `selfback`
- `geomagnetic_wlan`

provided their prepared manifests and split files are present.

The prepared audio path is intentionally separate from the canonical tri-modal sensor contract. Current audio integration order is:

1. `speech_commands`
2. `esc50`
3. `urbansound8k`
4. `audioset`

Current audio status:
- `speech_commands`: integrated into the runtime and Titan automation path
- `esc50`: integrated into the runtime and Titan automation path
- `urbansound8k`: integrated into the runtime and Titan automation path
- `audioset`: still blocked until audio media is acquired locally
