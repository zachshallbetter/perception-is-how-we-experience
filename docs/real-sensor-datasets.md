# Real Dataset Scaffold

This document defines the bounded real-data lineup that is currently integrated into the repository and Titan supervisors.

## Active Integrated Sensor Lineup

- `wisdm`
- `pamap2`
- `opportunity`
- `uci_har`
- `hhar`
- `mhealth`
- `selfback`
- `geomagnetic_wlan`
- `realworld_har`

## Active Integrated Audio Lineup

- `speech_commands`
- `esc50`
- `urbansound8k`
- `fsd50k`

## Deferred Legacy Sources

These are no longer part of the active bounded chain:

- `ubiqlog`
  - raw logs exist, but the repository still does not define a canonical supervised task target
  - the active HF-backed replacement is `realworld_har`
- `audioset`
  - only metadata is locally available, not audio media
  - the active HF-backed replacement is `fsd50k`

## Current Implementation Status

Sensor datasets with raw-to-canonical preparation implemented:

- `wisdm`
- `pamap2`
- `opportunity`
- `uci_har`
- `hhar`
- `mhealth`
- `selfback`
- `geomagnetic_wlan`
- `realworld_har`

Audio datasets with raw-to-canonical preparation implemented:

- `speech_commands`
- `esc50`
- `urbansound8k`
- `fsd50k`

`urbansound8k` uses an `ffmpeg` fallback for non-PCM WAVs.

`fsd50k` and `realworld_har` are HF-backed replacements and can bootstrap their raw assets during preparation when the local raw directory is missing.

## Dataset Matrix

| Dataset | Domain | Current status | Notes |
| --- | --- | --- | --- |
| `wisdm` | phone + smartwatch activity sensing | runnable | first working raw-to-canonical real-sensor path |
| `pamap2` | body-worn IMU + heart rate | runnable | body-sensing projection into the canonical sensor contract |
| `opportunity` | wearable + ambient + object sensing | runnable | locomotion target with canonical proxy routing |
| `uci_har` | smartphone inertial activity sensing | runnable | single-device benchmark projected into the tri-modal contract |
| `hhar` | heterogeneous phone/watch sensing | runnable | wearable routing benchmark with user-holdout splits |
| `mhealth` | multimodal body sensing | runnable | chest/wrist/ankle sensors projected into canonical modalities |
| `selfback` | activity / coaching sensing | runnable | phone/watch accelerometer routing benchmark |
| `geomagnetic_wlan` | indoor localization | runnable | phone/watch/WLAN localization benchmark |
| `realworld_har` | wearable accelerometer HAR (HF) | runnable | HF-hosted supervised HAR replacement for the old UbiqLog slot |
| `speech_commands` | spoken command recognition | runnable | first working prepared-audio path in the `multimodal` family |
| `esc50` | environmental audio classification | runnable | non-speech semantic audio follow-on |
| `urbansound8k` | urban event audio classification | runnable | ambient audio benchmark with `ffmpeg` WAV fallback |
| `fsd50k` | large-scale environmental audio events (HF) | runnable | HF-backed replacement for the old AudioSet candidate slot |

## Canonical Prepared Contracts

Sensor datasets use the `sensor_streaming` experiment family and prepare into:

- `data/real-sensor/<dataset>/manifest.json`
- `data/real-sensor/<dataset>/train.pt`
- `data/real-sensor/<dataset>/val.pt`

Each split file must be a `torch.save()` payload with:

- `labels`: `LongTensor[N]`
- `smartwatch`: `FloatTensor[N, T, 7]`
- `room_env`: `FloatTensor[N, T, 4]`
- `robot_arm`: `FloatTensor[N, T, 9]`

Audio datasets use the `multimodal` experiment family and prepare into:

- `data/real-audio/<dataset>/manifest.json`
- `data/real-audio/<dataset>/train.pt`
- `data/real-audio/<dataset>/val.pt`

Each prepared audio split must provide:

- `labels`: `LongTensor[N]`
- `audio`: `FloatTensor[N, T, 128]`

## Important Boundary

The repository still does **not** claim that every real dataset naturally maps one-to-one into the canonical modality identities.

The contract is:

1. choose and document an explicit projection into the canonical feature layout
2. record that projection in the dataset manifest / dataset card
3. use the same projection consistently across train and validation

`realworld_har` and `fsd50k` both follow this rule:

- `realworld_har` projects a single accelerometer stream into canonical proxy sensor modalities
- `fsd50k` projects multi-label environmental audio into the current single-label prepared-audio contract using a deterministic first-label rule

## Prepare Commands

Sensor example:

```bash
python3 scripts/prepare_real_sensor_dataset.py \
  --dataset realworld_har \
  --raw-dir data/raw/realworld_har
```

Audio example:

```bash
python3 scripts/prepare_real_audio_dataset.py \
  --dataset fsd50k \
  --raw-dir data/raw/audio/fsd50k
```

Behavior:

- integrated datasets parse raw files and write canonical prepared `train.pt` / `val.pt`
- `realworld_har` and `fsd50k` may bootstrap their raw HF assets automatically if needed
- `ubiqlog` and `audioset` remain deferred legacy sources, not active bounded-sequence datasets

## Real-Data Autonomous Path

The Titan-local supervisors are:

- `scripts/run_real_audio_sequence_autonomous.sh`
- `scripts/run_real_sensor_sequence_autonomous.sh`

They are bounded and honest:

1. they only launch datasets in their declared integrated lineups
2. they auto-prepare supported datasets
3. they do not silently resurrect deferred legacy datasets
4. they report only active integrated datasets and their completion/block states

Default bounded sequences now target:

- audio: `speech_commands`, `esc50`, `urbansound8k`, `fsd50k`
- sensor: `wisdm`, `pamap2`, `opportunity`, `uci_har`, `hhar`, `mhealth`, `selfback`, `geomagnetic_wlan`, `realworld_har`

Status commands:

```bash
bash scripts/real_audio_sequence_status.sh <sequence-id>
bash scripts/real_sensor_sequence_status.sh <sequence-id>
```

## Runtime Integration

Once prepared files exist, the standard train/eval entrypoints use them directly. Example:

```bash
python -m src.train \
  --experiment multimodal \
  --dataset fsd50k \
  --output_dir results/fsd50k_smoke
```

```bash
python -m src.train \
  --experiment sensor_streaming \
  --dataset realworld_har \
  --modality_routing_mode canonical \
  --output_dir results/realworld_har_smoke
```

The broader story stays the same:

- synthetic evidence remains the strongest completed perception-side result
- bootstrap real-data evidence remains useful
- integrated real-data sequences remain the path to the next canonical paper update
