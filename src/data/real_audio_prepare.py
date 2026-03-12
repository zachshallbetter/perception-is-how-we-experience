"""Preparation helpers for real audio datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .real_audio import (
    PREPARED_AUDIO_SCHEMA_VERSION,
    REAL_AUDIO_DATASET_SPECS,
    REAL_AUDIO_FEATURE_DIM,
    get_real_audio_dataset_dir,
)


def _import_hf_load_from_disk():
    from datasets import load_from_disk

    return load_from_disk


def _find_speech_commands_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir,
        raw_dir / "speech_commands_hf",
        raw_dir.parent / "speech_commands_hf",
        raw_dir / "speech_commands_enriched",
        raw_dir.parent / "speech_commands_enriched",
        raw_dir / "speech_commands_enriched_and_annotated",
        raw_dir.parent / "speech_commands_enriched_and_annotated",
    ]
    for candidate in candidates:
        if (candidate / "dataset_dict.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate a saved Speech Commands dataset under {raw_dir}"
    )


def _resample_waveform(waveform: np.ndarray, target_length: int) -> torch.Tensor:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.shape[0] == target_length:
        return torch.from_numpy(values)
    if values.shape[0] == 0:
        return torch.zeros(target_length, dtype=torch.float32)
    if values.shape[0] == 1:
        return torch.full((target_length,), float(values[0]), dtype=torch.float32)
    base_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    resampled = np.interp(target_x, base_x, values).astype(np.float32, copy=False)
    return torch.from_numpy(resampled)


def _waveform_to_log_spectrogram(
    waveform: torch.Tensor,
    *,
    n_fft: int = 256,
    hop_length: int = 128,
    target_frames: int = 96,
) -> torch.Tensor:
    effective_n_fft = min(n_fft, max(16, int(waveform.shape[0])))
    effective_hop = max(1, min(hop_length, effective_n_fft // 2))
    window = torch.hann_window(effective_n_fft, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=effective_n_fft,
        hop_length=effective_hop,
        win_length=effective_n_fft,
        window=window,
        return_complex=True,
        center=False,
    ).abs()
    if spec.shape[0] < REAL_AUDIO_FEATURE_DIM:
        spec = torch.nn.functional.pad(spec, (0, 0, 0, REAL_AUDIO_FEATURE_DIM - spec.shape[0]))
    else:
        spec = spec[:REAL_AUDIO_FEATURE_DIM, :]
    spec = torch.log1p(spec)
    spec = spec.transpose(0, 1)
    if spec.shape[0] == target_frames:
        return spec
    spec = spec.unsqueeze(0).transpose(1, 2)
    spec = torch.nn.functional.interpolate(
        spec,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    spec = spec.transpose(1, 2).squeeze(0)
    return spec


def _encode_audio_split(examples: Iterable[dict], *, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    audio_rows: List[torch.Tensor] = []
    labels: List[int] = []
    for example in examples:
        audio = example["audio"]
        waveform = _resample_waveform(audio["array"], sequence_length)
        features = _waveform_to_log_spectrogram(waveform)
        audio_rows.append(features)
        labels.append(int(example["label"]))
    return torch.stack(audio_rows), torch.tensor(labels, dtype=torch.long)


def _standardize_audio(train_audio: torch.Tensor, val_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flattened = train_audio.reshape(-1, train_audio.shape[-1])
    mean = flattened.mean(dim=0, keepdim=True).reshape(1, 1, -1)
    std = flattened.std(dim=0, keepdim=True).reshape(1, 1, -1)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (train_audio - mean) / std, (val_audio - mean) / std


def prepare_speech_commands_dataset(
    raw_dir: Path,
    output_root: Path,
    sequence_length: int,
) -> Dict[str, object]:
    load_from_disk = _import_hf_load_from_disk()
    raw_root = _find_speech_commands_root(raw_dir.resolve())
    dataset = load_from_disk(str(raw_root))

    train_split = dataset["train"]
    val_split = dataset["validation"]

    train_audio, train_labels = _encode_audio_split(train_split, sequence_length=sequence_length)
    val_audio, val_labels = _encode_audio_split(val_split, sequence_length=sequence_length)
    train_audio, val_audio = _standardize_audio(train_audio, val_audio)

    dataset_dir = get_real_audio_dataset_dir("speech_commands", output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"audio": train_audio, "labels": train_labels}, dataset_dir / "train.pt")
    torch.save({"audio": val_audio, "labels": val_labels}, dataset_dir / "val.pt")

    int2str = train_split.features["label"].int2str
    class_names = [int2str(index) for index in range(train_split.features["label"].num_classes)]
    manifest = {
        "schema_version": PREPARED_AUDIO_SCHEMA_VERSION,
        "dataset": "speech_commands",
        "source_name": REAL_AUDIO_DATASET_SPECS["speech_commands"].source_name,
        "description": REAL_AUDIO_DATASET_SPECS["speech_commands"].description,
        "raw_dir": str(raw_root),
        "sequence_length": sequence_length,
        "feature_dim": REAL_AUDIO_FEATURE_DIM,
        "feature_type": "log_spectrogram",
        "n_classes": len(class_names),
        "class_names": class_names,
        "splits": {
            "train": "train.pt",
            "val": "val.pt",
        },
        "notes": [
            "Prepared from Hugging Face Speech Commands v0.02.",
            "Audio is resampled/padded to a fixed 1D waveform before log-spectrogram conversion.",
            "Prepared tensors have shape [N, T, 128] for the canonical audio modality.",
        ],
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dataset_dir / "README.md").write_text(
        "\n".join(
            [
                "# Speech Commands Prepared Dataset Card",
                "",
                f"Raw source: {raw_root}",
                "",
                "Prepared output contract:",
                "- manifest: `manifest.json`",
                "- train split: `train.pt`",
                "- val split: `val.pt`",
                "",
                "Expected `torch.save()` payload for each split:",
                "- `labels`: `LongTensor[N]`",
                "- `audio`: `FloatTensor[N, T, 128]`",
                "",
                "Projection:",
                "- fixed-length waveform",
                "- `torch.stft()` log-spectrogram",
                "- 128 frequency bins",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "dataset": "speech_commands",
        "output_dir": str(dataset_dir),
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "manifest": str(dataset_dir / "manifest.json"),
    }
