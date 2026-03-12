"""Prepared real-audio dataset registry and loaders."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import torch
from torch.utils.data import Dataset


PREPARED_AUDIO_SCHEMA_VERSION = 1
DEFAULT_REAL_AUDIO_ROOT = Path("data/real-audio")
REAL_AUDIO_FEATURE_DIM = 128


@dataclass(frozen=True)
class RealAudioDatasetSpec:
    dataset_id: str
    source_name: str
    description: str


REAL_AUDIO_DATASET_SPECS: Dict[str, RealAudioDatasetSpec] = {
    "speech_commands": RealAudioDatasetSpec(
        dataset_id="speech_commands",
        source_name="Speech Commands",
        description="Keyword spotting benchmark prepared as log-spectrogram sequences.",
    ),
    "esc50": RealAudioDatasetSpec(
        dataset_id="esc50",
        source_name="ESC-50",
        description="Environmental audio classification scaffold.",
    ),
    "urbansound8k": RealAudioDatasetSpec(
        dataset_id="urbansound8k",
        source_name="UrbanSound8K",
        description="Urban event audio classification scaffold.",
    ),
    "audioset": RealAudioDatasetSpec(
        dataset_id="audioset",
        source_name="AudioSet",
        description="Large-scale audio event scaffold.",
    ),
}


def get_real_audio_root(root: Optional[Union[str, Path]] = None) -> Path:
    if root is not None:
        return Path(root)
    env_root = os.environ.get("SENSORY_REAL_AUDIO_ROOT")
    if env_root:
        return Path(env_root)
    return DEFAULT_REAL_AUDIO_ROOT


def get_real_audio_dataset_ids() -> Iterable[str]:
    return REAL_AUDIO_DATASET_SPECS.keys()


def is_real_audio_dataset(dataset_name: str) -> bool:
    return dataset_name in REAL_AUDIO_DATASET_SPECS


def get_real_audio_dataset_dir(dataset_name: str, root: Optional[Union[str, Path]] = None) -> Path:
    if not is_real_audio_dataset(dataset_name):
        raise ValueError(f"Unknown real audio dataset: {dataset_name}")
    return get_real_audio_root(root) / dataset_name


def get_real_audio_manifest_path(dataset_name: str, root: Optional[Union[str, Path]] = None) -> Path:
    return get_real_audio_dataset_dir(dataset_name, root) / "manifest.json"


def load_real_audio_manifest(dataset_name: str, root: Optional[Union[str, Path]] = None) -> dict:
    manifest_path = get_real_audio_manifest_path(dataset_name, root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared manifest missing for audio dataset '{dataset_name}': {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PREPARED_AUDIO_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prepared audio schema for '{dataset_name}': {payload.get('schema_version')}"
        )
    if payload.get("dataset") != dataset_name:
        raise ValueError(
            f"Manifest dataset mismatch: expected '{dataset_name}', got '{payload.get('dataset')}'"
        )
    return payload


def real_audio_dataset_is_prepared(
    dataset_name: str,
    root: Optional[Union[str, Path]] = None,
) -> bool:
    manifest_path = get_real_audio_manifest_path(dataset_name, root)
    if not manifest_path.exists():
        return False
    try:
        manifest = load_real_audio_manifest(dataset_name, root)
    except Exception:
        return False
    splits = manifest.get("splits", {})
    train_file = splits.get("train")
    val_file = splits.get("val")
    if not train_file or not val_file:
        return False
    dataset_dir = get_real_audio_dataset_dir(dataset_name, root)
    return (dataset_dir / train_file).exists() and (dataset_dir / val_file).exists()


def get_real_audio_n_classes(dataset_name: str, root: Optional[Union[str, Path]] = None) -> int:
    manifest = load_real_audio_manifest(dataset_name, root)
    n_classes = manifest.get("n_classes")
    if not isinstance(n_classes, int) or n_classes <= 0:
        raise ValueError(f"Manifest for '{dataset_name}' must define positive integer n_classes")
    return n_classes


def _resolve_split_file(dataset_name: str, split: str, root: Optional[Union[str, Path]] = None) -> Path:
    manifest = load_real_audio_manifest(dataset_name, root)
    split_file = manifest.get("splits", {}).get(split)
    if not split_file:
        raise ValueError(f"Manifest for '{dataset_name}' missing split '{split}'")
    return get_real_audio_dataset_dir(dataset_name, root) / split_file


class PreparedRealAudioDataset(Dataset):
    """Dataset wrapper for prepared real audio tensors in canonical feature shape."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        *,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.root = get_real_audio_root(root)
        self.manifest = load_real_audio_manifest(dataset_name, self.root)
        split_path = _resolve_split_file(dataset_name, split, self.root)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Prepared split missing for audio dataset '{dataset_name}' split '{split}': {split_path}"
            )
        payload = torch.load(split_path, map_location="cpu")
        self.labels = payload["labels"].to(dtype=torch.long)
        self.audio = payload["audio"].to(dtype=torch.float32)
        self._validate_payload()

    def _validate_payload(self) -> None:
        if self.labels.ndim != 1:
            raise ValueError("Prepared audio labels tensor must be 1D")
        if self.audio.ndim != 3:
            raise ValueError("Prepared audio tensor must be rank-3 [N, T, C]")
        if int(self.audio.shape[0]) != int(self.labels.shape[0]):
            raise ValueError("Prepared audio tensor and labels have mismatched lengths")
        if int(self.audio.shape[2]) != REAL_AUDIO_FEATURE_DIM:
            raise ValueError(
                f"Prepared audio feature dim {self.audio.shape[2]} != expected {REAL_AUDIO_FEATURE_DIM}"
            )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "audio": self.audio[idx],
            "label": self.labels[idx],
        }
