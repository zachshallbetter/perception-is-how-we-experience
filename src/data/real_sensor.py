"""Prepared real-sensor dataset registry and loaders."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import torch
from torch.utils.data import Dataset

PREPARED_SENSOR_SCHEMA_VERSION = 1
DEFAULT_REAL_SENSOR_ROOT = Path("data/real-sensor")
REAL_SENSOR_MODALITY_DIMS = {
    "smartwatch": 7,
    "room_env": 4,
    "robot_arm": 9,
}


@dataclass(frozen=True)
class RealSensorDatasetSpec:
    dataset_id: str
    source_name: str
    description: str


REAL_SENSOR_DATASET_SPECS: Dict[str, RealSensorDatasetSpec] = {
    "wisdm": RealSensorDatasetSpec(
        dataset_id="wisdm",
        source_name="WISDM",
        description="Wearable / phone activity-recognition scaffold.",
    ),
    "pamap2": RealSensorDatasetSpec(
        dataset_id="pamap2",
        source_name="PAMAP2",
        description="Body-worn multimodal activity-monitoring scaffold.",
    ),
    "opportunity": RealSensorDatasetSpec(
        dataset_id="opportunity",
        source_name="OPPORTUNITY",
        description="Wearable + ambient + object sensing scaffold.",
    ),
    "uci_har": RealSensorDatasetSpec(
        dataset_id="uci_har",
        source_name="UCI HAR",
        description="Smartphone inertial activity recognition projected into the canonical sensor contract.",
    ),
    "hhar": RealSensorDatasetSpec(
        dataset_id="hhar",
        source_name="HHAR",
        description="Phone/watch heterogeneity activity recognition projected into the canonical sensor contract.",
    ),
    "mhealth": RealSensorDatasetSpec(
        dataset_id="mhealth",
        source_name="MHEALTH",
        description="Body-worn multimodal activity monitoring projected into the canonical sensor contract.",
    ),
    "selfback": RealSensorDatasetSpec(
        dataset_id="selfback",
        source_name="SELFBACK",
        description="Phone/watch-style accelerometer activity dataset projected into the canonical sensor contract.",
    ),
    "geomagnetic_wlan": RealSensorDatasetSpec(
        dataset_id="geomagnetic_wlan",
        source_name="Geo-magnetic + WLAN",
        description="Indoor localization dataset projected into the canonical sensor contract.",
    ),
    "realworld_har": RealSensorDatasetSpec(
        dataset_id="realworld_har",
        source_name="RealWorld HAR (HF)",
        description="HF-hosted wearable accelerometer activity benchmark projected into the canonical sensor contract.",
    ),
}


def get_real_sensor_root(root: Optional[Union[str, Path]] = None) -> Path:
    if root is not None:
        return Path(root)
    env_root = os.environ.get("SENSORY_REAL_SENSOR_ROOT")
    if env_root:
        return Path(env_root)
    return DEFAULT_REAL_SENSOR_ROOT


def get_real_sensor_dataset_ids() -> Iterable[str]:
    return REAL_SENSOR_DATASET_SPECS.keys()


def is_real_sensor_dataset(dataset_name: str) -> bool:
    return dataset_name in REAL_SENSOR_DATASET_SPECS


def get_real_sensor_dataset_dir(
    dataset_name: str, root: Optional[Union[str, Path]] = None
) -> Path:
    if not is_real_sensor_dataset(dataset_name):
        raise ValueError(f"Unknown real sensor dataset: {dataset_name}")
    return get_real_sensor_root(root) / dataset_name


def get_real_sensor_manifest_path(
    dataset_name: str, root: Optional[Union[str, Path]] = None
) -> Path:
    return get_real_sensor_dataset_dir(dataset_name, root) / "manifest.json"


def real_sensor_dataset_is_prepared(
    dataset_name: str,
    root: Optional[Union[str, Path]] = None,
) -> bool:
    manifest_path = get_real_sensor_manifest_path(dataset_name, root)
    if not manifest_path.exists():
        return False
    try:
        manifest = load_real_sensor_manifest(dataset_name, root)
    except Exception:
        return False
    splits = manifest.get("splits", {})
    train_file = splits.get("train")
    val_file = splits.get("val")
    if not train_file or not val_file:
        return False
    dataset_dir = get_real_sensor_dataset_dir(dataset_name, root)
    return (dataset_dir / train_file).exists() and (dataset_dir / val_file).exists()


def load_real_sensor_manifest(
    dataset_name: str, root: Optional[Union[str, Path]] = None
) -> dict:
    manifest_path = get_real_sensor_manifest_path(dataset_name, root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared manifest missing for dataset '{dataset_name}': {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PREPARED_SENSOR_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prepared sensor schema for '{dataset_name}': "
            f"{payload.get('schema_version')}"
        )
    if payload.get("dataset") != dataset_name:
        raise ValueError(
            f"Manifest dataset mismatch: expected '{dataset_name}', got '{payload.get('dataset')}'"
        )
    return payload


def get_real_sensor_n_classes(
    dataset_name: str, root: Optional[Union[str, Path]] = None
) -> int:
    manifest = load_real_sensor_manifest(dataset_name, root)
    n_classes = manifest.get("n_classes")
    if not isinstance(n_classes, int) or n_classes <= 0:
        raise ValueError(
            f"Manifest for '{dataset_name}' must define positive integer n_classes"
        )
    return n_classes


def _resolve_split_file(
    dataset_name: str, split: str, root: Optional[Union[str, Path]] = None
) -> Path:
    manifest = load_real_sensor_manifest(dataset_name, root)
    split_file = manifest.get("splits", {}).get(split)
    if not split_file:
        raise ValueError(f"Manifest for '{dataset_name}' missing split '{split}'")
    return get_real_sensor_dataset_dir(dataset_name, root) / split_file


class PreparedRealSensorDataset(Dataset):
    """Dataset wrapper for prepared real sensor tensors in canonical modality shape."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        *,
        merge_modalities: bool = False,
        sensor_variant: str = "baseline",
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.merge_modalities = merge_modalities
        self.sensor_variant = sensor_variant
        self.root = get_real_sensor_root(root)
        self.manifest = load_real_sensor_manifest(dataset_name, self.root)
        split_path = _resolve_split_file(dataset_name, split, self.root)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Prepared split missing for dataset '{dataset_name}' split '{split}': {split_path}"
            )
        payload = torch.load(split_path, map_location="cpu")
        self.labels = payload["labels"].to(dtype=torch.long)
        self.data = {
            modality: payload[modality].to(dtype=torch.float32)
            for modality in REAL_SENSOR_MODALITY_DIMS
        }
        self._validate_payload()
        self._apply_sensor_variant()

    def _validate_payload(self) -> None:
        n_sequences = int(self.labels.shape[0])
        if self.labels.ndim != 1:
            raise ValueError("Prepared labels tensor must be 1D")
        for modality, expected_dim in REAL_SENSOR_MODALITY_DIMS.items():
            tensor = self.data.get(modality)
            if tensor is None:
                raise ValueError(f"Prepared split missing modality tensor '{modality}'")
            if tensor.ndim != 3:
                raise ValueError(
                    f"Prepared modality '{modality}' must be rank-3 [N, T, C]"
                )
            if int(tensor.shape[0]) != n_sequences:
                raise ValueError(
                    f"Prepared modality '{modality}' has {tensor.shape[0]} sequences, expected {n_sequences}"
                )
            if int(tensor.shape[2]) != expected_dim:
                raise ValueError(
                    f"Prepared modality '{modality}' feature dim {tensor.shape[2]} != expected {expected_dim}"
                )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _apply_sensor_variant(self) -> None:
        if self.sensor_variant == "baseline":
            return
        if self.sensor_variant.startswith("drop_"):
            dropped_modality = self.sensor_variant.replace("drop_", "", 1)
            self.data[dropped_modality] = torch.zeros_like(self.data[dropped_modality])
            return
        if self.sensor_variant.startswith("noise_"):
            target = self.sensor_variant.replace("noise_", "", 1)
            self.data[target] = self._inject_noise_and_drift(target, self.data[target])

    def _inject_noise_and_drift(
        self, modality: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        sequence_length = tensor.shape[1]
        drift_scale = {"smartwatch": 0.45, "room_env": 2.5, "robot_arm": 0.3}[modality]
        noise_scale = {"smartwatch": 0.18, "room_env": 0.35, "robot_arm": 0.14}[
            modality
        ]
        drift = torch.linspace(0.0, 1.0, sequence_length).view(
            1, sequence_length, 1
        )
        direction = torch.randn(1, 1, tensor.shape[-1], dtype=tensor.dtype)
        return (
            tensor
            + noise_scale * torch.randn_like(tensor)
            + drift_scale * drift * direction
        )

    def __getitem__(self, idx: int) -> dict:
        if self.merge_modalities:
            merged = torch.cat(
                [
                    self.data["smartwatch"][idx],
                    self.data["room_env"][idx],
                    self.data["robot_arm"][idx],
                ],
                dim=-1,
            )
            return {"sensor_stream": merged, "label": self.labels[idx]}
        return {
            "smartwatch": self.data["smartwatch"][idx],
            "room_env": self.data["room_env"][idx],
            "robot_arm": self.data["robot_arm"][idx],
            "label": self.labels[idx],
        }
