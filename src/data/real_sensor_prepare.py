"""Preparation helpers for real sensor datasets."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .real_sensor import (
    PREPARED_SENSOR_SCHEMA_VERSION,
    REAL_SENSOR_DATASET_SPECS,
    get_real_sensor_dataset_dir,
)


WISDM_SENSOR_DIRS = {
    "phone_accel": ("phone", "accel"),
    "phone_gyro": ("phone", "gyro"),
    "watch_accel": ("watch", "accel"),
    "watch_gyro": ("watch", "gyro"),
}


def _find_wisdm_raw_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir,
        raw_dir / "raw",
        raw_dir / "WISDM" / "raw",
        raw_dir / "wisdm-dataset" / "raw",
        raw_dir / "WISDM-dataset" / "raw",
    ]
    for candidate in candidates:
        if (candidate / "phone" / "accel").exists() and (candidate / "watch" / "gyro").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate WISDM raw directories. Expected phone/watch accel/gyro under "
        f"{raw_dir} or a nested raw/ directory."
    )


def _load_wisdm_activity_map(raw_root: Path) -> Dict[str, str]:
    candidates = [
        raw_root.parent / "activity_key.txt",
        raw_root.parent.parent / "activity_key.txt",
        raw_root / "activity_key.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            activity_map: Dict[str, str] = {}
            for line in candidate.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" in stripped:
                    code, name = stripped.split("=", 1)
                elif "," in stripped:
                    code, name = stripped.split(",", 1)
                else:
                    parts = stripped.split(None, 1)
                    if len(parts) != 2:
                        continue
                    code, name = parts
                activity_map[code.strip()] = name.strip()
            if activity_map:
                return activity_map
    return {}


def _parse_wisdm_line(line: str) -> Optional[Tuple[int, str, int, np.ndarray]]:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) != 6:
        return None
    subject_id = int(parts[0])
    activity_code = parts[1]
    timestamp = int(parts[2])
    values = np.asarray([float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
    return subject_id, activity_code, timestamp, values


def _resample_timeseries(values: np.ndarray, sequence_length: int) -> np.ndarray:
    if values.shape[0] == sequence_length:
        return values.astype(np.float32, copy=False)
    if values.shape[0] == 1:
        return np.repeat(values.astype(np.float32, copy=False), sequence_length, axis=0)
    base_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    channels = [np.interp(target_x, base_x, values[:, idx]) for idx in range(values.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def _vector_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32, copy=False)


def _build_wisdm_modalities(
    phone_accel: np.ndarray,
    phone_gyro: np.ndarray,
    watch_accel: np.ndarray,
    watch_gyro: np.ndarray,
) -> Dict[str, np.ndarray]:
    smartwatch = np.concatenate(
        [watch_accel, watch_gyro, _vector_norm(watch_accel)],
        axis=1,
    )
    room_env = np.concatenate(
        [phone_accel, _vector_norm(phone_accel)],
        axis=1,
    )
    accel_delta = watch_accel - phone_accel
    gyro_delta = watch_gyro - phone_gyro
    robot_arm = np.concatenate(
        [
            phone_accel,
            phone_gyro,
            _vector_norm(accel_delta),
            _vector_norm(gyro_delta),
            _vector_norm(watch_accel) - _vector_norm(phone_accel),
        ],
        axis=1,
    )
    return {
        "smartwatch": smartwatch.astype(np.float32, copy=False),
        "room_env": room_env.astype(np.float32, copy=False),
        "robot_arm": robot_arm.astype(np.float32, copy=False),
    }


def _encode_split_payload(
    samples: List[Dict[str, np.ndarray]],
    labels: List[int],
    stats: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, torch.Tensor]:
    payload = {
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    for modality in ("smartwatch", "room_env", "robot_arm"):
        stacked = np.stack([sample[modality] for sample in samples], axis=0)
        mean = stats[modality]["mean"]
        std = stats[modality]["std"]
        standardized = (stacked - mean) / std
        payload[modality] = torch.from_numpy(standardized.astype(np.float32, copy=False))
    return payload


def _compute_standardization(samples: List[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for modality in ("smartwatch", "room_env", "robot_arm"):
        stacked = np.stack([sample[modality] for sample in samples], axis=0)
        flattened = stacked.reshape(-1, stacked.shape[-1])
        mean = flattened.mean(axis=0, keepdims=True).reshape(1, 1, -1)
        std = flattened.std(axis=0, keepdims=True).reshape(1, 1, -1)
        std = np.where(std < 1e-6, 1.0, std)
        stats[modality] = {
            "mean": mean.astype(np.float32, copy=False),
            "std": std.astype(np.float32, copy=False),
        }
    return stats


def _group_wisdm_measurements(raw_root: Path) -> Dict[int, Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]]]:
    grouped: Dict[int, Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]]] = {}
    for sensor_key, (device, sensor_type) in WISDM_SENSOR_DIRS.items():
        sensor_dir = raw_root / device / sensor_type
        if not sensor_dir.exists():
            raise FileNotFoundError(f"Missing WISDM sensor directory: {sensor_dir}")
        for sensor_file in sorted(sensor_dir.glob("data_*_*.txt")):
            for line in sensor_file.read_text(encoding="utf-8").splitlines():
                parsed = _parse_wisdm_line(line)
                if parsed is None:
                    continue
                subject_id, activity_code, timestamp, values = parsed
                subject_bucket = grouped.setdefault(subject_id, {})
                activity_bucket = subject_bucket.setdefault(activity_code, {})
                sensor_bucket = activity_bucket.setdefault(sensor_key, [])
                sensor_bucket.append((timestamp, values))
    return grouped


def prepare_wisdm_dataset(raw_dir: Path, output_root: Path, sequence_length: int) -> Dict[str, object]:
    raw_root = _find_wisdm_raw_root(raw_dir.resolve())
    activity_map = _load_wisdm_activity_map(raw_root)
    grouped = _group_wisdm_measurements(raw_root)
    subjects = sorted(grouped)
    if not subjects:
        raise ValueError("No WISDM measurements were found in the provided raw directory")

    label_codes = sorted(
        {
            activity_code
            for activities in grouped.values()
            for activity_code in activities
        }
    )
    label_map = {code: idx for idx, code in enumerate(label_codes)}

    prepared_samples: List[Dict[str, object]] = []
    for subject_id in subjects:
        for activity_code, sensors in grouped[subject_id].items():
            if any(sensor_key not in sensors for sensor_key in WISDM_SENSOR_DIRS):
                continue
            aligned = {}
            for sensor_key in WISDM_SENSOR_DIRS:
                ordered = sorted(sensors[sensor_key], key=lambda item: item[0])
                raw_values = np.stack([values for _, values in ordered], axis=0)
                aligned[sensor_key] = _resample_timeseries(raw_values, sequence_length)
            prepared_samples.append(
                {
                    "subject_id": subject_id,
                    "activity_code": activity_code,
                    "label": label_map[activity_code],
                    "modalities": _build_wisdm_modalities(
                        aligned["phone_accel"],
                        aligned["phone_gyro"],
                        aligned["watch_accel"],
                        aligned["watch_gyro"],
                    ),
                }
            )

    if not prepared_samples:
        raise ValueError("No WISDM samples had all four required sensor streams")

    val_subject_count = max(1, round(len(subjects) * 0.2))
    val_subjects = set(subjects[-val_subject_count:])
    train_samples = [sample for sample in prepared_samples if sample["subject_id"] not in val_subjects]
    val_samples = [sample for sample in prepared_samples if sample["subject_id"] in val_subjects]
    if not train_samples or not val_samples:
        raise ValueError("WISDM prepared split would be empty; check subject partitioning")

    train_modalities = [sample["modalities"] for sample in train_samples]
    val_modalities = [sample["modalities"] for sample in val_samples]
    train_labels = [int(sample["label"]) for sample in train_samples]
    val_labels = [int(sample["label"]) for sample in val_samples]

    stats = _compute_standardization(train_modalities)
    train_payload = _encode_split_payload(train_modalities, train_labels, stats)
    val_payload = _encode_split_payload(val_modalities, val_labels, stats)

    dataset_dir = get_real_sensor_dataset_dir("wisdm", output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_payload, dataset_dir / "train.pt")
    torch.save(val_payload, dataset_dir / "val.pt")

    class_names = [activity_map.get(code, code) for code in label_codes]
    manifest = {
        "schema_version": PREPARED_SENSOR_SCHEMA_VERSION,
        "dataset": "wisdm",
        "source_name": REAL_SENSOR_DATASET_SPECS["wisdm"].source_name,
        "description": REAL_SENSOR_DATASET_SPECS["wisdm"].description,
        "raw_dir": str(raw_dir.resolve()),
        "sequence_length": sequence_length,
        "n_classes": len(label_codes),
        "class_names": class_names,
        "activity_codes": label_codes,
        "splits": {
            "train": "train.pt",
            "val": "val.pt",
        },
        "required_modalities": {
            "smartwatch": 7,
            "room_env": 4,
            "robot_arm": 9,
        },
        "projection_strategy": {
            "smartwatch": "watch accel xyz + watch gyro xyz + watch accel norm",
            "room_env": "phone accel xyz + phone accel norm (proxy environmental context)",
            "robot_arm": (
                "phone accel xyz + phone gyro xyz + accel delta norm + gyro delta norm + "
                "watch-vs-phone accel norm delta (proxy control stream)"
            ),
        },
        "split_strategy": {
            "type": "subject-holdout",
            "train_subjects": [int(subject) for subject in subjects[:-val_subject_count]],
            "val_subjects": [int(subject) for subject in subjects[-val_subject_count:]],
        },
        "sample_counts": {
            "all": len(prepared_samples),
            "train": len(train_samples),
            "val": len(val_samples),
        },
        "notes": [
            "WISDM does not naturally provide room_env or robot_arm semantics.",
            "This prepared projection uses phone/watch inertial streams as explicit proxy channels for the canonical modality contract.",
            "Claims from WISDM runs must be described as proxy canonical routing evidence, not direct ambient/robot equivalence.",
        ],
    }

    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dataset_card = (
        f"# {REAL_SENSOR_DATASET_SPECS['wisdm'].source_name} Prepared Dataset Card\n\n"
        f"Raw source:\n- {raw_dir.resolve()}\n\n"
        "Prepared projection:\n"
        "- smartwatch: watch accel xyz + watch gyro xyz + watch accel norm\n"
        "- room_env: phone accel xyz + phone accel norm\n"
        "- robot_arm: phone accel xyz + phone gyro xyz + cross-device delta features\n\n"
        "This is a proxy projection into the canonical sensor-streaming contract. "
        "It should be reported as such in experiments.\n"
    )
    (dataset_dir / "README.md").write_text(dataset_card, encoding="utf-8")

    return {
        "dataset": "wisdm",
        "output_dir": str(dataset_dir),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "n_classes": len(label_codes),
        "class_names": class_names,
        "manifest": str(dataset_dir / "manifest.json"),
    }
