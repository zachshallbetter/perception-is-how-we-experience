"""Preparation helpers for real sensor datasets."""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

PAMAP2_ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

SENSOR_MODALITIES = ("smartwatch", "room_env", "robot_arm")


def _resolve_candidate_root(raw_dir: Path, candidates: Sequence[str]) -> Path:
    for relative in candidates:
        candidate = raw_dir / relative if relative else raw_dir
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve raw dataset root under {raw_dir}")


def _bootstrap_realworld_har_raw(raw_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    files = [
        "downstream/har/realworld/x_train.npy",
        "downstream/har/realworld/y_train.npy",
        "downstream/har/realworld/x_valid.npy",
        "downstream/har/realworld/y_valid.npy",
    ]
    for filename in files:
        downloaded = hf_hub_download(
            repo_id="sha-ce/SensorData",
            repo_type="dataset",
            filename=filename,
        )
        target = raw_dir / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(downloaded, target)
    return raw_dir


def _resample_timeseries(values: np.ndarray, sequence_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == sequence_length:
        return values.astype(np.float32, copy=False)
    if values.shape[0] == 0:
        return np.zeros((sequence_length, values.shape[1]), dtype=np.float32)
    if values.shape[0] == 1:
        return np.repeat(values.astype(np.float32, copy=False), sequence_length, axis=0)
    base_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    channels = [
        np.interp(target_x, base_x, values[:, idx]) for idx in range(values.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def _vector_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32, copy=False)


def _encode_split_payload(
    samples: List[Dict[str, np.ndarray]],
    labels: List[int],
    stats: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, torch.Tensor]:
    payload = {"labels": torch.tensor(labels, dtype=torch.long)}
    for modality in SENSOR_MODALITIES:
        stacked = np.stack([sample[modality] for sample in samples], axis=0)
        mean = stats[modality]["mean"]
        std = stats[modality]["std"]
        standardized = (stacked - mean) / std
        payload[modality] = torch.from_numpy(
            standardized.astype(np.float32, copy=False)
        )
    return payload


def _compute_standardization(
    samples: List[Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for modality in SENSOR_MODALITIES:
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


def _window_array(
    values: np.ndarray, sequence_length: int, step: Optional[int] = None
) -> List[np.ndarray]:
    if values.shape[0] < sequence_length:
        return [_resample_timeseries(values, sequence_length)]
    stride = step if step is not None else max(1, sequence_length // 2)
    windows = []
    for start in range(0, values.shape[0] - sequence_length + 1, stride):
        windows.append(values[start : start + sequence_length])
    if not windows:
        windows.append(_resample_timeseries(values, sequence_length))
    return windows


def _sanitize_numeric(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if np.isnan(values).all():
        return np.zeros_like(values, dtype=np.float32)
    col_means = np.nanmean(values, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(values))
    if nan_rows.size:
        values[nan_rows, nan_cols] = col_means[nan_cols]
    return values.astype(np.float32, copy=False)


def _finalize_sensor_dataset(
    *,
    dataset_id: str,
    raw_dir: Path,
    output_root: Path,
    sequence_length: int,
    train_samples: List[Dict[str, np.ndarray]],
    train_labels: List[int],
    val_samples: List[Dict[str, np.ndarray]],
    val_labels: List[int],
    class_names: List[str],
    notes: List[str],
    projection_strategy: Dict[str, str],
    split_strategy: Dict[str, object],
) -> Dict[str, object]:
    if not train_samples or not val_samples:
        raise ValueError(f"{dataset_id} produced an empty train or val split")
    stats = _compute_standardization(train_samples)
    train_payload = _encode_split_payload(train_samples, train_labels, stats)
    val_payload = _encode_split_payload(val_samples, val_labels, stats)

    dataset_dir = get_real_sensor_dataset_dir(dataset_id, output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_payload, dataset_dir / "train.pt")
    torch.save(val_payload, dataset_dir / "val.pt")

    manifest = {
        "schema_version": PREPARED_SENSOR_SCHEMA_VERSION,
        "dataset": dataset_id,
        "source_name": REAL_SENSOR_DATASET_SPECS[dataset_id].source_name,
        "description": REAL_SENSOR_DATASET_SPECS[dataset_id].description,
        "raw_dir": str(raw_dir.resolve()),
        "sequence_length": sequence_length,
        "n_classes": len(class_names),
        "class_names": class_names,
        "splits": {"train": "train.pt", "val": "val.pt"},
        "required_modalities": {"smartwatch": 7, "room_env": 4, "robot_arm": 9},
        "projection_strategy": projection_strategy,
        "split_strategy": split_strategy,
        "sample_counts": {"train": len(train_samples), "val": len(val_samples)},
        "notes": notes,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    dataset_card = [
        f"# {REAL_SENSOR_DATASET_SPECS[dataset_id].source_name} Prepared Dataset Card",
        "",
        "Prepared projection:",
    ]
    for key, value in projection_strategy.items():
        dataset_card.append(f"- {key}: {value}")
    dataset_card += ["", "Notes:"] + [f"- {note}" for note in notes]
    (dataset_dir / "README.md").write_text(
        "\n".join(dataset_card) + "\n", encoding="utf-8"
    )

    return {
        "dataset": dataset_id,
        "output_dir": str(dataset_dir),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "n_classes": len(class_names),
        "class_names": class_names,
        "manifest": str(dataset_dir / "manifest.json"),
    }


def _build_wisdm_modalities(
    phone_accel: np.ndarray,
    phone_gyro: np.ndarray,
    watch_accel: np.ndarray,
    watch_gyro: np.ndarray,
) -> Dict[str, np.ndarray]:
    smartwatch = np.concatenate(
        [watch_accel, watch_gyro, _vector_norm(watch_accel)], axis=1
    )
    room_env = np.concatenate([phone_accel, _vector_norm(phone_accel)], axis=1)
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


def _find_wisdm_raw_root(raw_dir: Path) -> Path:
    candidates = ["", "raw", "WISDM/raw", "wisdm-dataset/raw", "WISDM-dataset/raw"]
    for candidate in candidates:
        root = raw_dir / candidate if candidate else raw_dir
        if (root / "phone" / "accel").exists() and (root / "watch" / "gyro").exists():
            return root
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
    values = np.asarray(
        [float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32
    )
    return subject_id, activity_code, timestamp, values


def _group_wisdm_measurements(
    raw_root: Path,
) -> Dict[int, Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]]]:
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


def prepare_wisdm_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    raw_root = _find_wisdm_raw_root(raw_dir.resolve())
    activity_map = _load_wisdm_activity_map(raw_root)
    grouped = _group_wisdm_measurements(raw_root)
    subjects = sorted(grouped)
    if not subjects:
        raise ValueError(
            "No WISDM measurements were found in the provided raw directory"
        )

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
                    "label": label_map[activity_code],
                    "modalities": _build_wisdm_modalities(
                        aligned["phone_accel"],
                        aligned["phone_gyro"],
                        aligned["watch_accel"],
                        aligned["watch_gyro"],
                    ),
                }
            )

    val_subject_count = max(1, round(len(subjects) * 0.2))
    val_subjects = set(subjects[-val_subject_count:])
    train_samples = [
        sample["modalities"]
        for sample in prepared_samples
        if sample["subject_id"] not in val_subjects
    ]
    val_samples = [
        sample["modalities"]
        for sample in prepared_samples
        if sample["subject_id"] in val_subjects
    ]
    train_labels = [
        int(sample["label"])
        for sample in prepared_samples
        if sample["subject_id"] not in val_subjects
    ]
    val_labels = [
        int(sample["label"])
        for sample in prepared_samples
        if sample["subject_id"] in val_subjects
    ]

    class_names = [activity_map.get(code, code) for code in label_codes]
    return _finalize_sensor_dataset(
        dataset_id="wisdm",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_names,
        notes=[
            "WISDM does not naturally provide room_env or robot_arm semantics.",
            "This prepared projection uses phone/watch inertial streams as explicit proxy channels for the canonical modality contract.",
            "Claims from WISDM runs must be described as proxy canonical routing evidence, not direct ambient/robot equivalence.",
        ],
        projection_strategy={
            "smartwatch": "watch accel xyz + watch gyro xyz + watch accel norm",
            "room_env": "phone accel xyz + phone accel norm (proxy environmental context)",
            "robot_arm": "phone accel xyz + phone gyro xyz + cross-device delta norms (proxy control stream)",
        },
        split_strategy={
            "type": "subject-holdout",
            "train_subjects": [
                int(subject) for subject in subjects[:-val_subject_count]
            ],
            "val_subjects": [int(subject) for subject in subjects[-val_subject_count:]],
        },
    )


def prepare_uci_har_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    candidates = [raw_dir.resolve(), raw_dir.resolve() / "UCI HAR Dataset"]
    root = None
    for candidate in candidates:
        if (candidate / "train" / "Inertial Signals" / "body_acc_x_train.txt").exists():
            root = candidate
            break
    if root is None:
        raise FileNotFoundError(
            f"Could not locate UCI HAR inertial signal files under {raw_dir}"
        )

    def load_signal_split(split: str, prefix: str) -> np.ndarray:
        signals_dir = root / split / "Inertial Signals"
        axes = [
            np.loadtxt(signals_dir / f"{prefix}_{axis}_{split}.txt", dtype=np.float32)
            for axis in ("x", "y", "z")
        ]
        return np.stack(axes, axis=2)

    train_body_acc = load_signal_split("train", "body_acc")
    train_body_gyro = load_signal_split("train", "body_gyro")
    train_total_acc = load_signal_split("train", "total_acc")
    test_body_acc = load_signal_split("test", "body_acc")
    test_body_gyro = load_signal_split("test", "body_gyro")
    test_total_acc = load_signal_split("test", "total_acc")
    gravity_train = train_total_acc - train_body_acc
    gravity_test = test_total_acc - test_body_acc

    def build_samples(
        body_acc: np.ndarray,
        body_gyro: np.ndarray,
        total_acc: np.ndarray,
        gravity: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        samples = []
        for idx in range(body_acc.shape[0]):
            smartwatch = np.concatenate(
                [body_acc[idx], body_gyro[idx], _vector_norm(body_acc[idx])], axis=1
            )
            room_env = np.concatenate(
                [total_acc[idx], _vector_norm(total_acc[idx])], axis=1
            )
            robot_arm = np.concatenate(
                [
                    body_gyro[idx],
                    gravity[idx],
                    _vector_norm(body_gyro[idx]),
                    _vector_norm(gravity[idx]),
                    _vector_norm(body_acc[idx]),
                ],
                axis=1,
            )
            samples.append(
                {"smartwatch": smartwatch, "room_env": room_env, "robot_arm": robot_arm}
            )
        return samples

    train_samples = build_samples(
        train_body_acc, train_body_gyro, train_total_acc, gravity_train
    )
    val_samples = build_samples(
        test_body_acc, test_body_gyro, test_total_acc, gravity_test
    )
    train_labels = (
        (np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1)
        .astype(int)
        .tolist()
    )
    val_labels = (
        (np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1)
        .astype(int)
        .tolist()
    )
    class_names = [
        "walking",
        "walking_upstairs",
        "walking_downstairs",
        "sitting",
        "standing",
        "laying",
    ]
    return _finalize_sensor_dataset(
        dataset_id="uci_har",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=train_body_acc.shape[1],
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_names,
        notes=[
            "UCI HAR is a smartphone-only benchmark; the canonical tri-modal contract is a proxy projection.",
            "Validation uses the dataset's native test split.",
        ],
        projection_strategy={
            "smartwatch": "body acceleration xyz + body acceleration norm",
            "room_env": "total acceleration xyz + total acceleration norm",
            "robot_arm": "body gyro xyz + gravity xyz + modality norms",
        },
        split_strategy={"type": "native-train-test"},
    )


def _load_csv_xyz(
    path: Path, value_keys: Tuple[str, str, str], group_keys: Tuple[str, str]
) -> Dict[Tuple[str, str], np.ndarray]:
    grouped: Dict[Tuple[str, str], List[List[float]]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get(group_keys[1], "").strip()
            user = row.get(group_keys[0], "").strip()
            if not user or not label or label == "null":
                continue
            grouped.setdefault((user, label), []).append(
                [float(row[key]) for key in value_keys]
            )
    return {
        key: np.asarray(values, dtype=np.float32) for key, values in grouped.items()
    }


def _ensure_hhar_activity_csvs(root: Path, raw_dir: Path) -> None:
    required = (
        "Phones_accelerometer.csv",
        "Phones_gyroscope.csv",
        "Watch_accelerometer.csv",
        "Watch_gyroscope.csv",
    )
    missing = [name for name in required if not (root / name).exists()]
    if not missing:
        return
    archive = raw_dir / "Activity recognition exp.zip"
    if not archive.exists():
        raise FileNotFoundError(
            f"HHAR raw root is missing {missing} and archive {archive} does not exist"
        )
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = {
            Path(info.filename).name: info.filename
            for info in zf.infolist()
            if Path(info.filename).name in missing
        }
        unresolved = [name for name in missing if name not in members]
        if unresolved:
            raise FileNotFoundError(
                f"HHAR archive {archive} does not contain required files {unresolved}"
            )
        for name in missing:
            source = members[name]
            with zf.open(source) as src, (root / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_hhar_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = _resolve_candidate_root(
        raw_dir.resolve(), ["activity_recognition/Activity recognition exp", ""]
    )
    _ensure_hhar_activity_csvs(root, raw_dir.resolve())
    phone_acc = _load_csv_xyz(
        root / "Phones_accelerometer.csv", ("x", "y", "z"), ("User", "gt")
    )
    phone_gyro = _load_csv_xyz(
        root / "Phones_gyroscope.csv", ("x", "y", "z"), ("User", "gt")
    )
    watch_acc = _load_csv_xyz(
        root / "Watch_accelerometer.csv", ("x", "y", "z"), ("User", "gt")
    )
    watch_gyro = _load_csv_xyz(
        root / "Watch_gyroscope.csv", ("x", "y", "z"), ("User", "gt")
    )
    keys = sorted(set(phone_acc) & set(phone_gyro) & set(watch_acc) & set(watch_gyro))
    if not keys:
        raise ValueError("HHAR did not yield any aligned phone/watch activity groups")
    class_codes = sorted({label for _, label in keys})
    class_map = {label: idx for idx, label in enumerate(class_codes)}
    users = sorted({user for user, _ in keys})
    val_users = set(users[-max(1, round(len(users) * 0.2)) :])
    train_samples: List[Dict[str, np.ndarray]] = []
    val_samples: List[Dict[str, np.ndarray]] = []
    train_labels: List[int] = []
    val_labels: List[int] = []
    for user, label in keys:
        min_len = min(
            len(phone_acc[(user, label)]),
            len(phone_gyro[(user, label)]),
            len(watch_acc[(user, label)]),
            len(watch_gyro[(user, label)]),
        )
        if min_len < 16:
            continue
        pa = phone_acc[(user, label)][:min_len]
        pg = phone_gyro[(user, label)][:min_len]
        wa = watch_acc[(user, label)][:min_len]
        wg = watch_gyro[(user, label)][:min_len]
        for w_pa, w_pg, w_wa, w_wg in zip(
            _window_array(pa, sequence_length),
            _window_array(pg, sequence_length),
            _window_array(wa, sequence_length),
            _window_array(wg, sequence_length),
        ):
            sample = {
                "smartwatch": np.concatenate([w_wa, w_wg, _vector_norm(w_wa)], axis=1),
                "room_env": np.concatenate([w_pa, _vector_norm(w_pa)], axis=1),
                "robot_arm": np.concatenate([w_pa, w_pg, w_wa - w_pa], axis=1),
            }
            if user in val_users:
                val_samples.append(sample)
                val_labels.append(class_map[label])
            else:
                train_samples.append(sample)
                train_labels.append(class_map[label])
    return _finalize_sensor_dataset(
        dataset_id="hhar",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_codes,
        notes=[
            "HHAR provides phone and watch inertial streams but no ambient or robot channels.",
            "Canonical room_env and robot_arm channels are explicit proxy projections derived from phone/watch dynamics.",
        ],
        projection_strategy={
            "smartwatch": "watch accel xyz + watch gyro xyz + watch accel norm",
            "room_env": "phone accel xyz + phone accel norm",
            "robot_arm": "phone accel xyz + phone gyro xyz + watch-minus-phone accel xyz",
        },
        split_strategy={"type": "user-holdout", "val_users": sorted(val_users)},
    )


def _contiguous_windows(
    values: np.ndarray, labels: np.ndarray, sequence_length: int
) -> List[Tuple[np.ndarray, int]]:
    windows: List[Tuple[np.ndarray, int]] = []
    start = 0
    while start < len(labels):
        label = int(labels[start])
        end = start + 1
        while end < len(labels) and int(labels[end]) == label:
            end += 1
        if label > 0:
            segment = values[start:end]
            for window in _window_array(segment, sequence_length):
                windows.append((window, label))
        start = end
    return windows


def prepare_mhealth_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = _resolve_candidate_root(raw_dir.resolve(), ["MHEALTHDATASET", ""])
    files = sorted(root.glob("mHealth_subject*.log"))
    if not files:
        raise FileNotFoundError(f"No MHEALTH subject logs found under {root}")
    subjects = [path.stem.replace("mHealth_subject", "") for path in files]
    val_subjects = set(subjects[-max(1, round(len(subjects) * 0.2)) :])
    train_samples: List[Dict[str, np.ndarray]] = []
    val_samples: List[Dict[str, np.ndarray]] = []
    train_labels: List[int] = []
    val_labels: List[int] = []
    class_ids: set[int] = set()
    for path in files:
        subject = path.stem.replace("mHealth_subject", "")
        data = np.loadtxt(path, dtype=np.float32)
        labels = data[:, -1].astype(np.int64)
        class_ids.update(int(label) for label in labels if label > 0)
        chest_acc = data[:, 0:3]
        chest_ecg = data[:, 3:5]
        ankle = data[:, 5:14]
        wrist_acc = data[:, 14:17]
        wrist_gyro = data[:, 17:20]
        wrist_mag = data[:, 20:23]
        stacked = np.concatenate(
            [
                np.concatenate(
                    [wrist_acc, wrist_gyro, _vector_norm(wrist_acc)], axis=1
                ),
                np.concatenate(
                    [chest_acc, chest_ecg.mean(axis=1, keepdims=True)], axis=1
                ),
                ankle,
            ],
            axis=1,
        )
        for window, label in _contiguous_windows(stacked, labels, sequence_length):
            sample = {
                "smartwatch": window[:, 0:7],
                "room_env": window[:, 7:11],
                "robot_arm": window[:, 11:20],
            }
            target_samples = val_samples if subject in val_subjects else train_samples
            target_labels = val_labels if subject in val_subjects else train_labels
            target_samples.append(sample)
            target_labels.append(label - 1)
    class_names = [f"activity_{class_id}" for class_id in sorted(class_ids)]
    return _finalize_sensor_dataset(
        dataset_id="mhealth",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_names,
        notes=[
            "MHEALTH supplies chest, wrist, and ankle streams that map cleanly into the canonical contract as a proxy body-sensing benchmark.",
        ],
        projection_strategy={
            "smartwatch": "right wrist accel xyz + gyro xyz + wrist accel norm",
            "room_env": "chest accel xyz + mean ECG proxy",
            "robot_arm": "left ankle accel xyz + gyro xyz + magnetic xyz",
        },
        split_strategy={
            "type": "subject-holdout",
            "val_subjects": sorted(val_subjects),
        },
    )


def prepare_pamap2_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    raw_root = raw_dir.resolve()
    protocol_candidates = [
        raw_root / "Protocol",
        raw_root / "PAMAP2_Dataset" / "Protocol",
        raw_root,
    ]
    protocol_dir = None
    for candidate in protocol_candidates:
        if candidate.exists() and any(candidate.glob("subject*.dat")):
            protocol_dir = candidate
            break
    if protocol_dir is None:
        protocol_dir = raw_root / "Protocol"
    files = sorted(protocol_dir.glob("subject*.dat"))
    if not files:
        raise FileNotFoundError(f"No PAMAP2 protocol files found under {protocol_dir}")
    subject_ids = [path.stem.replace("subject", "") for path in files]
    val_subjects = set(subject_ids[-max(1, round(len(subject_ids) * 0.2)) :])
    class_ids: set[int] = set()
    train_samples: List[Dict[str, np.ndarray]] = []
    val_samples: List[Dict[str, np.ndarray]] = []
    train_labels: List[int] = []
    val_labels: List[int] = []
    for path in files:
        subject = path.stem.replace("subject", "")
        data = _sanitize_numeric(np.genfromtxt(path, dtype=np.float32))
        labels = data[:, 1].astype(np.int64)
        class_ids.update(int(label) for label in labels if label > 0)
        heart_rate = np.nan_to_num(data[:, 2:3], nan=0.0)
        hand_acc = data[:, 4:7]
        hand_gyro = data[:, 10:13]
        chest_acc = data[:, 21:24]
        ankle_acc = data[:, 38:41]
        ankle_gyro = data[:, 44:47]
        ankle_mag = data[:, 47:50]
        stacked = np.concatenate(
            [
                np.concatenate([hand_acc, hand_gyro, _vector_norm(hand_acc)], axis=1),
                np.concatenate([chest_acc, heart_rate], axis=1),
                np.concatenate([ankle_acc, ankle_gyro, ankle_mag], axis=1),
            ],
            axis=1,
        )
        for window, label in _contiguous_windows(stacked, labels, sequence_length):
            sample = {
                "smartwatch": window[:, 0:7],
                "room_env": window[:, 7:11],
                "robot_arm": window[:, 11:20],
            }
            target_samples = val_samples if subject in val_subjects else train_samples
            target_labels = val_labels if subject in val_subjects else train_labels
            target_samples.append(sample)
            target_labels.append(label - 1)
    sorted_class_ids = sorted(class_ids)
    label_map = {
        class_id: mapped_id for mapped_id, class_id in enumerate(sorted_class_ids)
    }
    train_labels = [label_map[label + 1] for label in train_labels]
    val_labels = [label_map[label + 1] for label in val_labels]
    class_names = [
        PAMAP2_ACTIVITY_NAMES.get(class_id, f"activity_{class_id}")
        for class_id in sorted_class_ids
    ]
    return _finalize_sensor_dataset(
        dataset_id="pamap2",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_names,
        notes=[
            "PAMAP2 provides hand, chest, and ankle/body sensors and is projected into the canonical contract with heart rate folded into room_env.",
        ],
        projection_strategy={
            "smartwatch": "hand IMU acc16g xyz + gyro xyz + hand accel norm",
            "room_env": "chest IMU acc16g xyz + heart rate",
            "robot_arm": "ankle IMU acc16g xyz + gyro xyz + magnetic xyz",
        },
        split_strategy={
            "type": "subject-holdout",
            "val_subjects": sorted(val_subjects),
        },
    )


def prepare_opportunity_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = _resolve_candidate_root(
        raw_dir.resolve(),
        ["OpportunityUCIDataset/dataset", "dataset", "OpportunityUCIDataset", ""],
    )
    dataset_dir = root if root.name == "dataset" else root / "dataset"
    files = sorted(dataset_dir.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"No OPPORTUNITY data files found under {dataset_dir}")
    val_files = {path.name for path in files[-max(1, round(len(files) * 0.2)) :]}
    class_ids: set[int] = set()
    train_samples: List[Dict[str, np.ndarray]] = []
    val_samples: List[Dict[str, np.ndarray]] = []
    train_labels: List[int] = []
    val_labels: List[int] = []
    for path in files:
        data = _sanitize_numeric(np.genfromtxt(path, dtype=np.float32))
        labels = data[:, 243].astype(np.int64)
        class_ids.update(int(label) for label in labels if label > 0)
        smartwatch = np.concatenate(
            [data[:, 22:25], data[:, 31:34], _vector_norm(data[:, 22:25])], axis=1
        )
        room_env = np.concatenate(
            [data[:, 16:19], _vector_norm(data[:, 16:19])], axis=1
        )
        robot_arm = np.concatenate(
            [data[:, 7:10], data[:, 25:28], data[:, 34:37]], axis=1
        )
        stacked = np.concatenate([smartwatch, room_env, robot_arm], axis=1)
        for window, label in _contiguous_windows(stacked, labels, sequence_length):
            sample = {
                "smartwatch": window[:, 0:7],
                "room_env": window[:, 7:11],
                "robot_arm": window[:, 11:20],
            }
            target_samples = val_samples if path.name in val_files else train_samples
            target_labels = val_labels if path.name in val_files else train_labels
            target_samples.append(sample)
            target_labels.append(sorted(class_ids).index(int(label)))
    class_id_order = sorted(class_ids)
    class_names = [f"locomotion_{class_id}" for class_id in class_id_order]
    return _finalize_sensor_dataset(
        dataset_id="opportunity",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_names,
        notes=[
            "OPPORTUNITY is used here with the locomotion label column as the supervised target.",
            "Wearable/body accelerometer channels are projected into the canonical sensor contract; the benchmark remains a proxy for broader ambient fusion.",
        ],
        projection_strategy={
            "smartwatch": "right-wrist acc xyz + left-wrist acc xyz + right-wrist accel norm",
            "room_env": "back acc xyz + back accel norm",
            "robot_arm": "left upper arm acc xyz + right upper arm acc xyz + right hand acc xyz",
        },
        split_strategy={"type": "file-holdout", "val_files": sorted(val_files)},
    )


def prepare_selfback_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = _resolve_candidate_root(raw_dir.resolve(), ["selfBACK", ""])
    paired_trace_root = root / "wt"
    if not paired_trace_root.exists():
        phone_root = root / "t"
        watch_root = root / "w"
        activity_dirs = sorted(
            [
                path.name
                for path in phone_root.iterdir()
                if path.is_dir() and not path.name.startswith("._")
            ]
        )
        class_map = {activity: idx for idx, activity in enumerate(activity_dirs)}
        train_samples: List[Dict[str, np.ndarray]] = []
        val_samples: List[Dict[str, np.ndarray]] = []
        train_labels: List[int] = []
        val_labels: List[int] = []
        all_file_ids: List[str] = []
        paired_files: List[Tuple[str, Path, Path]] = []
        for activity in activity_dirs:
            for phone_file in sorted((phone_root / activity).glob("*.csv")):
                if phone_file.name.startswith("._"):
                    continue
                watch_file = watch_root / activity / phone_file.name
                if not watch_file.exists() or watch_file.name.startswith("._"):
                    continue
                paired_files.append((activity, phone_file, watch_file))
                all_file_ids.append(phone_file.stem)
        unique_file_ids = sorted(set(all_file_ids))
        val_ids = set(unique_file_ids[-max(1, round(len(unique_file_ids) * 0.2)) :])
        for activity, phone_file, watch_file in paired_files:
            phone = np.genfromtxt(
                phone_file,
                delimiter=",",
                skip_header=1,
                usecols=(1, 2, 3),
                dtype=np.float32,
            )
            watch = np.genfromtxt(
                watch_file,
                delimiter=",",
                skip_header=1,
                usecols=(1, 2, 3),
                dtype=np.float32,
            )
            min_len = min(len(phone), len(watch))
            if min_len < 16:
                continue
            phone = phone[:min_len]
            watch = watch[:min_len]
            for p_win, w_win in zip(
                _window_array(phone, sequence_length), _window_array(watch, sequence_length)
            ):
                sample = {
                    "smartwatch": np.concatenate(
                        [w_win, p_win, _vector_norm(w_win)], axis=1
                    ),
                    "room_env": np.concatenate([p_win, _vector_norm(p_win)], axis=1),
                    "robot_arm": np.concatenate([p_win, w_win, w_win - p_win], axis=1),
                }
                if phone_file.stem in val_ids:
                    val_samples.append(sample)
                    val_labels.append(class_map[activity])
                else:
                    train_samples.append(sample)
                    train_labels.append(class_map[activity])
        class_names = activity_dirs
        split_strategy = {"type": "file-holdout", "val_file_ids": sorted(val_ids)}
        notes = [
            "SELFBACK provides phone/watch-style accelerometer streams only; room_env and robot_arm are explicit proxy projections.",
            "MacOS resource-fork files prefixed with '._' are ignored during preparation.",
        ]
    else:
        trace_files = sorted(
            [
                path
                for path in paired_trace_root.iterdir()
                if path.is_file() and not path.name.startswith(".")
            ]
        )
        activities = sorted(
            {
                file_path.name.split("_", 1)[1]
                for file_path in trace_files
                if "_" in file_path.name
            }
        )
        class_map = {activity: idx for idx, activity in enumerate(activities)}
        train_samples = []
        val_samples = []
        train_labels = []
        val_labels = []
        subject_ids = sorted(
            {
                file_path.name.split("_", 1)[0]
                for file_path in trace_files
                if "_" in file_path.name
            }
        )
        val_ids = set(subject_ids[-max(1, round(len(subject_ids) * 0.2)) :])
        for trace_file in trace_files:
            if "_" not in trace_file.name:
                continue
            subject_id, activity = trace_file.name.split("_", 1)
            if activity not in class_map:
                continue
            paired = np.genfromtxt(trace_file, delimiter=",", dtype=np.float32)
            if paired.ndim == 1:
                paired = paired.reshape(1, -1)
            if paired.shape[0] < 16 or paired.shape[1] < 6:
                continue
            phone = paired[:, 0:3]
            watch = paired[:, 3:6]
            min_len = min(len(phone), len(watch))
            if min_len < 16:
                continue
            phone = phone[:min_len]
            watch = watch[:min_len]
            for p_win, w_win in zip(
                _window_array(phone, sequence_length), _window_array(watch, sequence_length)
            ):
                sample = {
                    "smartwatch": np.concatenate(
                        [w_win, p_win, _vector_norm(w_win)], axis=1
                    ),
                    "room_env": np.concatenate([p_win, _vector_norm(p_win)], axis=1),
                    "robot_arm": np.concatenate([p_win, w_win, w_win - p_win], axis=1),
                }
                if subject_id in val_ids:
                    val_samples.append(sample)
                    val_labels.append(class_map[activity])
                else:
                    train_samples.append(sample)
                    train_labels.append(class_map[activity])
        class_names = activities
        split_strategy = {"type": "subject-holdout", "val_subject_ids": sorted(val_ids)}
        notes = [
            "SELFBACK wt layout provides paired six-axis traces only; room_env and robot_arm are explicit proxy projections.",
            "MacOS resource-fork files prefixed with '._' are ignored during preparation.",
        ]
    return _finalize_sensor_dataset(
        dataset_id="selfback",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=class_names,
        notes=notes,
        projection_strategy={
            "smartwatch": "watch accel xyz + phone accel xyz + watch accel norm",
            "room_env": "phone accel xyz + phone accel norm",
            "robot_arm": "phone accel xyz + watch accel xyz + watch-minus-phone accel xyz",
        },
        split_strategy=split_strategy,
    )


def prepare_geomagnetic_wlan_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = raw_dir.resolve()
    wifi_by_place: Dict[int, np.ndarray] = {}
    train_samples: List[Dict[str, np.ndarray]] = []
    val_samples: List[Dict[str, np.ndarray]] = []
    train_labels: List[int] = []
    val_labels: List[int] = []
    place_ids: set[int] = set()
    for measure in (1, 2):
        phone_name = (
            f"measure{measure}_smartphone_sens.csv"
            if measure == 1
            else f"measure{measure}_phone_sens.csv"
        )
        phone_sens = np.genfromtxt(
            root / phone_name, delimiter=",", skip_header=1, dtype=np.float32
        )
        watch_name = (
            f"measure{measure}_smartwatch_sens.csv"
            if measure == 1
            else f"measure{measure}_watch_sens.csv"
        )
        watch_sens = np.genfromtxt(
            root / watch_name, delimiter=",", skip_header=1, dtype=np.float32
        )
        phone_wifi = np.genfromtxt(
            root / f"measure{measure}_smartphone_wifi.csv",
            delimiter=",",
            dtype=np.float32,
        )
        intervals = np.genfromtxt(
            root / f"measure{measure}_timestamp_id.csv", delimiter=",", dtype=np.int64
        )
        for row in phone_wifi:
            place_id = int(row[0])
            wifi_by_place[place_id] = row[1:]
        for start_ts, end_ts, place_id in intervals:
            phone_mask = (phone_sens[:, 0] >= start_ts) & (phone_sens[:, 0] <= end_ts)
            watch_mask = (watch_sens[:, 0] >= start_ts) & (watch_sens[:, 0] <= end_ts)
            if (
                int(place_id) not in wifi_by_place
                or not phone_mask.any()
                or not watch_mask.any()
            ):
                continue
            place_ids.add(int(place_id))
            phone_rows = phone_sens[phone_mask][:, 1:13]
            watch_rows = watch_sens[watch_mask][:, 1:13]
            phone_rows = _resample_timeseries(phone_rows, sequence_length)
            watch_rows = _resample_timeseries(watch_rows, sequence_length)
            wifi = wifi_by_place[int(place_id)]
            top4 = np.sort(wifi)[-4:].astype(np.float32) / 100.0
            room_env = np.repeat(top4.reshape(1, -1), sequence_length, axis=0)
            sample = {
                "smartwatch": np.concatenate(
                    [
                        watch_rows[:, 0:3],
                        watch_rows[:, 9:12],
                        _vector_norm(watch_rows[:, 0:3]),
                    ],
                    axis=1,
                ),
                "room_env": room_env,
                "robot_arm": np.concatenate(
                    [phone_rows[:, 0:3], phone_rows[:, 9:12], phone_rows[:, 3:6]],
                    axis=1,
                ),
            }
            if measure == 1:
                train_samples.append(sample)
                train_labels.append(int(place_id))
            else:
                val_samples.append(sample)
                val_labels.append(int(place_id))
    place_id_order = sorted(place_ids)
    remap = {place_id: idx for idx, place_id in enumerate(place_id_order)}
    train_labels = [remap[label] for label in train_labels]
    val_labels = [remap[label] for label in val_labels]
    return _finalize_sensor_dataset(
        dataset_id="geomagnetic_wlan",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=train_labels,
        val_samples=val_samples,
        val_labels=val_labels,
        class_names=[f"place_{place_id}" for place_id in place_id_order],
        notes=[
            "Geo-magnetic + WLAN is treated as an indoor localization benchmark.",
            "room_env is a repeated top-4 Wi-Fi RSSI summary per place interval; smartwatch and robot_arm are projected from watch/phone sensor streams.",
        ],
        projection_strategy={
            "smartwatch": "watch accel xyz + watch gyro xyz + watch accel norm",
            "room_env": "top-4 Wi-Fi RSSI values repeated across the interval window",
            "robot_arm": "phone accel xyz + phone gyro xyz + phone magnetic xyz",
        },
        split_strategy={
            "type": "measure-holdout",
            "train_measure": 1,
            "val_measure": 2,
        },
    )


def prepare_realworld_har_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    resolved_raw = raw_dir.resolve()
    required = ["x_train.npy", "y_train.npy", "x_valid.npy", "y_valid.npy"]
    if not all((resolved_raw / name).exists() for name in required):
        resolved_raw = _bootstrap_realworld_har_raw(resolved_raw)

    x_train = np.load(resolved_raw / "x_train.npy").astype(np.float32)
    y_train = np.load(resolved_raw / "y_train.npy").astype(np.int64)
    x_val = np.load(resolved_raw / "x_valid.npy").astype(np.float32)
    y_val = np.load(resolved_raw / "y_valid.npy").astype(np.int64)

    def build_sample(window: np.ndarray) -> Dict[str, np.ndarray]:
        if window.shape[0] == 3 and window.shape[1] != 3:
            window = window.transpose(1, 0)
        window = _resample_timeseries(window, sequence_length)
        delta = np.diff(window, axis=0, prepend=window[:1])
        sq = np.square(window, dtype=np.float32)
        return {
            "smartwatch": np.concatenate([window, delta, _vector_norm(window)], axis=1),
            "room_env": np.concatenate([window, _vector_norm(window)], axis=1),
            "robot_arm": np.concatenate([window, delta, sq], axis=1),
        }

    train_samples = [build_sample(sample) for sample in x_train]
    val_samples = [build_sample(sample) for sample in x_val]
    max_label = int(
        max(
            int(np.max(y_train)) if y_train.size else 0,
            int(np.max(y_val)) if y_val.size else 0,
        )
    )
    class_names = [f"activity_{idx}" for idx in range(max_label + 1)]
    return _finalize_sensor_dataset(
        dataset_id="realworld_har",
        raw_dir=resolved_raw,
        output_root=output_root,
        sequence_length=sequence_length,
        train_samples=train_samples,
        train_labels=[int(label) for label in y_train],
        val_samples=val_samples,
        val_labels=[int(label) for label in y_val],
        class_names=class_names,
        notes=[
            "Prepared from the HF sha-ce/SensorData realworld HAR split.",
            "The source is single-stream accelerometer data; room_env and robot_arm remain explicit proxy projections to preserve the canonical sensor contract.",
        ],
        projection_strategy={
            "smartwatch": "accelerometer xyz + timestep delta xyz + accel norm",
            "room_env": "accelerometer xyz + accel norm",
            "robot_arm": "accelerometer xyz + timestep delta xyz + squared accel xyz",
        },
        split_strategy={
            "type": "hf-downstream-holdout",
            "train_split": "x_train/y_train",
            "val_split": "x_valid/y_valid",
        },
    )


def prepare_real_sensor_dataset(
    dataset: str, raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    if dataset == "wisdm":
        return prepare_wisdm_dataset(raw_dir, output_root, sequence_length)
    if dataset == "pamap2":
        return prepare_pamap2_dataset(raw_dir, output_root, sequence_length)
    if dataset == "opportunity":
        return prepare_opportunity_dataset(raw_dir, output_root, sequence_length)
    if dataset == "uci_har":
        return prepare_uci_har_dataset(raw_dir, output_root, sequence_length)
    if dataset == "hhar":
        return prepare_hhar_dataset(raw_dir, output_root, sequence_length)
    if dataset == "mhealth":
        return prepare_mhealth_dataset(raw_dir, output_root, sequence_length)
    if dataset == "selfback":
        return prepare_selfback_dataset(raw_dir, output_root, sequence_length)
    if dataset == "geomagnetic_wlan":
        return prepare_geomagnetic_wlan_dataset(raw_dir, output_root, sequence_length)
    if dataset == "realworld_har":
        return prepare_realworld_har_dataset(raw_dir, output_root, sequence_length)
    raise ValueError(f"Unsupported real sensor dataset preparation target: {dataset}")
