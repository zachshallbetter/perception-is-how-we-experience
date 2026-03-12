"""Preparation helpers for real audio datasets."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
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


def _import_scipy_wavfile():
    from scipy.io import wavfile

    return wavfile


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


def _find_esc50_root(raw_dir: Path) -> Path:
    candidates = [raw_dir, raw_dir / "ESC-50-master", raw_dir.parent / "ESC-50-master"]
    for candidate in candidates:
        if (candidate / "meta" / "esc50.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate ESC-50 under {raw_dir}")


def _find_urbansound_root(raw_dir: Path) -> Path:
    candidates = [raw_dir, raw_dir / "UrbanSound8K", raw_dir.parent / "UrbanSound8K"]
    for candidate in candidates:
        if (candidate / "metadata" / "UrbanSound8K.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate UrbanSound8K under {raw_dir}")


def _find_fsd50k_root(raw_dir: Path) -> Path:
    candidates = [raw_dir, raw_dir / "fsd50k", raw_dir.parent / "fsd50k"]
    for candidate in candidates:
        if (candidate / "ground_truth" / "dev.csv").exists() and any(
            (candidate / "data").glob("*.parquet")
        ):
            return candidate
    raise FileNotFoundError(f"Could not locate FSD50K under {raw_dir}")


def _bootstrap_fsd50k_raw(raw_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="philgzl/fsd50k",
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=["data/*.parquet", "ground_truth/*.csv", "README.md"],
    )
    return raw_dir


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
        spec = torch.nn.functional.pad(
            spec, (0, 0, 0, REAL_AUDIO_FEATURE_DIM - spec.shape[0])
        )
    else:
        spec = spec[:REAL_AUDIO_FEATURE_DIM, :]
    spec = torch.log1p(spec).transpose(0, 1)
    if spec.shape[0] == target_frames:
        return spec
    spec = spec.unsqueeze(0).transpose(1, 2)
    spec = torch.nn.functional.interpolate(
        spec, size=target_frames, mode="linear", align_corners=False
    )
    return spec.transpose(1, 2).squeeze(0)


def _standardize_audio(
    train_audio: torch.Tensor, val_audio: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    flattened = train_audio.reshape(-1, train_audio.shape[-1])
    mean = flattened.mean(dim=0, keepdim=True).reshape(1, 1, -1)
    std = flattened.std(dim=0, keepdim=True).reshape(1, 1, -1)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (train_audio - mean) / std, (val_audio - mean) / std


def _decode_with_ffmpeg(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"ffmpeg decoded empty waveform for {path}")
    return waveform


def _decode_audio_bytes(payload: bytes, suffix: str) -> np.ndarray:
    try:
        waveform, _sample_rate = sf.read(BytesIO(payload), dtype="float32")
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.size == 0:
            raise ValueError("decoded empty waveform from in-memory payload")
        return waveform
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
            temp_file.write(payload)
            temp_file.flush()
            return _decode_with_ffmpeg(Path(temp_file.name))


def _decode_wave_file(path: Path) -> np.ndarray:
    try:
        wavfile = _import_scipy_wavfile()
        _sample_rate, waveform = wavfile.read(str(path))
        waveform = np.asarray(waveform)
        if waveform.ndim == 2:
            waveform = waveform.astype(np.float32).mean(axis=1)
        else:
            waveform = waveform.astype(np.float32)
        if waveform.dtype == np.uint8:
            waveform = (waveform / 128.0) - 1.0
        elif np.issubdtype(waveform.dtype, np.integer):
            max_abs = (
                float(np.iinfo(waveform.dtype).max)
                if waveform.dtype != np.uint8
                else 128.0
            )
            waveform = waveform / max_abs
        return np.asarray(waveform, dtype=np.float32)
    except Exception:
        return _decode_with_ffmpeg(path)


def _encode_audio_examples(
    examples: Iterable[Tuple[Path, int]], *, sequence_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    features: List[torch.Tensor] = []
    labels: List[int] = []
    for path, label in examples:
        waveform = _decode_wave_file(path)
        waveform_tensor = _resample_waveform(waveform, sequence_length)
        features.append(_waveform_to_log_spectrogram(waveform_tensor))
        labels.append(int(label))
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def _write_audio_dataset(
    *,
    dataset_id: str,
    raw_dir: Path,
    output_root: Path,
    sequence_length: int,
    train_examples: List[Tuple[Path, int]],
    val_examples: List[Tuple[Path, int]],
    class_names: List[str],
    notes: List[str],
    split_strategy: Dict[str, object],
) -> Dict[str, object]:
    if not train_examples or not val_examples:
        raise ValueError(f"{dataset_id} produced an empty train or val split")
    train_audio, train_labels = _encode_audio_examples(
        train_examples, sequence_length=sequence_length
    )
    val_audio, val_labels = _encode_audio_examples(
        val_examples, sequence_length=sequence_length
    )
    train_audio, val_audio = _standardize_audio(train_audio, val_audio)

    dataset_dir = get_real_audio_dataset_dir(dataset_id, output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"audio": train_audio, "labels": train_labels}, dataset_dir / "train.pt")
    torch.save({"audio": val_audio, "labels": val_labels}, dataset_dir / "val.pt")

    manifest = {
        "schema_version": PREPARED_AUDIO_SCHEMA_VERSION,
        "dataset": dataset_id,
        "source_name": REAL_AUDIO_DATASET_SPECS[dataset_id].source_name,
        "description": REAL_AUDIO_DATASET_SPECS[dataset_id].description,
        "raw_dir": str(raw_dir.resolve()),
        "sequence_length": sequence_length,
        "feature_dim": REAL_AUDIO_FEATURE_DIM,
        "feature_type": "log_spectrogram",
        "n_classes": len(class_names),
        "class_names": class_names,
        "splits": {"train": "train.pt", "val": "val.pt"},
        "split_strategy": split_strategy,
        "notes": notes,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    card = [
        f"# {REAL_AUDIO_DATASET_SPECS[dataset_id].source_name} Prepared Dataset Card",
        "",
        "Projection:",
        "- fixed-length waveform",
        "- log spectrogram with canonical 128 feature bins",
        "",
        "Notes:",
    ] + [f"- {note}" for note in notes]
    (dataset_dir / "README.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    return {
        "dataset": dataset_id,
        "output_dir": str(dataset_dir),
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "manifest": str(dataset_dir / "manifest.json"),
    }


def prepare_speech_commands_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    load_from_disk = _import_hf_load_from_disk()
    raw_root = _find_speech_commands_root(raw_dir.resolve())
    dataset = load_from_disk(str(raw_root))

    train_split = dataset["train"]
    val_split = dataset["validation"]

    train_audio = []
    train_labels = []
    for example in train_split:
        waveform = _resample_waveform(example["audio"]["array"], sequence_length)
        train_audio.append(_waveform_to_log_spectrogram(waveform))
        train_labels.append(int(example["label"]))
    val_audio = []
    val_labels = []
    for example in val_split:
        waveform = _resample_waveform(example["audio"]["array"], sequence_length)
        val_audio.append(_waveform_to_log_spectrogram(waveform))
        val_labels.append(int(example["label"]))
    train_audio = torch.stack(train_audio)
    val_audio = torch.stack(val_audio)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    train_audio, val_audio = _standardize_audio(train_audio, val_audio)

    dataset_dir = get_real_audio_dataset_dir("speech_commands", output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"audio": train_audio, "labels": train_labels_tensor}, dataset_dir / "train.pt"
    )
    torch.save(
        {"audio": val_audio, "labels": val_labels_tensor}, dataset_dir / "val.pt"
    )

    int2str = train_split.features["label"].int2str
    class_names = [
        int2str(index) for index in range(train_split.features["label"].num_classes)
    ]
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
        "splits": {"train": "train.pt", "val": "val.pt"},
        "notes": [
            "Prepared from Hugging Face Speech Commands.",
            "Audio is resampled/padded to a fixed waveform before log-spectrogram conversion.",
            "Prepared tensors have shape [N, T, 128] for the canonical audio modality.",
        ],
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
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
        "train_samples": int(train_labels_tensor.shape[0]),
        "val_samples": int(val_labels_tensor.shape[0]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "manifest": str(dataset_dir / "manifest.json"),
    }


def prepare_esc50_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = _find_esc50_root(raw_dir.resolve())
    train_examples: List[Tuple[Path, int]] = []
    val_examples: List[Tuple[Path, int]] = []
    class_names: Dict[int, str] = {}
    with (root / "meta" / "esc50.csv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = int(row["target"])
            class_names[label] = row["category"]
            audio_path = root / "audio" / row["filename"]
            if int(row["fold"]) == 5:
                val_examples.append((audio_path, label))
            else:
                train_examples.append((audio_path, label))
    ordered_names = [class_names[index] for index in sorted(class_names)]
    return _write_audio_dataset(
        dataset_id="esc50",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_examples=train_examples,
        val_examples=val_examples,
        class_names=ordered_names,
        notes=[
            "ESC-50 uses the official fold metadata; fold 5 is used as validation and folds 1-4 as training.",
            "This is the first non-speech environmental audio benchmark in the prepared-audio lineup.",
        ],
        split_strategy={
            "type": "fold-holdout",
            "train_folds": [1, 2, 3, 4],
            "val_folds": [5],
        },
    )


def prepare_urbansound8k_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    root = _find_urbansound_root(raw_dir.resolve())
    train_examples: List[Tuple[Path, int]] = []
    val_examples: List[Tuple[Path, int]] = []
    class_names: Dict[int, str] = {}
    with (root / "metadata" / "UrbanSound8K.csv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = int(row["classID"])
            class_names[label] = row["class"]
            audio_path = root / "audio" / f"fold{row['fold']}" / row["slice_file_name"]
            if int(row["fold"]) == 10:
                val_examples.append((audio_path, label))
            else:
                train_examples.append((audio_path, label))
    ordered_names = [class_names[index] for index in sorted(class_names)]
    return _write_audio_dataset(
        dataset_id="urbansound8k",
        raw_dir=raw_dir,
        output_root=output_root,
        sequence_length=sequence_length,
        train_examples=train_examples,
        val_examples=val_examples,
        class_names=ordered_names,
        notes=[
            "UrbanSound8K uses the official fold metadata; fold 10 is used as validation and folds 1-9 as training.",
            "This benchmark extends the prepared-audio path into ambient urban events.",
        ],
        split_strategy={
            "type": "fold-holdout",
            "train_folds": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "val_folds": [10],
        },
    )


def prepare_fsd50k_dataset(
    raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    import pyarrow.parquet as pq

    resolved_raw = raw_dir.resolve()
    try:
        root = _find_fsd50k_root(resolved_raw)
    except FileNotFoundError:
        root = _find_fsd50k_root(_bootstrap_fsd50k_raw(resolved_raw))
    vocabulary_path = root / "ground_truth" / "vocabulary.csv"
    dev_truth_path = root / "ground_truth" / "dev.csv"
    eval_truth_path = root / "ground_truth" / "eval.csv"
    if (
        not vocabulary_path.exists()
        or not dev_truth_path.exists()
        or not eval_truth_path.exists()
    ):
        raise FileNotFoundError(f"FSD50K metadata is incomplete under {root}")

    vocabulary: Dict[int, str] = {}
    with vocabulary_path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            vocabulary[int(row[0])] = row[1]

    def first_label_index(labels_csv: str) -> int:
        first_label = labels_csv.split(",", 1)[0].strip()
        for index, name in vocabulary.items():
            if name == first_label:
                return index
        raise ValueError(f"Unknown FSD50K label '{first_label}'")

    train_lookup: Dict[str, int] = {}
    val_lookup: Dict[str, int] = {}
    with dev_truth_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = first_label_index(row["labels"])
            if row["split"] == "train":
                train_lookup[row["fname"]] = label
            elif row["split"] == "val":
                val_lookup[row["fname"]] = label
    eval_lookup: Dict[str, int] = {}
    with eval_truth_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            eval_lookup[row["fname"]] = first_label_index(row["labels"])

    train_audio: List[torch.Tensor] = []
    train_labels: List[int] = []
    val_audio: List[torch.Tensor] = []
    val_labels: List[int] = []

    for parquet_path in sorted((root / "data").glob("dev-*.parquet")):
        table = pq.read_table(parquet_path, columns=["audio", "name"])
        for row in table.to_pylist():
            stem = Path(row["name"]).stem.split("_", 1)[0]
            if stem in train_lookup:
                target_audio = train_audio
                target_labels = train_labels
                label = train_lookup[stem]
            elif stem in val_lookup:
                target_audio = val_audio
                target_labels = val_labels
                label = val_lookup[stem]
            else:
                continue
            waveform = _decode_audio_bytes(
                row["audio"], Path(row["name"]).suffix or ".ogg"
            )
            waveform_tensor = _resample_waveform(waveform, sequence_length)
            target_audio.append(_waveform_to_log_spectrogram(waveform_tensor))
            target_labels.append(label)

    for parquet_path in sorted((root / "data").glob("eval-*.parquet")):
        table = pq.read_table(parquet_path, columns=["audio", "name"])
        for row in table.to_pylist():
            stem = Path(row["name"]).stem.split("_", 1)[0]
            label = eval_lookup.get(stem)
            if label is None:
                continue
            waveform = _decode_audio_bytes(
                row["audio"], Path(row["name"]).suffix or ".ogg"
            )
            waveform_tensor = _resample_waveform(waveform, sequence_length)
            val_audio.append(_waveform_to_log_spectrogram(waveform_tensor))
            val_labels.append(label)

    if not train_audio or not val_audio:
        raise ValueError("FSD50K preparation produced an empty train or val split")

    train_audio_tensor = torch.stack(train_audio)
    val_audio_tensor = torch.stack(val_audio)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    train_audio_tensor, val_audio_tensor = _standardize_audio(
        train_audio_tensor, val_audio_tensor
    )

    class_names = [name for _, name in sorted(vocabulary.items())]
    dataset_dir = get_real_audio_dataset_dir("fsd50k", output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"audio": train_audio_tensor, "labels": train_labels_tensor},
        dataset_dir / "train.pt",
    )
    torch.save(
        {"audio": val_audio_tensor, "labels": val_labels_tensor}, dataset_dir / "val.pt"
    )
    manifest = {
        "schema_version": PREPARED_AUDIO_SCHEMA_VERSION,
        "dataset": "fsd50k",
        "source_name": REAL_AUDIO_DATASET_SPECS["fsd50k"].source_name,
        "description": REAL_AUDIO_DATASET_SPECS["fsd50k"].description,
        "raw_dir": str(root),
        "sequence_length": sequence_length,
        "feature_dim": REAL_AUDIO_FEATURE_DIM,
        "feature_type": "log_spectrogram",
        "n_classes": len(class_names),
        "class_names": class_names,
        "splits": {"train": "train.pt", "val": "val.pt"},
        "notes": [
            "Prepared from Hugging Face FSD50K parquet shards.",
            "Multi-label annotations are projected deterministically to the first listed label to preserve the canonical single-label contract.",
            "Development train/val plus eval are merged into the validation side to keep the current bounded audio classification protocol simple.",
        ],
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "README.md").write_text(
        "\n".join(
            [
                "# FSD50K Prepared Dataset Card",
                "",
                f"Raw source: {root}",
                "",
                "Prepared output contract:",
                "- manifest: `manifest.json`",
                "- train split: `train.pt`",
                "- val split: `val.pt`",
                "",
                "Projection:",
                "- HF parquet audio bytes decoded to mono waveforms",
                "- fixed-length waveform",
                "- `torch.stft()` log-spectrogram",
                "- deterministic single-label projection using the first listed class",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "dataset": "fsd50k",
        "output_dir": str(dataset_dir),
        "train_samples": int(train_labels_tensor.shape[0]),
        "val_samples": int(val_labels_tensor.shape[0]),
        "n_classes": len(class_names),
        "class_names": class_names,
        "manifest": str(dataset_dir / "manifest.json"),
    }


def prepare_real_audio_dataset(
    dataset: str, raw_dir: Path, output_root: Path, sequence_length: int
) -> Dict[str, object]:
    if dataset == "speech_commands":
        return prepare_speech_commands_dataset(raw_dir, output_root, sequence_length)
    if dataset == "esc50":
        return prepare_esc50_dataset(raw_dir, output_root, sequence_length)
    if dataset == "urbansound8k":
        return prepare_urbansound8k_dataset(raw_dir, output_root, sequence_length)
    if dataset == "fsd50k":
        return prepare_fsd50k_dataset(raw_dir, output_root, sequence_length)
    raise ValueError(f"Unsupported real audio dataset preparation target: {dataset}")
