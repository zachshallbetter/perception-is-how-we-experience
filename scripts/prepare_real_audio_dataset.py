#!/usr/bin/env python3
"""Prepare or bootstrap real audio datasets for the multimodal experiment family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.real_audio import (
    PREPARED_AUDIO_SCHEMA_VERSION,
    REAL_AUDIO_DATASET_SPECS,
    get_real_audio_dataset_dir,
)
from src.data.real_audio_prepare import prepare_real_audio_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(REAL_AUDIO_DATASET_SPECS), required=True)
    parser.add_argument("--raw-dir", required=True, help="Path to the downloaded raw dataset root")
    parser.add_argument(
        "--output-root",
        default="data/real-audio",
        help="Root directory for prepared real-audio datasets",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16000,
        help="Target waveform length before spectrogram conversion",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Only create manifest/README scaffold even for supported prepared parsers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    if args.dataset in {"speech_commands", "esc50", "urbansound8k"} and not args.bootstrap_only:
        result = prepare_real_audio_dataset(args.dataset, raw_dir, Path(args.output_root), args.sequence_length)
        print(json.dumps(result, indent=2))
        return 0

    dataset_dir = get_real_audio_dataset_dir(args.dataset, args.output_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": PREPARED_AUDIO_SCHEMA_VERSION,
        "dataset": args.dataset,
        "source_name": REAL_AUDIO_DATASET_SPECS[args.dataset].source_name,
        "description": REAL_AUDIO_DATASET_SPECS[args.dataset].description,
        "raw_dir": str(raw_dir),
        "sequence_length": 96,
        "feature_dim": 128,
        "feature_type": "log_spectrogram",
        "n_classes": None,
        "class_names": [],
        "splits": {
            "train": "train.pt",
            "val": "val.pt",
        },
        "required_modalities": {
            "audio": 128,
        },
        "notes": [
            "Fill n_classes and class_names before training.",
            "Prepared split files must be torch.save payloads with keys: labels, audio.",
            "Audio tensors must have shape [N, T, 128] using the canonical prepared audio contract.",
            "Use --bootstrap-only to preserve scaffold-only behavior after parser support lands.",
        ],
    }

    dataset_card = f"""# {REAL_AUDIO_DATASET_SPECS[args.dataset].source_name} Prepared Dataset Card

Raw source:
- {raw_dir}

Prepared output contract:
- manifest: `manifest.json`
- train split: `train.pt`
- val split: `val.pt`

Expected `torch.save()` payload for each split:
- `labels`: `LongTensor[N]`
- `audio`: `FloatTensor[N, T, 128]`

Scaffold status:
- bootstrap created
- raw parsing implementation still TODO for this dataset
- this directory is ready for canonical prepared audio tensors
"""

    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (dataset_dir / "README.md").write_text(dataset_card, encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "raw_dir": str(raw_dir),
                "output_dir": str(dataset_dir),
                "manifest": str(dataset_dir / "manifest.json"),
                "dataset_card": str(dataset_dir / "README.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
