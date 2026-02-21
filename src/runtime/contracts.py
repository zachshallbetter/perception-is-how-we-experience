"""Runtime contract validation for training, evaluation, and orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

from .errors import AppError, ErrorCode
from .mime_cases import (
    SENSOR_STREAMING_EXPERIMENT_ID,
    get_sensor_streaming_modality_setup,
)


EXPERIMENT_DATASETS = {
    "continual": {"split_cifar", "permuted_mnist"},
    SENSOR_STREAMING_EXPERIMENT_ID: {
        "sensor_streaming",
        "sensor_streaming_smoke",
        "sensor_streaming_small",
    },
    "multimodal": {"audiovisual"},
}


def validate_experiment_dataset(experiment: str, dataset: str) -> None:
    """Validate experiment and dataset combination."""
    allowed_datasets = EXPERIMENT_DATASETS.get(experiment)
    if allowed_datasets is None:
        raise AppError(
            ErrorCode.CONFIG_VALIDATION_FAILED,
            f"Unknown experiment: {experiment}",
            context={"experiment": experiment},
        )
    if dataset not in allowed_datasets:
        raise AppError(
            ErrorCode.DATASET_MISMATCH,
            f"Dataset '{dataset}' is invalid for experiment '{experiment}'",
            context={
                "experiment": experiment,
                "dataset": dataset,
                "allowed_datasets": sorted(allowed_datasets),
            },
        )


def resolve_device(device_value: str) -> torch.device:
    """Resolve and validate target device."""
    try:
        device = torch.device(device_value)
    except Exception as exc:
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            f"Invalid device string: {device_value}",
            context={"device": device_value},
        ) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise AppError(
                ErrorCode.DEVICE_UNAVAILABLE,
                "CUDA requested but not available",
                context={"device": device_value},
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise AppError(
                ErrorCode.DEVICE_UNAVAILABLE,
                f"CUDA device index {device.index} out of range",
                context={
                    "device": device_value,
                    "device_count": torch.cuda.device_count(),
                },
            )
    if device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise AppError(
                ErrorCode.DEVICE_UNAVAILABLE,
                "MPS requested but not available",
                context={"device": device_value},
            )

    return device


def resolve_modality_setup(experiment: str) -> Tuple[Dict[str, Dict], int]:
    """Resolve modality config and class count for experiment."""
    if experiment == "continual":
        return {"vision": {"input_dim": 3, "use_conv": True}}, 5
    if experiment == SENSOR_STREAMING_EXPERIMENT_ID:
        return (get_sensor_streaming_modality_setup(), 5)
    if experiment == "multimodal":
        return (
            {
                "vision": {"input_dim": 3, "use_conv": True},
                "audio": {"input_dim": 128, "use_conv": False},
            },
            10,
        )
    raise AppError(
        ErrorCode.CONFIG_VALIDATION_FAILED,
        f"Unknown experiment: {experiment}",
        context={"experiment": experiment},
    )


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            f"{name} must be > 0",
            context={name: value},
        )


def _validate_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            f"{name} must be >= 0",
            context={name: value},
        )


def _validate_unit_interval(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            f"{name} must be within [0, 1]",
            context={name: value},
        )


@dataclass(frozen=True)
class TrainRuntimeConfig:
    experiment: str
    dataset: str
    d_model: int
    num_heads: int
    depth: int
    mlp_dim: int
    dropout: float
    replay_buffer_size: int
    contrastive_temp: float
    replay_mode: str
    batch_size: int
    epochs: int
    learning_rate: float
    warmup_steps: int
    num_workers: int
    device: torch.device
    seed: int
    output_dir: Path

    @classmethod
    def from_namespace(cls, args) -> "TrainRuntimeConfig":
        validate_experiment_dataset(args.experiment, args.dataset)
        if args.replay_mode not in {"er", "contrastive"}:
            raise AppError(
                ErrorCode.INVALID_ARGUMENT,
                "replay_mode must be 'er' or 'contrastive'",
                context={"replay_mode": args.replay_mode},
            )

        _validate_positive(args.d_model, "d_model")
        _validate_positive(args.num_heads, "num_heads")
        _validate_positive(args.depth, "depth")
        _validate_positive(args.mlp_dim, "mlp_dim")
        _validate_unit_interval(args.dropout, "dropout")
        _validate_non_negative(args.replay_buffer_size, "replay_buffer_size")
        _validate_positive(args.contrastive_temp, "contrastive_temp")
        _validate_positive(args.batch_size, "batch_size")
        _validate_positive(args.epochs, "epochs")
        if args.learning_rate <= 0:
            raise AppError(
                ErrorCode.INVALID_ARGUMENT,
                "learning_rate must be > 0",
                context={"learning_rate": args.learning_rate},
            )
        _validate_non_negative(args.warmup_steps, "warmup_steps")
        _validate_non_negative(args.num_workers, "num_workers")
        _validate_non_negative(args.seed, "seed")

        return cls(
            experiment=args.experiment,
            dataset=args.dataset,
            d_model=args.d_model,
            num_heads=args.num_heads,
            depth=args.depth,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            replay_buffer_size=args.replay_buffer_size,
            contrastive_temp=args.contrastive_temp,
            replay_mode=args.replay_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            num_workers=args.num_workers,
            device=resolve_device(args.device),
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )


@dataclass(frozen=True)
class EvalRuntimeConfig:
    checkpoint: Path
    experiment: str
    dataset: str
    batch_size: int
    num_workers: int
    device: torch.device
    output_file: Path
    seed: int

    @classmethod
    def from_namespace(cls, args) -> "EvalRuntimeConfig":
        validate_experiment_dataset(args.experiment, args.dataset)
        _validate_positive(args.batch_size, "batch_size")
        _validate_non_negative(args.num_workers, "num_workers")
        _validate_non_negative(args.seed, "seed")

        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise AppError(
                ErrorCode.CHECKPOINT_LOAD_FAILED,
                f"Checkpoint does not exist: {checkpoint}",
                context={"checkpoint": str(checkpoint)},
            )

        return cls(
            checkpoint=checkpoint,
            experiment=args.experiment,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=resolve_device(args.device),
            output_file=Path(args.output_file),
            seed=args.seed,
        )


@dataclass(frozen=True)
class OrchestratorRuntimeConfig:
    mode: str
    n_gpus: int
    parallel: bool
    push_to_github: bool

    @classmethod
    def from_namespace(cls, args) -> "OrchestratorRuntimeConfig":
        if args.mode not in {"full", "quick", "monitor"}:
            raise AppError(
                ErrorCode.INVALID_ARGUMENT,
                f"Unknown mode: {args.mode}",
                context={"mode": args.mode},
            )
        _validate_positive(args.n_gpus, "n_gpus")
        return cls(
            mode=args.mode,
            n_gpus=args.n_gpus,
            parallel=args.parallel,
            push_to_github=args.push_to_github,
        )
