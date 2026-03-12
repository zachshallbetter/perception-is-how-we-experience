"""Runtime contract validation for training, evaluation, and orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.data.real_audio import get_real_audio_dataset_ids, get_real_audio_n_classes, is_real_audio_dataset
from src.data.real_sensor import get_real_sensor_dataset_ids, get_real_sensor_n_classes

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
        *get_real_sensor_dataset_ids(),
    },
    "multimodal": {"audiovisual", *get_real_audio_dataset_ids()},
}

DEFAULT_CONTINUAL_TASKS = {
    "split_cifar": 20,
    "permuted_mnist": 20,
}

SENSOR_STREAMING_VARIANTS = {
    "baseline",
    "drop_smartwatch",
    "drop_room_env",
    "drop_robot_arm",
    "corrupt_bindings",
    "noise_smartwatch",
    "noise_room_env",
    "noise_robot_arm",
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


def resolve_modality_setup(
    experiment: str, dataset: Optional[str] = None, modality_routing_mode: str = "canonical"
) -> Tuple[Dict[str, Dict], int]:
    """Resolve modality config and class count for experiment."""
    if modality_routing_mode not in {"canonical", "merged"}:
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            "modality_routing_mode must be 'canonical' or 'merged'",
            context={"modality_routing_mode": modality_routing_mode},
        )

    if experiment == "continual":
        if modality_routing_mode != "canonical":
            raise AppError(
                ErrorCode.INVALID_ARGUMENT,
                "modality_routing_mode=merged is only valid for sensor_streaming",
                context={
                    "experiment": experiment,
                    "modality_routing_mode": modality_routing_mode,
                },
            )
        return {"vision": {"input_dim": 3, "use_conv": True}}, 5
    if experiment == SENSOR_STREAMING_EXPERIMENT_ID:
        n_classes = 5
        if dataset is not None and dataset not in {"sensor_streaming", "sensor_streaming_smoke", "sensor_streaming_small"}:
            n_classes = get_real_sensor_n_classes(dataset)
        if modality_routing_mode == "merged":
            return ({"sensor_stream": {"input_dim": 20, "use_conv": False}}, n_classes)
        return (get_sensor_streaming_modality_setup(), n_classes)
    if experiment == "multimodal":
        if modality_routing_mode != "canonical":
            raise AppError(
                ErrorCode.INVALID_ARGUMENT,
                "modality_routing_mode=merged is only valid for sensor_streaming",
                context={
                    "experiment": experiment,
                    "modality_routing_mode": modality_routing_mode,
                },
            )
        if dataset is not None and is_real_audio_dataset(dataset):
            return (
                {
                    "audio": {"input_dim": 128, "use_conv": False},
                },
                get_real_audio_n_classes(dataset),
            )
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


def resolve_training_setup(
    experiment: str,
    dataset: str,
    continual_task_count: int,
    modality_routing_mode: str = "canonical",
) -> Tuple[Dict[str, Dict], int, int]:
    """Resolve modality config, class count, and effective task count for training."""
    modality_configs, default_n_classes = resolve_modality_setup(
        experiment, dataset=dataset, modality_routing_mode=modality_routing_mode
    )

    if experiment != "continual":
        return modality_configs, default_n_classes, 1

    effective_task_count = (
        continual_task_count
        if continual_task_count > 1
        else DEFAULT_CONTINUAL_TASKS.get(dataset, 1)
    )

    if dataset == "split_cifar":
        n_classes = 100 // effective_task_count
    elif dataset == "permuted_mnist":
        n_classes = 10
    else:
        raise AppError(
            ErrorCode.DATASET_MISMATCH,
            f"Dataset '{dataset}' is invalid for experiment '{experiment}'",
            context={"experiment": experiment, "dataset": dataset},
        )

    return modality_configs, n_classes, effective_task_count


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


def validate_sensor_variant(sensor_variant: str) -> None:
    if sensor_variant not in SENSOR_STREAMING_VARIANTS:
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unknown sensor_variant: {sensor_variant}",
            context={
                "sensor_variant": sensor_variant,
                "allowed_variants": sorted(SENSOR_STREAMING_VARIANTS),
            },
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
    distillation_temp: float
    replay_mode: str
    batch_size: int
    epochs: int
    learning_rate: float
    warmup_steps: int
    num_workers: int
    device: torch.device
    seed: int
    output_dir: Path
    adapter_rollover: bool = False
    continual_task_count: int = 1
    modality_routing_mode: str = "canonical"
    sensor_variant: str = "baseline"

    @classmethod
    def from_namespace(cls, args) -> "TrainRuntimeConfig":
        validate_experiment_dataset(args.experiment, args.dataset)
        if args.replay_mode not in {"er", "contrastive", "feature", "distill"}:
            raise AppError(
                ErrorCode.INVALID_ARGUMENT,
                "replay_mode must be 'er', 'contrastive', 'feature', or 'distill'",
                context={"replay_mode": args.replay_mode},
            )

        _validate_positive(args.d_model, "d_model")
        _validate_positive(args.num_heads, "num_heads")
        _validate_positive(args.depth, "depth")
        _validate_positive(args.mlp_dim, "mlp_dim")
        _validate_unit_interval(args.dropout, "dropout")
        _validate_non_negative(args.replay_buffer_size, "replay_buffer_size")
        _validate_positive(args.contrastive_temp, "contrastive_temp")
        _validate_positive(args.distillation_temp, "distillation_temp")
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
        _validate_positive(
            getattr(args, "continual_task_count", 1), "continual_task_count"
        )
        validate_sensor_variant(getattr(args, "sensor_variant", "baseline"))

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
            distillation_temp=args.distillation_temp,
            replay_mode=args.replay_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            num_workers=args.num_workers,
            device=resolve_device(args.device),
            seed=args.seed,
            output_dir=Path(args.output_dir),
            adapter_rollover=bool(getattr(args, "adapter_rollover", False)),
            continual_task_count=int(getattr(args, "continual_task_count", 1)),
            modality_routing_mode=str(
                getattr(args, "modality_routing_mode", "canonical")
            ),
            sensor_variant=str(getattr(args, "sensor_variant", "baseline")),
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
    modality_routing_mode: str = "canonical"
    sensor_variant: str = "baseline"

    @classmethod
    def from_namespace(cls, args) -> "EvalRuntimeConfig":
        validate_experiment_dataset(args.experiment, args.dataset)
        _validate_positive(args.batch_size, "batch_size")
        _validate_non_negative(args.num_workers, "num_workers")
        _validate_non_negative(args.seed, "seed")
        validate_sensor_variant(getattr(args, "sensor_variant", "baseline"))

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
            modality_routing_mode=str(
                getattr(args, "modality_routing_mode", "canonical")
            ),
            sensor_variant=str(getattr(args, "sensor_variant", "baseline")),
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
