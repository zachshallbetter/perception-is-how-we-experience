"""Versioned schema validation for persisted runtime config."""

from typing import Dict

from .errors import AppError, ErrorCode

TRAIN_CONFIG_SCHEMA_VERSION = 1

REQUIRED_TRAIN_CONFIG_KEYS = {
    "config_schema_version",
    "experiment",
    "dataset",
    "d_model",
    "num_heads",
    "depth",
    "mlp_dim",
    "dropout",
    "replay_buffer_size",
    "contrastive_temp",
    "replay_mode",
    "batch_size",
    "epochs",
    "learning_rate",
    "warmup_steps",
    "num_workers",
    "seed",
    "device",
    "output_dir",
    "modality_configs",
    "n_classes",
}


def validate_training_config_schema(config: Dict) -> None:
    """Validate checkpoint training config against canonical schema."""
    missing_keys = sorted(REQUIRED_TRAIN_CONFIG_KEYS - set(config.keys()))
    if missing_keys:
        raise AppError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            "Training config is missing required fields",
            context={"missing_keys": missing_keys},
        )

    version = config.get("config_schema_version")
    if version != TRAIN_CONFIG_SCHEMA_VERSION:
        raise AppError(
            ErrorCode.SCHEMA_VALIDATION_FAILED,
            (
                f"Unsupported training config schema version: {version}. "
                f"Expected {TRAIN_CONFIG_SCHEMA_VERSION}"
            ),
            context={
                "schema_version": version,
                "expected_schema_version": TRAIN_CONFIG_SCHEMA_VERSION,
            },
        )

# --- Live Streaming Schemas ---

from dataclasses import dataclass
from enum import Enum
from typing import List

import torch


# Exhaustive discriminators for allowed modalities
class ModalityType(Enum):
    SMARTWATCH = "smartwatch"
    ROOM_ENV = "room_env"
    ROBOT_ARM = "robot_arm"


class StreamStatus(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    MISSING = "missing"


@dataclass(frozen=True)
class ModalityConfig:
    feature_dim: int


# SSoT for modality dimensions
CANONICAL_MODALITIES: Dict[ModalityType, ModalityConfig] = {
    ModalityType.SMARTWATCH: ModalityConfig(feature_dim=7),
    ModalityType.ROOM_ENV: ModalityConfig(feature_dim=4),
    ModalityType.ROBOT_ARM: ModalityConfig(feature_dim=9),
}


@dataclass(frozen=True)
class SensorWindow:
    """A purely constructed, frozen chunk of synchronized data for a single modality."""
    modality: ModalityType
    tensor: torch.Tensor  # shape: [B, time_steps, feature_dim]
    status: StreamStatus

    def __post_init__(self) -> None:
        expected_dim = CANONICAL_MODALITIES[self.modality].feature_dim
        if self.tensor.ndim != 3:
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SensorWindow for {self.modality} must be rank-3 [B, T, C]. Got: {self.tensor.shape}"
            )
        if self.tensor.shape[2] != expected_dim:
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SensorWindow for {self.modality} must have feature dim {expected_dim}. Got: {self.tensor.shape[2]}"
            )


@dataclass(frozen=True)
class FrameEnvelope:
    """Canonical input event format from the ingest pipeline."""
    trace_id: str
    timestamp_ms: int
    modality_id: str
    payload: List[float]

    def get_modality_type(self) -> ModalityType:
        try:
            return ModalityType(self.modality_id)
        except ValueError:
            # No-fallback: hard error on unknown discriminator
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SCHEMA_CHANGED: Unknown modality discriminator encountered: {self.modality_id}"
            )
