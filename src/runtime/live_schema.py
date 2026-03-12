"""Canonical schemas for live streaming perception."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch

from src.runtime.errors import AppError, ErrorCode


# Exhaustive discriminators for allowed modalities in the live system
class ModalityType(Enum):
    SMARTWATCH = "smartwatch"
    ROOM_ENV = "room_env"
    ROBOT_ARM = "robot_arm"
    AUDIO = "audio"
    VISION = "vision"
    POSE = "pose"
    DEPTH = "depth"
    WEARABLE = "wearable"
    ROOM = "room"


class StreamStatus(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    MISSING = "missing"


@dataclass(frozen=True)
class ModalityConfig:
    feature_dim: int


# SSoT for modality dimensions in the live system
# Real sensor dimensions reflect production bounds
CANONICAL_MODALITIES: Dict[ModalityType, ModalityConfig] = {
    ModalityType.SMARTWATCH: ModalityConfig(feature_dim=7),
    ModalityType.ROOM_ENV: ModalityConfig(feature_dim=4),
    ModalityType.ROBOT_ARM: ModalityConfig(feature_dim=9),
    ModalityType.AUDIO: ModalityConfig(feature_dim=128),     # e.g., mel spectrogram bin count
    ModalityType.VISION: ModalityConfig(feature_dim=512),    # e.g., frozen visual backbone feature dim
    ModalityType.POSE: ModalityConfig(feature_dim=99),       # e.g., 33 keypoints x 3 dims
    ModalityType.DEPTH: ModalityConfig(feature_dim=64),      # e.g., depth patches
    ModalityType.WEARABLE: ModalityConfig(feature_dim=12),   # e.g., advanced IMU + HR
    ModalityType.ROOM: ModalityConfig(feature_dim=15),       # e.g., occupancy + temp + air quality
}


@dataclass(frozen=True)
class SensorWindow:
    """A purely constructed, frozen chunk of synchronized data for a single modality."""
    modality: ModalityType
    tensor: torch.Tensor       # actual data points, shape: [B, time_steps, feature_dim]
    valid_mask: torch.Tensor   # explicitly marks whether each timestep in `tensor` contains valid sensor data, shape: [B, time_steps]
    status: StreamStatus

    def __post_init__(self) -> None:
        expected_dim = CANONICAL_MODALITIES[self.modality].feature_dim
        if self.tensor.ndim != 3:
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SensorWindow tensor for {self.modality} must be rank-3 [B, T, C]. Got: {self.tensor.shape}"
            )
        if self.tensor.shape[2] != expected_dim:
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SensorWindow tensor for {self.modality} must have feature dim {expected_dim}. Got: {self.tensor.shape[2]}"
            )
        if self.valid_mask.ndim != 2:
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SensorWindow valid_mask for {self.modality} must be rank-2 [B, T]. Got: {self.valid_mask.shape}"
            )
        if self.valid_mask.shape[0] != self.tensor.shape[0] or self.valid_mask.shape[1] != self.tensor.shape[1]:
            raise AppError(
                ErrorCode.SCHEMA_VALIDATION_FAILED,
                f"SensorWindow valid_mask shape {self.valid_mask.shape} must match tensor batch and time steps {self.tensor.shape[:2]}"
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
