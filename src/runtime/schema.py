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
