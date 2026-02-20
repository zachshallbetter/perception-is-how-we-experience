"""Runtime contract and error definitions."""

from .contracts import (
    EvalRuntimeConfig,
    OrchestratorRuntimeConfig,
    TrainRuntimeConfig,
    resolve_device,
    resolve_modality_setup,
    validate_experiment_dataset,
)
from .bootstrap import validate_environment
from .errors import AppError, ErrorCode, classify_exception, format_error
from .schema import TRAIN_CONFIG_SCHEMA_VERSION, validate_training_config_schema

__all__ = [
    "AppError",
    "ErrorCode",
    "classify_exception",
    "format_error",
    "validate_environment",
    "TrainRuntimeConfig",
    "EvalRuntimeConfig",
    "OrchestratorRuntimeConfig",
    "resolve_device",
    "resolve_modality_setup",
    "validate_experiment_dataset",
    "TRAIN_CONFIG_SCHEMA_VERSION",
    "validate_training_config_schema",
]
