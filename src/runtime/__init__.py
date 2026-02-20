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
from .mime_cases import (
    SENSOR_STREAMING_CANONICAL_LABEL,
    SENSOR_STREAMING_EXPERIMENT_ID,
    SENSOR_STREAMING_INPUT_CASES,
    get_sensor_streaming_modality_setup,
    validate_sensor_streaming_mime_bindings,
)
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
    "SENSOR_STREAMING_EXPERIMENT_ID",
    "SENSOR_STREAMING_CANONICAL_LABEL",
    "SENSOR_STREAMING_INPUT_CASES",
    "get_sensor_streaming_modality_setup",
    "validate_sensor_streaming_mime_bindings",
    "TRAIN_CONFIG_SCHEMA_VERSION",
    "validate_training_config_schema",
]
