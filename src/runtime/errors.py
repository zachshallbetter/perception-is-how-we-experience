"""Typed runtime errors and error codes for production-style entrypoints."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Stable error codes for CLI/runtime boundaries."""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"
    DATASET_MISMATCH = "DATASET_MISMATCH"
    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    CHECKPOINT_LOAD_FAILED = "CHECKPOINT_LOAD_FAILED"
    ORCHESTRATION_FAILED = "ORCHESTRATION_FAILED"
    IO_ERROR = "IO_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Application error with typed code and retryability metadata."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.context = context or {}


def format_error(error: AppError) -> str:
    """Format an AppError for logs."""
    context_parts = [f"{key}={value}" for key, value in sorted(error.context.items())]
    context = ", ".join(context_parts) if context_parts else "none"
    return (
        f"code={error.code.value} retryable={error.retryable} "
        f"message={error} context={context}"
    )
