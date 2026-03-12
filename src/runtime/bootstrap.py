"""Runtime bootstrap and environment validation."""

import os
import sys
from typing import Iterable, Optional

import torch

from .errors import AppError, ErrorCode

MIN_PYTHON = (3, 10)


def validate_environment(required_env: Optional[Iterable[str]] = None) -> None:
    """Validate runtime environment before starting a service."""
    if sys.version_info < MIN_PYTHON:
        raise AppError(
            ErrorCode.ENV_VALIDATION_FAILED,
            (
                "Unsupported Python runtime: "
                f"{sys.version_info.major}.{sys.version_info.minor}. "
                f"Minimum required is {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
            ),
            context={
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "min_python": f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            },
        )

    if not hasattr(torch, "__version__") or not torch.__version__:
        raise AppError(
            ErrorCode.ENV_VALIDATION_FAILED,
            "PyTorch is not available in runtime environment",
        )

    missing_vars = [
        name for name in (required_env or []) if name and name not in os.environ
    ]
    if missing_vars:
        raise AppError(
            ErrorCode.ENV_VALIDATION_FAILED,
            "Missing required environment variables",
            context={"missing_env": sorted(missing_vars)},
        )
