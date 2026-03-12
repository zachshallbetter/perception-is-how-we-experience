"""Pure inference evaluation loop mapping frozen sensor windows to predictions."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import torch

from src.runtime.errors import AppError, ErrorCode
from src.runtime.live_schema import ModalityType, SensorWindow, StreamStatus


class InferenceErrorCodes:
    MISSING_REQUIRED_MODALITIES = "MISSING_REQUIRED_MODALITIES"
    INVALID_WINDOW_SHAPE = "INVALID_WINDOW_SHAPE"
    MODEL_EVALUATION_FAILED = "MODEL_EVALUATION_FAILED"
    MODEL_CONTRACT_MISMATCH = "MODEL_CONTRACT_MISMATCH"


@dataclass(frozen=True)
class InferencePlan:
    """Pure data output of an inference step."""

    prediction_logits: torch.Tensor
    is_safe: bool
    confidence_score: float
    error_code: Optional[str]


class LiveTensorAdapter(torch.nn.Module):
    """Adapter for legacy tensor-only models to consume canonical live payloads.

    This keeps the live inference boundary explicit without forcing the entire
    model stack to adopt the new structured input contract in one step.
    """

    supports_live_valid_mask = True

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, inputs: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        legacy_inputs: Dict[str, torch.Tensor] = {}
        for modality, payload in inputs.items():
            tensor = payload["tensor"]
            valid_mask = payload["valid_mask"]
            if tensor.ndim != 3 or valid_mask.ndim != 2:
                raise AppError(
                    ErrorCode.INVALID_ARGUMENT,
                    "LiveTensorAdapter expected rank-3 tensor and rank-2 valid_mask",
                    context={
                        "modality": modality,
                        "tensor_shape": tuple(tensor.shape),
                        "valid_mask_shape": tuple(valid_mask.shape),
                    },
                )
            if tensor.shape[:2] != valid_mask.shape:
                raise AppError(
                    ErrorCode.INVALID_ARGUMENT,
                    "LiveTensorAdapter valid_mask must match tensor batch/time dimensions",
                    context={
                        "modality": modality,
                        "tensor_shape": tuple(tensor.shape),
                        "valid_mask_shape": tuple(valid_mask.shape),
                    },
                )

            masked = tensor * valid_mask.unsqueeze(-1).to(dtype=tensor.dtype)
            legacy_inputs[modality] = masked

        return self.base_model(legacy_inputs)


def build_live_model_inputs(
    frozen_windows: Dict[ModalityType, SensorWindow],
) -> Dict[str, Dict[str, Any]]:
    """Build the canonical live model input payload."""
    model_inputs: Dict[str, Dict[str, Any]] = {}
    for mod_type, window in frozen_windows.items():
        if window.status == StreamStatus.MISSING:
            continue
        model_inputs[mod_type.value] = {
            "tensor": window.tensor,
            "valid_mask": window.valid_mask,
            "status": window.status.value,
        }
    return model_inputs


def resolve_inference_plan(
    frozen_windows: Dict[ModalityType, SensorWindow],
    model: torch.nn.Module,
    required_modalities: Set[ModalityType],
) -> InferencePlan:
    """
    Pure evaluation: maps frozen input windows + model weights to a fixed prediction response.

    Adheres to no-fallback: if required modalities are missing, hard failure return.
    """
    # 1. Evaluate prerequisites
    for req_mod in required_modalities:
        window = frozen_windows.get(req_mod)
        if window is None or window.status == StreamStatus.MISSING:
            return InferencePlan(
                prediction_logits=torch.empty(0),
                is_safe=False,
                confidence_score=0.0,
                error_code=InferenceErrorCodes.MISSING_REQUIRED_MODALITIES,
            )

    # 2. Build explicit input batch for the perceptual core.
    # Live-capable models must accept the canonical structured payload so that
    # degraded windows are visible through valid_mask rather than hidden behind raw tensors.
    if not getattr(model, "supports_live_valid_mask", False):
        raise AppError(
            ErrorCode.INVALID_ARGUMENT,
            "Live inference model does not declare support for canonical valid_mask inputs",
            context={"required_attribute": "supports_live_valid_mask"},
        )

    model_inputs = build_live_model_inputs(frozen_windows)

    # 3. Model evaluation. In production, wrap this in timeouts/compute isolation,
    # but here we consider model() a pure mathematical step.
    try:
        with torch.no_grad():
            output_logits = model(model_inputs)
    except AppError:
        raise
    except Exception as e:
        # TODO: Define specific model evaluation errors explicitly rather than catching all.
        # Fallback to strict hard error to avoid silent evaluation failures.
        raise AppError(
            ErrorCode.INTERNAL_ERROR,
            f"Pure model evaluation failed unexpectedly: {e}",
            context={"exception": str(e)},
        ) from e

    # Assume a standard softmax confidence baseline for now
    if output_logits.numel() > 0:
        # Assuming logits are shape [B, num_classes]
        probs = torch.nn.functional.softmax(output_logits, dim=-1)
        confidence = probs.max().item()
    else:
        confidence = 0.0

    is_safe = confidence > 0.65

    return InferencePlan(
        prediction_logits=output_logits,
        is_safe=is_safe,
        confidence_score=confidence,
        error_code=None,
    )
