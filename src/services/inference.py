"""Pure inference evaluation loop mapping frozen sensor windows to predictions."""

from dataclasses import dataclass
from typing import Dict, Optional, Set

import torch

from src.runtime.schema import ModalityType, SensorWindow, StreamStatus


class InferenceErrorCodes:
    MISSING_REQUIRED_MODALITIES = "MISSING_REQUIRED_MODALITIES"
    INVALID_WINDOW_SHAPE = "INVALID_WINDOW_SHAPE"
    MODEL_EVALUATION_FAILED = "MODEL_EVALUATION_FAILED"


@dataclass(frozen=True)
class InferencePlan:
    """Pure data output of an inference step."""
    prediction_logits: torch.Tensor
    is_safe: bool
    confidence_score: float
    error_code: Optional[str]


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

    # 2. Build explicit input batch for the perceptual core
    # We pass the valid modalities into the architecture
    model_inputs = {}
    for mod_type, window in frozen_windows.items():
        if window.status != StreamStatus.MISSING:
            # We enforce exhaustive handling by relying on Enum keys
            # Exposing the input by standard string key as expected by standard model signatures
            model_inputs[mod_type.value] = window.tensor
        else:
            # Explicitly omit missing modality from model inputs rather than passing zeros unexpectedly,
            # trusting the architecture to adapt (or the requirement check above to throw).
            pass

    # 3. Model evaluation. In production, wrap this in timeouts/compute isolation,
    # but here we consider model() a pure mathematical step.
    try:
        with torch.no_grad():
            output_logits = model(model_inputs)
    except Exception:
        return InferencePlan(
            prediction_logits=torch.empty(0),
            is_safe=False,
            confidence_score=0.0,
            error_code=InferenceErrorCodes.MODEL_EVALUATION_FAILED,
        )

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
