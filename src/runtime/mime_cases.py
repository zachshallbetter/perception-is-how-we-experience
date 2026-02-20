"""Canonical MIME-style input case definitions for runtime modalities."""

from dataclasses import dataclass
from typing import Dict, Mapping

from .errors import AppError, ErrorCode


@dataclass(frozen=True)
class InputCase:
    """Canonical definition of one modality input case."""

    modality: str
    mime_type: str
    feature_dim: int
    description: str


SENSOR_STREAMING_EXPERIMENT_ID = "sensor_streaming"
SENSOR_STREAMING_CANONICAL_LABEL = "sensor_streaming"

SENSOR_STREAMING_INPUT_CASES: Dict[str, InputCase] = {
    "smartwatch": InputCase(
        modality="smartwatch",
        mime_type="application/vnd.sensory.smartwatch.timeseries+tensor",
        feature_dim=7,
        description="Wearable stream (heart rate, accelerometer, skin response, aux).",
    ),
    "room_env": InputCase(
        modality="room_env",
        mime_type="application/vnd.sensory.environment.timeseries+tensor",
        feature_dim=4,
        description="Ambient stream (temperature, humidity, CO2, light).",
    ),
    "robot_arm": InputCase(
        modality="robot_arm",
        mime_type="application/vnd.sensory.robot-arm.timeseries+tensor",
        feature_dim=9,
        description="Actuation/kinematics stream (joints + force/torque).",
    ),
}


def get_sensor_streaming_modality_setup() -> Dict[str, Dict]:
    """Return runtime modality setup derived from canonical sensor-streaming input cases."""
    return {
        case.modality: {"input_dim": case.feature_dim, "use_conv": False}
        for case in SENSOR_STREAMING_INPUT_CASES.values()
    }


def validate_sensor_streaming_mime_bindings(bindings: Mapping[str, str]) -> None:
    """Validate modality→MIME bindings for sensor-streaming inputs."""
    for modality, mime_type in bindings.items():
        expected_case = SENSOR_STREAMING_INPUT_CASES.get(modality)
        if expected_case is None:
            raise AppError(
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"Unknown sensor-streaming modality: {modality}",
                context={"modality": modality},
            )
        if mime_type != expected_case.mime_type:
            raise AppError(
                ErrorCode.CONFIG_VALIDATION_FAILED,
                f"Invalid MIME type for modality '{modality}'",
                context={
                    "modality": modality,
                    "expected_mime_type": expected_case.mime_type,
                    "received_mime_type": mime_type,
                },
            )
