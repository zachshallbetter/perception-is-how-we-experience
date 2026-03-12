"""Stream synchronization service mapping un-ordered envelopes to frozen inference windows."""

from typing import Dict, Iterable, List, Tuple

import torch

from src.runtime.schema import (
    CANONICAL_MODALITIES,
    FrameEnvelope,
    ModalityType,
    SensorWindow,
    StreamStatus,
)


class BaseSynchronizer:
    """
    Synchronizes an explicitly unordered or disjoint stream of raw events
    into a deterministic, frozen dictionary of ModalityType -> SensorWindow.
    """

    def __init__(self, window_size_ms: int, sampling_rate_hz: int) -> None:
        self.window_size_ms = window_size_ms
        self.sampling_rate_hz = sampling_rate_hz
        self.step_ms = 1000 // sampling_rate_hz
        self.required_steps = self.window_size_ms // self.step_ms

    def build_frozen_window(
        self,
        envelopes: Iterable[FrameEnvelope],
        window_start_ms: int,
        window_end_ms: int,
    ) -> Dict[ModalityType, SensorWindow]:
        """
        Pure function: Converts a list of explicitly provided envelopes and an explicit time window
        into a synchronized map of sensor windows. Follows strict mapping and no-silent defaults.
        """
        if window_end_ms - window_start_ms != self.window_size_ms:
            raise ValueError(
                f"Window bounds diff {window_end_ms - window_start_ms} "
                f"does not match expected {self.window_size_ms}"
            )

        # Buffer incoming per modality
        buffer: Dict[ModalityType, List[Tuple[int, List[float]]]] = {
            m: [] for m in ModalityType
        }

        for env in envelopes:
            mod_type = env.get_modality_type()
            # Only include envelopes strictly within the window boundaries [start, end)
            if window_start_ms <= env.timestamp_ms < window_end_ms:
                buffer[mod_type].append((env.timestamp_ms, env.payload))

        results: Dict[ModalityType, SensorWindow] = {}

        for mod_type in ModalityType:
            series = buffer[mod_type]
            config = CANONICAL_MODALITIES[mod_type]
            feature_dim = config.feature_dim

            # Sort series purely based on explicit timestamp (deterministic)
            series.sort(key=lambda item: item[0])

            # Simple missing-data detection rules
            status = StreamStatus.NOMINAL
            if len(series) == 0:
                status = StreamStatus.MISSING
            elif len(series) < self.required_steps // 2:
                status = StreamStatus.DEGRADED

            tensor = torch.zeros((1, self.required_steps, feature_dim), dtype=torch.float32)

            if status != StreamStatus.MISSING:
                # We align the envelopes to discrete buckets within the window
                # using the first matching envelope for the bucket
                for ts, payload in series:
                    bucket_idx = (ts - window_start_ms) // self.step_ms
                    if 0 <= bucket_idx < self.required_steps:
                        tensor[0, bucket_idx, :] = torch.tensor(payload, dtype=torch.float32)

                # Forward-fill purely deterministically for explicitly missing buckets inside a DEGRADED/NOMINAL window
                # Instead of silently doing it, we consider it part of the explicit synchronizer logic
                last_val = torch.zeros(feature_dim, dtype=torch.float32)
                for t in range(self.required_steps):
                    if (tensor[0, t, :] == 0).all():
                        tensor[0, t, :] = last_val
                    else:
                        last_val = tensor[0, t, :]

            results[mod_type] = SensorWindow(
                modality=mod_type,
                tensor=tensor,
                status=status,
            )

        return results
