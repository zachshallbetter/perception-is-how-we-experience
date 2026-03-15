# Healthcare Wearables

**Status**
- Label: **Conceptual**
- Evidence status: partially adjacent to WISDM/bootstrap sensor work, but not yet promoted as a healthcare result
- Patent relevance: medium

## Concept

Combine wearables, medical device telemetry, room context, and physiological streams while keeping modality identity explicit and bounded adaptation auditable.

## Why It Matters

This domain emphasizes:
- body-state vs room-state separation
- long-horizon adaptation
- degraded-sensor tolerance
- patient-specific updating without full retraining

## Strong Modality Split

The cleanest healthcare/wearables split is usually:

- physiological waveforms or body-state signals
- slower contextual vitals or device telemetry
- room or ambient context

This is attractive scientifically because it stresses multi-timescale distinction rather than just “more sensors.”

## Good First Evidence Shape

The most credible first healthcare promotion path is:

- one bounded outcome task,
- one explicit calibration and missingness protocol,
- one typed-routing versus merged comparison,
- and one retention or drift analysis over time.

## Source Artifacts

- [/Users/zachshallbetter/Projects/Perception/reference/sensor-types/Sensor References/Interoception.md](/Users/zachshallbetter/Projects/Perception/reference/sensor-types/Sensor%20References/Interoception%201ac760f264f144859d34418a5de0a4af.md)
- [/Users/zachshallbetter/Projects/Perception/reference/sensor-types/Sensor References/Proprioception.md](/Users/zachshallbetter/Projects/Perception/reference/sensor-types/Sensor%20References/Proprioception%20d722fb4b0fb7479185fd9d45614a94a7.md)
