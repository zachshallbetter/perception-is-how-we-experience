# Future Concepts

This directory is the non-canonical concept library for Sensory Perceiver.

It exists to capture reusable architecture patterns, domain examples, and invention candidates **without** promoting them into the current paper claim surface.

Authority boundary:
- Canonical paper claims live under [paper/sections](/Users/zachshallbetter/Projects/Perception/paper/sections).
- Code-aligned operational guidance lives under [docs](/Users/zachshallbetter/Projects/Perception/docs).
- This directory is for **Conceptual** and **future-use** material only.
- Patent-oriented derivatives belong under [paper/patents](/Users/zachshallbetter/Projects/Perception/paper/patents).

## Labeling Rule

Use these labels consistently:

- **Implemented**: exists in the current repo code path
- **Evidenced**: backed by packaged results or completed reports
- **Conceptual**: design idea or architecture pattern not yet promoted
- **Patent candidate**: novelty-oriented idea tracked separately
- **Blocked / open problem**: not runnable, not fully defined, or not yet evidence-ready

## Current Concept Map

- [artifact-map.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/artifact-map.md)
- [distinction-and-separation.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/distinction-and-separation.md)
- [towers-and-modality-semantics.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/towers-and-modality-semantics.md)
- [early-fusion-vs-towers.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/early-fusion-vs-towers.md)
- [replay-through-towers.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/replay-through-towers.md)
- [hierarchical-modality-grouping.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/hierarchical-modality-grouping.md)
- [dynamic-modality-composition.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/dynamic-modality-composition.md)
- [async-sensor-synchronization.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/async-sensor-synchronization.md)
- [modality-selection-and-importance.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/modality-selection-and-importance.md)
- [live-perception-to-action-loops.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/live-perception-to-action-loops.md)
- [domain-patterns/security-monitoring.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/domain-patterns/security-monitoring.md)
- [domain-patterns/healthcare-wearables.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/domain-patterns/healthcare-wearables.md)
- [domain-patterns/robotics.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/domain-patterns/robotics.md)
- [domain-patterns/smart-home.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/domain-patterns/smart-home.md)
- [domain-patterns/live-audio-and-visualization.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/domain-patterns/live-audio-and-visualization.md)

## Source Artifact Policy

The concept notes in this directory may draw from external artifacts such as:

- `/Users/zachshallbetter/Downloads/security_camera_tower_design.html`
- `/Users/zachshallbetter/Downloads/replay_data_flow_interactive.html`
- `/Users/zachshallbetter/Downloads/early_fusion_vs_towers_analysis.html`
- `/Users/zachshallbetter/Downloads/async_sensor_synchronization.html`
- `/Users/zachshallbetter/Downloads/hierarchical_6_modality_perceiver.py`
- `/Users/zachshallbetter/Downloads/dynamic_modality_composition.py`
- `/Users/zachshallbetter/Downloads/concert_audio_perceiver.html`
- `/Users/zachshallbetter/Downloads/perceiver_io_specification.py`

Those artifacts are idea sources, not canonical paper evidence.

## New Artifact Intake Workflow

When a new external artifact arrives:

1. add it to [artifact-map.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/artifact-map.md)
2. assign it to exactly one primary lane:
   - concept library
   - patent track
   - discarded artifact
3. normalize the useful ideas into a concept note or patent note
4. do not mention it in the canonical paper unless it later becomes both:
   - **Implemented**
   - **Evidenced**

## Promotion Rule

A concept may move from this directory into the canonical paper only when it is both:

1. **Implemented** enough to be described honestly in the current repo, and
2. **Evidenced** by artifact-backed results or completed reports.
