# Docs

This directory is the operator and research bridge between the codebase and the paper.

## Read This First

1. `training-guide.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/training-guide.md](/Users/zachshallbetter/Projects/Perception/docs/training-guide.md)
   - canonical Titan workflow
   - primary launchers
   - monitoring and packaging flow
2. `productization-roadmap.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/productization-roadmap.md](/Users/zachshallbetter/Projects/Perception/docs/productization-roadmap.md)
   - how the current research result translates into a usable system roadmap
   - what still has to be true before strong product claims
3. `live-deployment-architecture.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/live-deployment-architecture.md](/Users/zachshallbetter/Projects/Perception/docs/live-deployment-architecture.md)
   - concrete live-environment interpretation of the current architecture
   - capture, synchronization, inference, adaptation, and product-surface responsibilities
4. `baselines.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/baselines.md](/Users/zachshallbetter/Projects/Perception/docs/baselines.md)
   - baseline environments and Titan-safe defaults
5. `sensor-mime-experiment-plan.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/sensor-mime-experiment-plan.md](/Users/zachshallbetter/Projects/Perception/docs/sensor-mime-experiment-plan.md)
   - perception-side evidence ladder
   - canonical vs merged routing rationale
6. `sensor-streaming-mime-cases.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/sensor-streaming-mime-cases.md](/Users/zachshallbetter/Projects/Perception/docs/sensor-streaming-mime-cases.md)
   - authoritative sensor modality cases
7. `real-sensor-datasets.md`
   - [/Users/zachshallbetter/Projects/Perception/docs/real-sensor-datasets.md](/Users/zachshallbetter/Projects/Perception/docs/real-sensor-datasets.md)
   - real-data support matrix, prepared-data contract, and autonomous progression rules

## How To Read These Docs

- If you are operating Titan today, start with `training-guide.md`.
- If you are trying to understand what the paper currently proves, read [sensor-mime-experiment-plan.md](/Users/zachshallbetter/Projects/Perception/docs/sensor-mime-experiment-plan.md) together with [06-results.md](/Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md) and [07-discussion.md](/Users/zachshallbetter/Projects/Perception/paper/sections/07-discussion.md).
- If you are trying to decide whether this work is usable beyond the paper, read [productization-roadmap.md](/Users/zachshallbetter/Projects/Perception/docs/productization-roadmap.md).
- If you are thinking about a real live deployment, read [live-deployment-architecture.md](/Users/zachshallbetter/Projects/Perception/docs/live-deployment-architecture.md) before trying to infer product behavior from the training scripts.
- If you are trying to understand the user-facing outcome rather than the mechanism details, read [README.md](/Users/zachshallbetter/Projects/Perception/README.md) together with [productization-roadmap.md](/Users/zachshallbetter/Projects/Perception/docs/productization-roadmap.md).

## Experiment and Review Docs

- `dynamic-weighting.md`
- `reviewer-attack-map.md`
- `reviewer-rebuttal-plan.md`
- `changes-since-3bee0be.md`

## Canonical Authority Boundaries

- Paper claims are authoritative under `paper/sections/`.
- The generated paper monolith under `paper/Perception Is How We Experience.md` must be regenerated from section files, not hand-edited.
- Research and patent drafts under `paper/supplementary/` and `paper/patents/` are non-canonical and must stay within the claim boundaries defined in:
  - `paper/sections/00-implementation-status-matrix.md`
  - `paper/sections/10-concept-implementation-audit.md`

## Current Documentation Priorities

- keep README, docs, and canonical paper sections synchronized whenever experiment status changes
- prefer one named workflow per experiment family
- document automation and stop conditions explicitly rather than implying open-ended search
