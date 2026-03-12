# Docs

This directory is the operator and research bridge between the codebase and the paper.

## Read This First

1. `training-guide.md`
   - canonical Titan workflow
   - primary launchers
   - monitoring and packaging flow
2. `productization-roadmap.md`
   - how the current research result translates into a usable system roadmap
   - what still has to be true before strong product claims
3. `baselines.md`
   - baseline environments and Titan-safe defaults
4. `sensor-mime-experiment-plan.md`
   - perception-side evidence ladder
   - canonical vs merged routing rationale
5. `sensor-streaming-mime-cases.md`
   - authoritative sensor modality cases
6. `real-sensor-datasets.md`
   - prepared-data scaffold for WISDM, PAMAP2, and OPPORTUNITY

## How To Read These Docs

- If you are operating Titan today, start with `training-guide.md`.
- If you are trying to understand what the paper currently proves, read `sensor-mime-experiment-plan.md` together with `paper/sections/06-results.md` and `paper/sections/07-discussion.md`.
- If you are trying to decide whether this work is usable beyond the paper, read `productization-roadmap.md`.

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
