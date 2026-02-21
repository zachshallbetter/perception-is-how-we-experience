# Supplementary Paper 3: Applications in Robotics, Medical, and Smart Homes

## Scope

This directory defines application-domain evaluation frameworks for future validation tracks.
It is not authoritative for main-paper empirical claims.

Canonical authority remains:
- `paper/sections/05-experiments.md`
- `paper/sections/06-results.md`
- `paper/sections/00-implementation-status-matrix.md`

## Purpose

The applications supplement translates architecture/runtime concepts into domain-specific experiment specifications, including:
1. problem definitions,
2. modality contracts,
3. evaluation protocols,
4. risk and deployment constraints,
5. promotion criteria for canonical adoption.

## Directory Layout

- `paper.md`: cross-domain synthesis and governance.
- `robotics/README.md`: autonomous control track blueprint.
- `medical/README.md`: monitoring/diagnostics track blueprint.
- `smart-home/README.md`: activity-recognition track blueprint.

## Cross-Domain Rules

- Treat all domain metrics as exploratory unless promoted to canonical sections.
- Keep metric definitions compatible with canonical reporting where overlap exists.
- Clearly label evidence class: `observed`, `pilot`, `planned`.

## Promotion Criteria

A domain finding can move to canonical sections only if:
1. runtime path is active and reproducible,
2. data and metric definitions are explicit,
3. limitations and validity boundaries are documented,
4. canonical sections are updated in the same change.
