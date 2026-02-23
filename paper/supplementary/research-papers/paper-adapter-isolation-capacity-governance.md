# Supplementary Research Draft

This draft is part of the paper supplementary track and is non-canonical until promoted.

# Paper Draft: Adapter Isolation and Capacity Governance in Continual Learning

## Abstract

We evaluate task-scoped adapter isolation with freeze/rollover APIs and a trainable-capacity coverage metric to manage continual-learning scalability.

## Method

At task boundary `k`:

1. Freeze adapter `A_k`.
2. Allocate adapter `A_{k+1}`.
3. Compute

`coverage_k = trainable_params_k / total_params_k`.

Study relation between `coverage_k` trajectory and forgetting metrics.

## Experimental plan

- Sequential task benchmark with and without adapter rollover integration.
- Coverage-threshold interventions (e.g., stop, expand, compress).
- Metrics: CIL, BWT, parameter growth, training stability.

## Code references

- `src/models/core.py`
- `paper/sections/10-concept-implementation-audit.md`

## Realization tasks

- [ ] Integrate adapter rollover in service loop.
- [ ] Add coverage-triggered policy hooks.
- [ ] Report coverage and adapter lineage in checkpoints.
