# Supplementary Research Draft

This draft is part of the paper supplementary track and is non-canonical until promoted.

# Paper Draft: Replay Signal Placement in Continual Learning

## Abstract

We analyze how replay signal placement affects retention in continual learning. We compare feature-only replay against raw-input replay pathways that backpropagate through shared feature extractors.

## Method

Training objective:

`L_total = L_ce(x_t, y_t) + alpha * L_replay(x_r, y_r)`

with replay samples from a reservoir buffer.

For contrastive branch:

`L_total = L_ce + alpha * L_ntxent(z_t, z_r)`

where

`L_ntxent = -log( exp(sim(z_i, z_i+)/tau) / sum_j exp(sim(z_i, z_j)/tau) )`

## Hypothesis

Replay that re-enters shared modality towers yields better long-term retention than feature-only replay.

## Experimental plan

- Split-CIFAR-100 sequential tasks.
- Conditions: feature-only replay vs raw-input replay (`er`, `contrastive`).
- Metrics: final CIL, BWT, Task-0 retention, forgetting curve area.

## Code references

- `src/models/core.py`
- `src/services/training_service.py`
- `paper/sections/05-experiments.md`

## Realization tasks

- [ ] Add unified experiment script that toggles replay depth explicitly.
- [ ] Add statistical testing (paired seeds).
- [ ] Publish run manifests and hash-locked configs.
