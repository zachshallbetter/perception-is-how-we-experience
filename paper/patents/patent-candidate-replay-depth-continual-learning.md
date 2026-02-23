# Patent Track Draft

This draft is incubated from docs and maintained in the paper corpus for filing workflow alignment.

# Candidate: Replay-Depth Continual Learning with Raw-Input Reservoir Sampling

Status: filing-ready draft
Priority: P1

## Problem

Feature-only replay often fails to preserve deep representation stability in continual learning.

## Proposed invention

A replay mechanism that stores raw modality inputs with reservoir sampling and supports dual replay objectives (`er`, `contrastive`) during training.

## Formal core

Reservoir rule for sample index `t`, capacity `B`:

- include with probability `min(1, B/(t+1))`
- replacement index uniformly sampled from `[0, t]`

Replay loss options:

- `L = L_ce(current) + L_ce(replay)` (`er`)
- `L = L_ce(current) + lambda * L_ntxent(current_embed, replay_embed)` (`contrastive`)

## Code evidence

- Replay buffer and reservoir updates: `src/models/core.py`
- Replay-mode training branches: `src/services/training_service.py`
- Scope-constrained experimental framing: `paper/sections/05-experiments.md`

## Claim skeleton

1. A continual-learning method that combines raw-input reservoir replay with selectable supervised and contrastive replay objectives to preserve cross-task representations.

## TODO

- [ ] Add explicit ablation artifacts for replay placement depth.
- [ ] Add replay weight scheduling policy and proofs.
- [ ] Add per-task retention guarantee reporting.
