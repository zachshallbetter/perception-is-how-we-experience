# Supplementary Paper 1: Architecture Design and Scalability

## Positioning

This track is the architecture-deepening companion to the main paper. It explains *why* specific model decomposition choices were made and how those choices should be evaluated without overstepping canonical evidence boundaries.

Canonical claim source for benchmark-level conclusions remains the main section corpus.

## Document Set

- `paper.md`: architecture argument, experiment blueprint, and evidence gating.
- This README: scope, protocol, and execution checklist.

## Scope

Included:
- PerceiverTower decomposition rationale.
- Cross-modal fusion behavior and design tradeoffs.
- Replay interaction with backbone stability.
- Scaling hypotheses for modality count, memory, and compute.

Excluded:
- New canonical benchmark claims.
- Production deployment guarantees.
- Dataset-scale claims not represented in canonical results.

## Core Questions

1. Which architectural components are structurally necessary versus optional?
2. Which ablations best isolate replay, fusion, and modality-identity effects?
3. How should scaling behavior be reported without conflating proxy and large-scale ingestion paths?

## Method Expectations

- Define explicit ablation matrices before running experiments.
- Use identical metric definitions to canonical sections for any overlapping tasks.
- Tag each finding as one of: `observed`, `hypothesis`, `future experiment`.

## Deliverables

1. Architecture-focused narrative paper (`paper.md`).
2. Ablation matrix template for future runs.
3. Decision log for promotion candidates to canonical sections.
