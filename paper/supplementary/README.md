# Supplementary Papers Strategy

## Role in the Paper Stack

This directory extends the main paper with deeper technical analyses, domain-specific study designs, and systems-oriented implementation framing.

Supplementary documents are **non-canonical** for final benchmark claims. Canonical claim authority remains:
- `paper/sections/00-implementation-status-matrix.md`
- `paper/sections/05-experiments.md`
- `paper/sections/06-results.md`
- `paper/sections/10-concept-implementation-audit.md`

## Why Supplementary Exists

The main paper intentionally stays bounded to implementation-aligned, currently supported evidence.
Supplementary documents provide:
1. deeper explanation of design intent,
2. future experiment blueprints that are decision-ready,
3. domain adaptation templates for follow-on publications,
4. systems-level operational framing for scaling studies.

## Portfolio Structure

- `architecture-paper/`: architecture decomposition, scaling hypotheses, and ablation protocol templates.
- `orchestrator-paper/`: distributed training/orchestration framing, reproducibility controls, and operational test plans.
- `applications/`: robotics, medical, and smart-home benchmark-design packages.
- `research-papers/`: candidate publication drafts for replay placement, MIME contracts, and adapter isolation governance.

## Claim and Promotion Policy

A supplementary statement can be promoted to canonical sections only if all conditions are satisfied:
1. The experiment is reproducible through active runtime/service paths.
2. Evidence is represented in the status matrix with matching implementation state.
3. Metric definitions and evaluation protocol are explicitly documented.
4. The canonical section set is updated in the same change.

## Authoring Standards

- Use implementation-aligned status language: `implemented`, `partial`, `planned`.
- Distinguish **observed evidence** from **proposed experiments**.
- Avoid novelty or superiority claims without controlled evidence in canonical sections.
- Keep terminology consistent with canonical corpus (`sensor_streaming`, MIME boundary validation, modality-key forward dispatch).

## Update Workflow

When extending any supplementary paper:
1. Update the local supplementary README first (scope + assumptions).
2. Update the supplementary paper body (questions, protocol, risks, deliverables).
3. Run paper-wide consistency grep checks.
4. Regenerate monolith only if canonical section files changed.

## Current State Summary

- Architecture supplementary: expanded design rationale + evaluation blueprint; no canonical benchmark overrides.
- Orchestrator supplementary: expanded systems model + operational validation plan.
- Applications supplementary: expanded domain specs and promotion criteria for each domain track.
