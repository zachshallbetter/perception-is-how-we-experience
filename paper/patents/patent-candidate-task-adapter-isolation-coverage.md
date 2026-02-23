# Patent Track Draft

This draft is incubated from docs and maintained in the paper corpus for filing workflow alignment.

# Candidate: Task Adapter Isolation with Trainable Coverage Accounting

Status: filing-ready draft
Priority: P2

## Problem

Continual learning systems need explicit parameter isolation controls with measurable capacity usage over tasks.

## Proposed invention

A task adapter lifecycle protocol:

1. Active adapter for current task.
2. Freeze on task completion.
3. Allocate next adapter for subsequent task.
4. Report trainable-parameter coverage as a capacity signal.

## Formal core

After task `k`:

- `coverage_k = trainable_params_k / total_params_k`

Coverage trend provides an operational budget signal for long-horizon training.

## Code evidence

- Adapter API and freeze/rollover: `src/models/core.py`
- Implementation status language: `paper/sections/10-concept-implementation-audit.md`

## Claim skeleton

1. A parameter-isolation method using task-scoped adapter rollover and quantitative trainable-capacity coverage accounting for continual learning governance.

## TODO

- [ ] Integrate `freeze_task()` into default service training lifecycle.
- [ ] Add coverage-based early warning thresholds.
- [ ] Add adapter lifecycle receipts in checkpoints.
