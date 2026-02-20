# Results

We report a phase-based continual-learning result profile on Split-CIFAR-100.

## 6.1 Phase 1: Feature-Only Replay (Failure Mode)

Observed metrics:
- CIL: **4.86%**
- BWT: **−53.80%**
- Task 0 retention after final task: **0.6%**

Interpretation:
- Replay signals constrained to higher layers are insufficient to preserve stable backbone representations.
- The result is near-total forgetting for early tasks.

## 6.2 Phase 2: Raw-Input Replay + Reservoir Sampling

Observed metrics:
- CIL: **19.54%**
- BWT: **−34.91%**
- Task 0 retention after final task: **33.0%**

Relative change from Phase 1:
- CIL: +14.68 pp
- BWT: +18.89 pp (less negative)
- Task 0 retention: +32.4 pp

Interpretation:
- Replaying raw inputs restores gradient flow through shared feature extractors and reduces representation drift.
- Reservoir sampling improves long-horizon retention fairness across earlier and later tasks.

## 6.3 Consolidated Table

| Phase | Replay design | CIL | BWT | Task 0 retention |
|---|---|---:|---:|---:|
| 1 | Feature-only replay | 4.86% | −53.80% | 0.6% |
| 2 | Raw-input + reservoir sampling | 19.54% | −34.91% | 33.0% |

## 6.4 Implementation-Alignment Notes

The reported phase contrast is aligned with current code capabilities:

- Reservoir sampling and replay modes are implemented in model/service code.
- Raw-input replay path is present in active training logic.
- Task-adapter isolation APIs are implemented, but full sequential adapter-rollover integration in default service training remains future integration work.

Therefore, this section claims a robust replay result, not a full replay+isolation combined benchmark result.

## 6.5 Limits of Current Evidence

1. Current quantitative claim is based on a narrow benchmark setting (Split-CIFAR-100 phase analysis).
2. Multiseed confidence intervals are still needed for stronger statistical reporting.
3. Real multimodal dataset ingestion (AudioSet/Ego4D-scale) is not yet an active training/eval path in this code branch.
