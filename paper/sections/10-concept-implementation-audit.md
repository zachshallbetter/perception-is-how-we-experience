# 10. Concept-Implementation Audit

This section compares core paper concepts against current implementation behavior and evidentiary strength.

## 10.1 Concept Integrity Matrix

| Concept | Paper Position | Code Evidence | Status | Risk |
|---|---|---|---|---|
| Multimodal tower architecture with shared cross-modal integration | Core architecture supports modality-specific processing + shared integration | `src/models/core.py` (`PerceiverTower`, `SensoryPerceiver`) | Implemented | Low |
| Replay must reach shared feature extractors | Mechanistic claim from Phase 1 → Phase 2 | Phase-based results in `06-results.md`; replay applied in model/service (`_update_replay_buffer`, replay branches in `training_service.py`) | Implemented + evidenced (Split-CIFAR-100 scope) | Low |
| Reservoir sampling for long-horizon buffer fairness | Required buffer policy for continual learning | `src/models/core.py` (`_update_replay_buffer`) | Implemented | Low |
| Replay mode suite (`feature`, `er`, `contrastive`) | Current runtime supports feature-only baseline and raw-input replay objectives | `src/services/training_service.py` (`replay_mode` branches, `ContrastiveLoss`) | Implemented | Low |
| Parameter isolation via task adapters | API exists for freeze/rollover and trainable-coverage controls | `src/models/core.py` (`TaskAdapter.freeze`, `freeze_task`, `get_task_coverage`, `get_trainable_parameters`) | Implemented in model API | Medium |
| Parameter isolation as sequential service behavior | Available via explicit training flags (`adapter_rollover`, `continual_task_count`) | `src/services/training_service.py`, `src/train.py` | Implemented (opt-in path) | Low |
| MIME-style modality semantics as runtime source of truth | Canonical modality-case registry and validation at boundaries | `src/runtime/mime_cases.py`, `src/runtime/contracts.py` | Implemented | Low |
| MIME-string-driven dispatch inside model forward | Architectural ideal of semantic routing in-model | `src/models/core.py` forward dispatches by modality keys in input dict, not MIME strings | Partial by design | Medium |
| Modern runtime/service/observability architecture | Typed contracts + services + structured metrics/logging | `src/runtime/*`, `src/services/*`, `src/observability/*` | Implemented | Low |
| Bootstrap real-data benchmark ingestion | Bootstrap evidence is includable now | Speech Commands and WISDM sequence reports are present and packaged as bootstrap evidence | Implemented | Low |
| Canonical integrated real-data benchmark ingestion | Promoted only after bounded sequence completion | `real-audio-seq-002` and `real-sensor-seq-001` remain pending until completed | Partial / Pending | Medium |
| AudioSet media-backed integration | Explicit blocked state | Metadata present, media missing in active runtime path | Blocked | Medium |
| UbiqLog supervised integration | Explicit blocked state | Raw logs present, but no canonical supervised task definition | Blocked | Medium |

## 10.2 Concept-to-Concept Tensions

1. **MIME as routing primitive vs modality-key forward dispatch**
   - Current system uses MIME as canonical registry/validation at runtime boundaries.
   - Model execution still routes by modality dictionary keys.
   - Interpretation: semantic routing exists at the contract layer, not as direct MIME parsing in forward.

2. **Parameter isolation default vs opt-in execution**
   - Isolation APIs are present and functional in the model.
   - Service-path freeze/rollover execution is available through explicit CLI/runtime flags.
   - Interpretation: concept is integrated as an opt-in path rather than an always-on default.

3. **Multimodal architecture claims vs benchmark evidence scope**
   - Architecture and proxy multimodal datasets exist.
   - Reported quantitative CL outcomes are still centered on Split-CIFAR-100.
   - Interpretation: capability breadth exceeds currently published evidence breadth.

4. **Isolation availability vs isolation superiority**
   - Isolation/regularization paths are available and now directly runnable on Titan.
   - Preliminary BWT sweeps have not yet shown superiority over the tuned-core replay-centric baseline.
   - Interpretation: implementation availability should not be conflated with empirical dominance.

## 10.3 Normalized Claim Language (Recommended)

Use these canonical phrasings in the main paper to keep claims defensible:

- **Replay claim**: “Raw-input replay with reservoir sampling materially improves retention in the current Split-CIFAR-100 continual-learning setting.”
- **Isolation claim**: “Task-adapter isolation and sequential rollover are implemented and exposed as an opt-in service path, but their empirical advantage over the tuned replay-centric baseline remains under evaluation.”
- **MIME claim**: “MIME semantics are canonicalized at runtime-contract boundaries and drive modality validation/setup; model forward dispatch currently uses modality keys.”
- **Multimodal claim**: “Completed synthetic routing evidence exists, bootstrap real-data evidence exists on Speech Commands and WISDM, canonical integrated real-data sequences are pending, and blocked datasets remain explicit.”

## 10.4 Immediate Paper Synchronization Actions

1. Keep `00-implementation-status-matrix.md` and this section synchronized whenever implementation status changes.
2. Avoid unsupported novelty claims unless backed by explicit comparative evidence in-scope.
3. Keep benchmark evidence statements scoped to datasets actually exercised in current code paths.
