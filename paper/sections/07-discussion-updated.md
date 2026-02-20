# Discussion

## 7.1 Core Mechanistic Insight

The phase comparison supports a practical mechanism-level conclusion:

- Feature-only replay under-constrains shared representation layers.
- Raw-input replay forces replay gradients through the full feature stack.
- Reservoir sampling improves long-horizon sample retention fairness.

This explains the observed transition from near-total forgetting (Phase 1) to materially reduced forgetting (Phase 2).

## 7.2 What Is Implemented vs What Is Not

Implemented now:
- replay buffer with reservoir updates,
- replay mode control (`er`, `contrastive`),
- typed runtime/service orchestration,
- canonical sensor-streaming MIME case registry and validation.

Partially integrated:
- parameter-isolation primitives (task adapters with freeze/rollover APIs) exist in model code and tests, but are not yet the dominant default service training flow.

Planned:
- combined replay + full sequential adapter isolation benchmarks in the standard service path,
- real multimodal ingestion benchmarks at scale.

## 7.3 Naming and Protocol Semantics

The codebase now standardizes on `sensor_streaming` as canonical experiment/dataset identity. This reduces ambiguity and aligns runtime contracts, scripts, and documentation with a single term.

MIME semantics are formalized as a registry with explicit modality definitions and feature dimensions. At present this is used for contract-level validation and modality setup derivation; model forward dispatch remains modality-key based.

## 7.4 Research Consequences

1. **Replay quality and gradient path design matter as much as replay existence.**
2. **Protocolized modality definitions reduce hidden interface drift.**
3. **Parameter isolation is promising but should be presented as partially integrated until service-path activation is complete.**

## 7.5 Next Experimental Priorities

1. Activate and benchmark sequential adapter rollover in the default service training pipeline.
2. Run multiseed evaluation and confidence intervals for phase metrics.
3. Add real multimodal ingestion/evaluation paths and report results separately from proxy datasets.
