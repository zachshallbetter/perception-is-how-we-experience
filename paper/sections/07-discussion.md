# Discussion

## 7.1 Core Mechanistic Insight

The current evidence supports a practical mechanism-level conclusion:

- Feature-only replay under-constrains shared representation layers.
- Raw-input replay forces replay gradients through the full feature stack.
- Reservoir sampling improves long-horizon sample retention fairness.
- The tuned-core continual path strengthens this result further without changing the core replay-centric story.

This explains the observed transition from near-total forgetting (Phase 1) to materially reduced forgetting (Phase 2) and then to the stronger tuned-core result.

## 7.2 What Is Implemented vs What Is Not

Implemented now:
- replay buffer with reservoir updates,
- replay mode control (`er`, `contrastive`),
- typed runtime/service orchestration,
- canonical `sensor_streaming` MIME case registry and validation.

Integrated but not yet empirically dominant:
- parameter-isolation primitives (task adapters with freeze/rollover APIs) exist in model code and tests, and sequential rollover is available in the service path, but these paths are not yet the strongest evidence-bearing benchmark result.

Planned:
- combined replay + full sequential adapter isolation benchmarks in the standard service path,
- real multimodal ingestion benchmarks at scale.

## 7.3 Naming and Protocol Semantics

The codebase now standardizes on `sensor_streaming` as canonical experiment/dataset identity. This reduces ambiguity and aligns runtime contracts, scripts, and documentation with a single term.

MIME semantics are formalized as a registry with explicit modality definitions and feature dimensions. At present this is used for contract-level validation and modality setup derivation; model forward dispatch remains modality-key based.

## 7.4 Research Consequences

1. **Replay quality and gradient path design matter as much as replay existence.**
2. **Protocolized modality definitions reduce hidden interface drift.**
3. **Parameter isolation is promising but should be presented as exploratory until it clearly outperforms the tuned replay-centric baseline.**

## 7.5 What The New Titan Results Add

The newer Titan runs sharpen the paper in two ways:

1. They provide a stronger paper-primary continual baseline than the earlier raw-input replay phase result alone.
2. They provide negative evidence that moderate EWC/SI increases and larger replay buffers do not automatically improve BWT.

This second point is important. The paper should not imply that adding regularization or mild structural isolation necessarily solves forgetting in this architecture. The current evidence says the opposite: replay-path design is the validated lever, while follow-on anti-forgetting knobs remain mixed.

## 7.6 Next Experimental Priorities

1. Complete the current BWT sweep and identify whether any late-running variant surpasses the tuned-core BWT baseline.
2. Run multiseed evaluation and confidence intervals for the tuned-core continual benchmark.
3. Benchmark sequential adapter rollover against the tuned-core result, not just against older phase baselines.
4. Add real multimodal ingestion/evaluation paths and report them separately from proxy datasets.
