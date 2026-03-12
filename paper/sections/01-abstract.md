# Abstract

We present **Sensory Perceiver**, a multimodal architecture that combines modality-specific towers with shared cross-modal attention for continual learning settings. The system uses explicit modality identity and canonical MIME-style input case definitions for routing semantics, with a runtime registry for `sensor_streaming` modalities.

We evaluate continual learning behavior on Split-CIFAR-100 using a code-aligned progression of replay-centric experiments. In **Phase 1** (feature-only replay), the model exhibits severe forgetting (CIL 4.86%, BWT −53.80%). In **Phase 2** (raw-input replay with Reservoir Sampling), retention improves substantially (CIL 19.54%, BWT −34.91%), including Task 0 retention recovery from 0.6% to 33.0%. In the current tuned-core configuration, a 6-seed Titan Node run improves the mean result further to CIL 23.96% and BWT −30.39%. Together, these results indicate that replay signals must reach shared feature extractors, not only task heads, and that replay-path design dominates the current retention story.

The implementation includes a modernized runtime/service architecture with typed contracts, structured observability, and explicit modality registries. MIME semantics are enforced at runtime-contract boundaries for modality validation and setup derivation, while model forward dispatch remains modality-key based.

Parameter-isolation primitives (task adapters with freeze/rollover APIs) are implemented in the model and validated in targeted tests. Sequential activation is available as an opt-in service path, but current preliminary BWT sweeps with EWC and SI regularization have not yet surpassed the tuned-core replay result.

Current multimodal support includes proxy audiovisual and `sensor_streaming` datasets. The active perception-side evidence ladder now evaluates canonical typed sensor routing against a merged generic stream, followed by leave-one-out, corruption, noise, and capacity-matched controls. Real benchmark ingestion (for example AudioSet/Ego4D) and additional modality towers remain future-stage work.

Practically, the present contribution is therefore best understood as a blueprint for domain-specific perception systems: keep modality identity explicit, validate routing semantics at the runtime boundary, and make continual-learning safeguards act on the shared perceptual backbone rather than only late task heads.

**Keywords:** multimodal perception, continual learning, catastrophic forgetting, reservoir sampling, MIME-type routing, parameter isolation
