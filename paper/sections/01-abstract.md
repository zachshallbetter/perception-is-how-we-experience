# Abstract

We present **Sensory Perceiver**, a multimodal architecture that combines modality-specific towers with shared cross-modal attention for continual learning settings. The system uses explicit modality identity and canonical MIME-style input case definitions for routing semantics, with a runtime registry for sensor-streaming modalities.

We evaluate continual learning behavior on Split-CIFAR-100 using a phase-based replay analysis. In **Phase 1** (feature-only replay), the model exhibits severe forgetting (CIL 4.86%, BWT −53.80%). In **Phase 2** (raw-input replay with Reservoir Sampling), retention improves substantially (CIL 19.54%, BWT −34.91%), including Task 0 retention recovery from 0.6% to 33.0%. These results indicate that replay signals must reach shared feature extractors, not only task heads.

The implementation includes a modernized runtime/service architecture with typed contracts, structured observability, and explicit modality registries. Parameter-isolation primitives (task adapters with freeze/rollover APIs) are implemented in the model and validated in targeted tests, but full sequential activation in the default service training loop remains future integration work.

Current multimodal support includes proxy audiovisual and sensor-streaming datasets. Real benchmark ingestion (for example AudioSet/Ego4D) and additional modality towers are planned next-stage work.

**Keywords:** multimodal perception, continual learning, catastrophic forgetting, reservoir sampling, MIME-type routing, parameter isolation
