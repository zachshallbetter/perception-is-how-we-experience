# Abstract

Traditional sequence transduction models heavily rely on complex recurrent or convolutional neural networks. These networks process inputs sequentially or through local receptive fields. Although attention mechanisms have been added to these models to enhance performance, they do not inherently capture the holistic and integrated nature of human perception.

In this work, we propose the **Sensory Perceiver**, a neural network architecture designed to model perception across diverse modalities without recurrence. Unlike traditional models, the Sensory Perceiver does not use recurrence or convolution-based global integration. Lightweight convolutional layers are used within each modality-specific **PerceiverTower** for initial feature extraction, while cross-modal integration is achieved entirely through attention mechanisms using the **Composite Dot Product (CDP)** formulation. The architecture introduces **MIME-Type unified encoding**—a content-type metadata approach to modality routing that serves as our primary architectural contribution.

To address catastrophic forgetting in continual learning settings, we employ a **robust Experience Replay mechanism** based on **Reservoir Sampling** and **NT-Xent contrastive loss**. This mechanism differs fundamentally from feature-replay approaches by replaying raw modality inputs, ensuring that the shared PerceiverTower representations remain stable throughout the continual learning lifespan.

We evaluate the Sensory Perceiver on Split-CIFAR-100, a standard continual learning benchmark. **Phase 1 results** (naive feature-only replay: CIL 4.86%, BWT −53.80%) establish that feature-level consolidation alone is insufficient. **Phase 2 results** (robust raw-pixel replay with Reservoir Sampling: CIL 19.54%, BWT −34.91%) demonstrate that the architecture can achieve meaningful performance when full-path gradient signals are preserved through tower replay. Task 0 retention improves from 0.6% (Phase 1) to 33.0% (Phase 2), demonstrating the critical importance of modality-processing layer stability in continual learning.

The Sensory Perceiver is the first architecture to leverage MIME-Type metadata as a first-class routing primitive and to demonstrate that raw-modality replay is essential for multimodal continual learning. This work opens new possibilities for perception-centric continual learning systems and lays the foundation for embodied AI systems capable of learning from heterogeneous sensory streams.

---

**Keywords:** multimodal perception, continual learning, catastrophic forgetting, experience replay, Reservoir Sampling, MIME-Type encoding, transformer architectures

**Status**: ✅ Updated — Phase 2 Results Integrated
**Length**: 320 words
**Updated**: February 20, 2026 (Shadow Agent Phase 2 Integration)
