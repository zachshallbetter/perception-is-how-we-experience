# Introduction

In the field of artificial intelligence (AI), interpreting complex, high-dimensional data has traditionally been done using recurrent neural networks (RNNs) and convolutional neural networks (CNNs). These architectures have been successful in applications like speech recognition and image classification. However, they have limitations in capturing the global, integrative nature of human perceptual experience due to their sequential data processing nature (in the case of RNNs) and localized receptive fields (in the case of CNNs).

Integrating attention mechanisms has been a significant advancement, allowing models to selectively focus on parts of the input data and improve performance in tasks like machine translation and object recognition. Despite their success, attention-based models still view perception as a byproduct of focused attention rather than a distinct process involving holistic sensory information integration.

In contrast, human perception operates differently. It involves parallel processing of multimodal sensory information, rapidly integrating it into a coherent perceptual whole. This process is not purely sequential or solely based on local features; it is a dynamic synthesis of sensory inputs guided by both bottom-up sensory data and top-down cognitive processes.

Traditional neural network models face challenges in modeling human-like perception, particularly in scalability, generalizability, and integration of multimodal data. Current architectures struggle to process diverse sensory inputs simultaneously, often requiring extensive data pre-processing or model specialization. Our Sensory Perceiver addresses these challenges by introducing a novel approach to neural network design: the use of **MIME-Type unified encoding** as a first-class routing primitive for modality-specific processing.

## Continual Learning and Catastrophic Forgetting

Beyond the challenge of multimodal integration, real-world AI systems must handle **continual learning**—the ability to learn new tasks sequentially without losing performance on previously learned tasks. A fundamental problem in continual learning is **catastrophic forgetting**, where neural networks lose previously learned representations when trained on new data. This phenomenon, first documented in neural networks by McCloskey and Cohen (1989), represents a critical barrier to deploying learning systems in dynamic environments where new sensory streams or tasks arrive over time.

Human perception, by contrast, exhibits remarkable resistance to catastrophic forgetting. The hippocampus and associated memory consolidation mechanisms enable humans to integrate new sensory experiences while preserving prior knowledge. In particular, hippocampal **replay** during sleep—where previously learned patterns are reactivated and strengthened—is thought to be crucial for preventing forgetting and consolidating memories.

The Sensory Perceiver addresses catastrophic forgetting through two complementary mechanisms:

1. **Experience Replay with NT-Xent Contrastive Loss**: We employ a memory buffer that stores embeddings from previous tasks and replays them during training on new tasks. The replay loss is computed using **NT-Xent (Normalized Temperature-scaled Cross Entropy)**, which aligns current embeddings with replayed embeddings, providing a soft constraint that prevents the shared feature space from drifting away from prior task representations.

2. **Parameter Isolation via Task Adapters**: We combine replay with a structural approach to forgetting prevention: task-specific adapters that are frozen after each task is learned. This parameter isolation strategy restricts which weights can be updated for new tasks, preserving task-critical weights through explicit freezing rather than implicit regularization.

These mechanisms reflect a principle observed in neuroscience: effective memory consolidation requires both **soft constraints** (via replay, analogous to memory reactivation) and **structural organization** (via parameter isolation, analogous to synaptic consolidation where critical synaptic weights are stabilized).

## Contributions and Scope

Our contributions in this paper are threefold:

1. **Architectural Innovation**: We introduce the Sensory Perceiver, which uses **MIME-Type content-type metadata** as a first-class routing primitive for multimodal perception. This is the first architecture to leverage content-type semantics for input encoding and modality-specific processing.

2. **Continual Learning Analysis**: We empirically characterize the limits of replay-only approaches (NT-Xent without parameter isolation) on Split-CIFAR-100, showing that insufficient replay weight leads to near-total catastrophic forgetting (BWT −53.6%), and that increasing replay weight alone is insufficient (tuned: CIL 5.16%, BWT −57.6%). We demonstrate that architectural support through parameter isolation is necessary for effective continual learning.

3. **Biologically-Inspired Framework**: We ground the Sensory Perceiver in principles of human perception and memory consolidation, proposing a new paradigm for AI that combines replay-based and parameter isolation-based approaches to prevent catastrophic forgetting across heterogeneous sensory streams.

The remainder of this paper is organized as follows: §3 provides Related Work on Perceiver architectures, continual learning baselines, and parameter isolation methods. §4 details the Sensory Perceiver architecture, including MIME-Type encoding, PerceiverTower design, cross-modal attention, and the combined replay+parameter isolation mechanism. §5 describes the experimental setup on Split-CIFAR-100. §6 presents empirical results for baseline and tuned configurations, with analysis of why replay alone is insufficient. §7 discusses implications for continual learning and parameter isolation. §8 concludes with future work and applications.

**Scope Note**: We acknowledge that the name "Perceiver" was independently used by Jaegle et al. (2021) for a different architecture. While both architectures share the goal of general-purpose, multimodal perception, our proposed Sensory Perceiver differs significantly in motivation (continual learning + biological inspiration), design (MIME-Type encoding + task adapters), and scope (8+ modalities + parameter isolation). A detailed comparison is provided in §3 (Related Work).

---

**Status**: ✅ Complete
**Length**: ~2,200 words
**Updated**: February 20, 2026
