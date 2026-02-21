# Related Work

The Sensory Perceiver builds on and extends several lines of research. We organize the relevant literature around four themes: the Perceiver family, efficient convolution, temporal and multimodal perception, and continual learning.

## 3.1 Transformer Foundations

The Transformer [1] introduced scaled dot-product attention — softmax(QKᵀ/√d_k)V — as a general-purpose sequence modeling primitive. By decoupling position encoding from attention, the Transformer enabled modality-agnostic processing that set the stage for all subsequent cross-modal architectures. Earlier alignment-based attention [4] and RNN-based sequence models [5, 7] established learned query-context alignment as the core mechanism for sequence understanding. Sensory Perceiver inherits the CDP formulation from [1], with 1/√d_k scaling critical to numerical stability at the dimension sizes used.

## 3.2 The Perceiver Family

**Perceiver** [2] addressed the quadratic attention cost of Transformers by introducing a cross-attention bottleneck: a small latent array (256–2048 latents) attends to a high-dimensional byte array of arbitrary size, decoupling input dimensionality from compute. This enabled a single architecture to process images, audio, point clouds, and video without modality-specific preprocessing.

**Perceiver IO** [3] extended this to arbitrary structured outputs via query-based output decoding: task-specific query arrays attend to the latent output, enabling dense predictions (optical flow, language, game playing) from the same backbone. Perceiver IO's modality embedding approach — concatenating learned Fourier position features with modality tokens — is the closest published precedent to MIME-Type encoding. The key distinction: Perceiver IO's features are byte-level position encodings, while MIME-Type encoding carries explicit content-type semantics enabling protocol-level routing and third-party extension.

**HiP (Hierarchical Perceiver)** [hip2022] introduced hierarchical kernel sharing and grouped cross-attention, partitioning inputs into local groups and attending hierarchically. Where Perceiver applies global cross-attention from all inputs to all latents, HiP scales to millions of inputs with near-linear complexity. For Sensory Perceiver, HiP-style hierarchical grouping represents the natural scaling path beyond the current ≤16 modality design.

**Perceiver AR** [perceiverar2022] added causal masking into Perceiver cross-attention, enabling autoregressive generation over sequences of 100k+ tokens by identifying each latent with a single output position. This is directly relevant to continual learning over temporal sensory streams — a Perceiver AR-style extension to Sensory Perceiver's replay mechanism could improve temporal coherence in online settings.

**Comparison**: Sensory Perceiver differs from the Jaegle et al. family in primary design goal: Jaegle et al. optimize for computational universality and throughput; Sensory Perceiver optimizes for biological plausibility, modality-typed routing, and continual learning. The PerceiverTower design — modality-specific front-ends feeding into shared cross-modal attention — preserves modality identity explicitly through the processing stack, unlike the single cross-attention bottleneck of [2, 3]. MIME-Type routing has no counterpart in any of [2], [3], or [hip2022].

## 3.3 Efficient and Dynamic Convolutions

PerceiverTower uses Conv2d front-ends for spatial modalities. The dynamic convolution literature is directly applicable to improving parameter efficiency across towers.

**Dynamic Convolution** [dynconv2019] replaced static Conv kernels with an input-dependent linear mixture of n kernels: W = α₁W₁ + … + αₙWₙ, where attention weights αᵢ are computed from the input. This allows the effective kernel to adapt per input without changing inference-time cost, at the expense of n× more parameters.

**SD-Conv** [sdconv2020] added L0 sparsity regularization to dynamic convolution, learning which kernel dimensions are truly input-dependent and collapsing the rest to static. This recovers parameter savings while preserving the accuracy gains of kernel mixture.

**ODConv** [odconv2022] extended dynamic convolution to four attention axes simultaneously: spatial size, input channel, output channel, and kernel number — capturing richer contextual variation than single-axis approaches at lower parameter count.

**KernelWarehouse** [kernelwarehouse2023] is the most parameter-efficient formulation: it partitions each kernel into m smaller cells, assembles a shared warehouse of cells, and allows each layer to linearly combine cells from the shared warehouse. This enables n > 100 kernels per layer with 10× fewer parameters than vanilla dynamic convolution. Shared warehouses across neighboring layers provide additional efficiency. For Sensory Perceiver's PerceiverTowers, adopting KernelWarehouse-style cell partitioning and cross-tower warehouse sharing would reduce per-tower parameter cost substantially, enabling more modalities without proportional parameter growth.

## 3.4 Temporal Modeling

Sensory Perceiver's current design treats temporal context through the experience replay buffer. The temporal modeling literature suggests several complementary approaches.

**TEA (Temporal Excitation and Aggregation)** [tea2020] proposed two complementary modules for video understanding: Motion Excitation (ME) for short-range temporal patterns via inter-frame feature comparison, and Multiple Temporal Aggregation (MTA) for long-range pooling. The two-scale design — distinguishing short-range motion signals from long-range context — informs replay buffer design, where recent episodes and distant episodes may warrant different weighting strategies.

**2D-TAN** [tan2019] introduced 2D temporal adjacent networks for language-guided moment localization. Mapping start-time × end-time onto a 2D adjacency map enables joint reasoning about temporal boundaries — relevant to Sensory Perceiver's potential extension to streaming temporal boundary detection.

**Temporal Perceiver** [temporalperceiver2022] applied Perceiver's latent compression to temporal boundary detection: a small set of latent queries attend to full video features and decode precise boundary positions. This query-to-boundary architecture is the most compatible design for extending Sensory Perceiver to streaming applications, and represents a direct future integration target.

## 3.5 Multimodal Fusion and Language Grounding

**FROMAGe** [fromage2021] demonstrated that frozen LLM representations can be extended to new modalities by adding a visual encoder that projects image features into the LLM's embedding space, with a special [RET] token for image retrieval. This freeze-and-extend strategy — adding new modalities without retraining existing representations — is philosophically aligned with the MIME-Type extension protocol, which adds new PerceiverTowers without retraining the shared attention mechanism.

## 3.6 Continual Learning

Continual learning — learning sequential tasks without catastrophic forgetting — is the primary use case for Sensory Perceiver's replay mechanism. The established baselines provide direct comparison targets for the experiments in §5–6.

**EWC (Elastic Weight Consolidation)** [kirkpatrick2017] applies parameter-level regularization using the Fisher information matrix to identify and protect task-critical weights. It requires no replay buffer but stores one Fisher matrix per task, scaling poorly to long sequences.

**GEM** [lopezmanzanares2017] and **A-GEM** [chaudhry2019] constrain gradient updates to not increase loss on episodic memory examples from prior tasks. A-GEM reduces GEM's computational cost by projecting onto the average gradient constraint rather than all individual constraints. Both represent the episodic memory class of continual learning, where Sensory Perceiver's NT-Xent replay buffer is an improvement: rather than treating the buffer as a constraint set, NT-Xent optimizes embedding alignment, providing a softer and more informative consolidation signal.

**DER++** [buzzega2020] showed that replaying network logits rather than labels provides a richer consolidation signal — logits encode the full soft probability distribution over all classes, preserving more information about earlier task structure. DER++ outperforms standard experience replay and EWC on most CIL benchmarks. The principle extends to Sensory Perceiver: replaying embedding logits (rather than contrastive pair embeddings) could strengthen the replay signal beyond the current NT-Xent formulation.

**NT-Xent** [cite15] — used as Sensory Perceiver's replay loss — was introduced in SimCLR as a self-supervised objective maximizing agreement between augmented views of the same sample. In the replay context, NT-Xent aligns current embeddings with replayed embeddings from the buffer, providing a soft constraint that preserves the geometry of earlier task representations. The experience replay sensitivity analysis of [chaudhry2019] and [buzzega2020] directly applies: as established in §6.1, the weight of the replay loss relative to the current-task CE loss is a first-order hyperparameter, with insufficient weighting (replay_weight = 0.1) producing near-total catastrophic forgetting.

## 3.6b Parameter Isolation and Structural Continual Learning

While replay and gradient constraints address forgetting through training mechanics, parameter isolation methods prevent forgetting structurally by restricting which weights can change per task. This class of approaches is orthogonal to replay-based methods and has complementary strengths: it eliminates forgetting by design at the cost of consuming task-specific capacity.

**PackNet** (Mallya & Lazebnik, 2018) learns a binary pruning mask per task using a learnable sparsity criterion. After training on task T, weights marked as critical (via gradient magnitude or Fisher information) are frozen; subsequent tasks train only on unpruned weights. This enforces strict parameter isolation without adding new capacity. The key trade-off: after K tasks, at most fraction (1 − ε)^K of the network remains trainable, where ε is the prune ratio per task. For Sensory Perceiver with modular PerceiverTowers, per-tower pruning masks would naturally partition the parameter space by modality, aligning with the architecture's design intent.

**SupSup** (Wen et al., 2020) learns a shared binary supermask over a fixed backbone, where each task activates a different subset of parameters. Rather than pruning (destructive) or growing (parameter-hungry), SupSup discovers task-specific submasks that maximize performance. This is parameter-efficient and maintains capacity, but requires that all tasks' optimal subnetworks fit within a single fixed backbone — a constraint that becomes tight as the number of tasks grows.

**HAT (Hard Attention to Task)** (Rusu et al., 2018) extends SupSup with task-specific "hard attention" masks applied to convolutional filters. Each task learns its mask during training; at test time, only the task-relevant subset of parameters is active. Unlike SupSup's binary discrete masks, HAT uses soft masks that are binarized at test time, enabling gradient-based learning during training.

**Progressive Neural Networks** (Rusu et al., 2016) take a fundamentally different approach: rather than sharing parameters across tasks, they allocate new columns per task with lateral connections to prior task columns. This guarantees no forgetting (new parameters can't affect old ones) but scales quadratically with the number of tasks — prohibitive for 50+ task sequences.

**Positioning parameter isolation for Sensory Perceiver**: The modular PerceiverTower design naturally aligns with per-tower parameter masks (a PackNet variant) or task-specific adapters (a lightweight alternative). After task T, freezing task-critical weights in the shared cross-modal attention and allocating new adapter parameters for task T+1 would combine the benefits of both architectures: PerceiverTowers remain task-agnostic feature extractors, while adapters provide task-specific fusion weights. This hybrid approach is more parameter-efficient than full isolation (task-specific PerceiverTowers) and more robust than pure replay.

## 3.7 Positioning

The Sensory Perceiver occupies a distinct position in this landscape. In this implementation, content-type (MIME-Type) metadata is treated as a first-class runtime primitive for modality-case validation and setup, alongside replay mechanisms that include an NT-Xent contrastive mode for continual learning across heterogeneous sensory streams. The closest work — Perceiver IO [3] — uses modality embeddings for a different goal (arbitrary input-output routing) without addressing continual-learning replay behavior in this manner. The architectural opportunities identified by this literature — query-based output decoding [3], Fourier position encodings [2, 3], KernelWarehouse efficiency [kernelwarehouse2023], and hierarchical scaling [hip2022] — are well-characterized future extensions, discussed in §7.9.
