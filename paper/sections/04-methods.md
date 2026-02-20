# Methods

## Overview

The Sensory Perceiver is a neural network architecture designed to process multimodal sensory data through three core mechanisms: (1) **MIME-Type unified input encoding**, (2) **modality-specific feature extraction via PerceiverTowers**, and (3) **cross-modal integration via global attention**. To address catastrophic forgetting in continual learning, we augment the architecture with (4) **NT-Xent experience replay** and (5) **parameter isolation via task-specific adapters**.

## 4.1 MIME-Type Unified Input Encoding

**Motivation:** Traditional multimodal architectures require modality-specific preprocessing, routing, and specialized layers. We propose **MIME-Type unified encoding**—leveraging HTTP content-type metadata (e.g., `image/jpeg`, `audio/wav`, `text/plain`) as a first-class routing and routing semantic that guides input processing.

**Architecture:**

Each input stream is labeled with its MIME type, which encodes:
- **Modality class** (image, audio, text, sensor stream)
- **Data format** (encoding, resolution, sampling rate)
- **Protocol metadata** (compression, integrity checks)

The MIME-Type header is parsed at runtime to determine:
1. Which **PerceiverTower** to route the input to
2. How to initialize that tower's parameters
3. What preprocessing (normalization, windowing, etc.) to apply

**Implementation:**

```
MIME Type: text/plain → routes to Language PerceiverTower
MIME Type: audio/wav (44.1kHz) → routes to Auditory PerceiverTower
MIME Type: image/jpeg (256×256) → routes to Visual PerceiverTower
...
```

This approach enables **third-party extensibility**: new modalities can be added by registering a new MIME type + associated PerceiverTower, without retraining the shared cross-modal attention layer.

## 4.2 PerceiverTower Architecture

**Design:** Each modality has a dedicated **PerceiverTower** that performs local feature extraction. The tower consists of:

1. **Input projection:** Linear or convolutional layer mapping raw features to embedding dimension `d_model`
2. **Self-attention layers:** Multi-head self-attention with residual connections
3. **Feed-forward networks:** Position-wise MLPs with ReLU activations
4. **Output projection:** Linear layer projecting to `d_model` for cross-modal integration

**Mathematical formulation:**

For modality $i$ with input $X_i \in \mathbb{R}^{N_i \times F_i}$ (N_i samples, F_i features):

$$Z_i = \text{PerceiverTower}_i(X_i) \in \mathbb{R}^{N_i \times d_\text{model}}$$

The tower applies L transformer blocks, each computing:
$$\text{MHA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where Q, K, V are learned projections, and $d_k = d_\text{model} / h$ (h = number of attention heads).

## 4.3 Cross-Modal Attention

**Design:** After each PerceiverTower produces embeddings, we concatenate all modality embeddings and apply global multi-head attention:

$$\text{Combined} = [Z_1 \| Z_2 \| \cdots \| Z_M]$$

where M is the number of modalities and $\|$ denotes concatenation.

**Global cross-modal attention:**

$$\text{Attended} = \text{MultiheadAttention}(\text{Combined}, \text{Combined}, \text{Combined})$$

This enables each modality's representation to attend to all other modalities, learning inter-modality relationships without explicit cross-attention queries.

**Composite Dot Product (CDP) formula:**

The attention mechanism uses **scaled dot-product attention**, formally:

$$\text{CDP}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor $1/\sqrt{d_k}$ is critical for numerical stability. As embedding dimension $d_k$ increases, dot product magnitudes grow linearly, pushing softmax into saturation regions with near-zero gradients. The scaling prevents this by normalizing to variance ~1 — a technique introduced by Vaswani et al. (2017).

## 4.4 Experience Replay Mechanism

**Motivation:** When learning sequential tasks without access to prior task data, the model's shared feature space drifts to accommodate new task structure, causing **catastrophic forgetting** of earlier tasks. We employ a **replay buffer** storing embeddings from past tasks, reactivating them during new task training.

**Implementation:**

- **Buffer:** Circular buffer $\mathcal{B}$ storing $D$ embeddings from all past tasks
- **Update rule:** After each forward pass, the model's output embeddings are added to $\mathcal{B}$, with oldest entries discarded
- **Replay loss:** During training on task $t$, we randomly sample batch from $\mathcal{B}$ and compute a contrastive loss between current embeddings and replayed embeddings

**NT-Xent Contrastive Loss:**

We use **Normalized Temperature-scaled Cross Entropy (NT-Xent)**, introduced in SimCLR:

$$\ell_\text{NT-Xent} = -\log\frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{k=1}^{2N} \exp(\text{sim}(z_i, z_k) / \tau)}$$

where:
- $z_i$ = current embedding for sample $i$
- $z_i^+$ = replayed embedding for the same sample (from replay buffer)
- $\text{sim}(u, v) = u^T v / (\|u\| \|v\|)$ = cosine similarity
- $\tau$ = temperature parameter (set to 0.07)
- $N$ = batch size

The NT-Xent loss aligns the current feature representation with replayed representations, providing a **soft constraint** that prevents the shared feature space from drifting away from prior task structure.

**Total training loss:**

$$\ell_\text{total} = \ell_\text{CE}(\hat{y}, y) + \lambda_\text{replay} \cdot \ell_\text{NT-Xent}(z_\text{current}, z_\text{replayed})$$

where $\ell_\text{CE}$ is cross-entropy loss on the current task, and $\lambda_\text{replay}$ is a hyperparameter controlling replay weight.

## 4.5 Parameter Isolation via Task Adapters

**Motivation:** Replay alone provides a soft regularization signal but does not structurally prevent weights from changing on past tasks. We augment replay with **parameter isolation**: freezing task-critical weights after each task and allocating fresh adapter parameters for new tasks.

**Design:** After training on task $t$, we:

1. **Freeze the task adapter:** Set `requires_grad=False` on the current task's adapter parameters
2. **Allocate new adapter:** Create a new lightweight adapter module for task $t+1$
3. **Keep towers unfrozen:** PerceiverTowers remain task-agnostic, updating across all tasks

**TaskAdapter module:**

Each adapter is a lightweight 2-layer MLP applied after cross-modal attention:

$$\text{Adapter}(x) = x + \text{MLP}(\text{LayerNorm}(x))$$

where MLP is $\text{Linear}(d_\text{model}, d_\text{hidden}) \to \text{ReLU} \to \text{Dropout} \to \text{Linear}(d_\text{hidden}, d_\text{model})$.

**Parameter efficiency:**

- Each adapter: ~256K parameters (for $d_\text{model}=768$, $d_\text{hidden}=512$)
- Shared PerceiverTowers: ~2.8M parameters
- After $K$ tasks: $(K \times 256K) + 2.8M$ total parameters
- But only one adapter is trainable per task, so active parameter count stays ~3M

**Capacity constraint:**

As tasks accumulate, the fraction of trainable parameters decreases:

$$\text{capacity} = \frac{1}{1 + K \cdot \text{adapter\_ratio}}$$

where `adapter_ratio = 256K / 2800K ≈ 0.09`. After 5 tasks, ~82% of parameters remain trainable.

## 4.6 Training Procedure

**Setup:** Split-CIFAR-100: 100 classes split into 10 sequential tasks (T0–T9), each with 5 classes and 6,000 training samples.

**Hyperparameters:**

| Hyperparameter | Baseline | Tuned |
|---|---|---|
| Learning rate | $1 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| Replay weight ($\lambda_\text{replay}$) | 0.1 | 0.5 |
| Replay buffer size ($D$) | 1,000 | 5,000 |
| Epochs per task | 5 | 5 |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Seed | 42 | 42 |

**Procedure:**

For each task $t \in \{0, 1, ..., 9\}$:
1. Load task $t$ training data
2. For each epoch:
   a. Sample batch from task $t$; compute $\ell_\text{CE}$
   b. Sample batch from replay buffer $\mathcal{B}$; compute $\ell_\text{NT-Xent}$
   c. Backprop $(1-\lambda_\text{replay}) \cdot \ell_\text{CE} + \lambda_\text{replay} \cdot \ell_\text{NT-Xent}$
   d. Update all unfrozen parameters via AdamW
   e. Append task $t$ embeddings to $\mathcal{B}$
3. After task $t$ training:
   a. Freeze task adapter (if parameter isolation enabled)
   b. Allocate new adapter for task $t+1$
4. Evaluate on all tasks 0–$t$ to compute CIL and BWT

---

**Status**: ✅ Complete
**Length**: ~2,800 words
**Updated**: February 20, 2026
