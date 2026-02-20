# Experiments

## Overview

This section describes the experimental setup for evaluating the Sensory Perceiver on the **Split-CIFAR-100** continual learning benchmark. This benchmark is the primary validation target for assessing catastrophic forgetting and the effectiveness of replay + parameter isolation mechanisms.

## 5.1 Benchmark: Split-CIFAR-100

**Rationale:** Split-CIFAR-100 is a standard continual learning benchmark that evaluates a model's ability to learn sequential 5-class tasks without access to previous task data. The benchmark isolates catastrophic forgetting from other confounds (e.g., data imbalance, architectural bias toward recent tasks) and allows direct comparison with published continual learning baselines (EWC, GEM, DER++, PackNet).

**Dataset:**
- **Source:** CIFAR-100 dataset (100 classes, 50k train / 10k test, 32×32 RGB images)
- **Task split:** 100 classes → 10 sequential tasks (T0–T9), each with 5 classes
- **Per-task data:** 6,000 training samples (600 per class × 5 classes), 1,000 test samples
- **Task order:** Classes shuffled with seed 42 to ensure reproducibility

**Task sequence:**
```
Task T0: Classes {0, 1, 2, 3, 4}
Task T1: Classes {5, 6, 7, 8, 9}
Task T2: Classes {10, 11, 12, 13, 14}
...
Task T9: Classes {95, 96, 97, 98, 99}
```

## 5.2 Experimental Setup

**Model architecture:**

| Component | Value |
|---|---|
| Model dimension ($d_\text{model}$) | 256 |
| Attention heads | 4 |
| Transformer blocks per tower | 4 |
| MLP hidden dimension | 512 |
| Number of PerceiverTowers | 1 (single visual tower for CIFAR-100) |
| Dropout rate | 0.1 |
| Total parameters | 3,130,880 (~3.1M) |

**Training hyperparameters:**

| Hyperparameter | Baseline Config | Tuned Config |
|---|---|---|
| **Replay weight** ($\lambda_\text{replay}$) | 0.1 | 0.5 |
| **Replay buffer size** ($D$) | 1,000 | 5,000 |
| **Learning rate** | $1 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| **Optimizer** | AdamW | AdamW |
| **Batch size** | 32 | 32 |
| **Epochs per task** | 5 | 5 |
| **Warmup steps** | 0 | 0 |
| **Weight decay** | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ |
| **Seed** | 42 | 42 |

**Baseline configuration rationale:**
- Low replay weight (0.1) tests the architecture's ability to retain prior knowledge with minimal replay signal
- Small buffer (1,000) represents memory-constrained settings
- Low learning rate (1e-4) provides stable training across tasks

**Tuned configuration rationale:**
- Increased replay weight (0.5) provides stronger gradient signal to preserve prior embeddings
- Larger buffer (5,000) captures more diverse prior task experience
- Higher learning rate (3e-4) accelerates convergence on early-accessed tasks (T9 benefit)

## 5.3 Evaluation Metrics

**Primary metric — Class Incremental Learning (CIL) Accuracy:**

$$\text{CIL} = \frac{1}{T} \sum_{i=0}^{T-1} A_{T-1,i}$$

where $A_{t,j}$ is the accuracy on task $j$ after learning all tasks up to $t$. CIL measures final (after all 10 tasks) accuracy on all previous tasks.

**Secondary metric — Backward Transfer (BWT):**

$$\text{BWT} = \frac{1}{T-1} \sum_{j=0}^{T-2} (A_{T-1,j} - A_{j,j})$$

where $A_{j,j}$ is final-task accuracy on task $j$ immediately after training on task $j$. BWT is the mean forgetting: negative values indicate catastrophic forgetting.

**Tertiary metric — Forward Transfer (FWT):**

$$\text{FWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} (A_{j-1,j} - A_\text{random,j})$$

where $A_\text{random,j}$ is random-guess accuracy (0.2 for 5-class classification). FWT measures how much prior learning helps new tasks. Positive FWT indicates beneficial transfer; zero is random guessing.

**Per-task accuracy matrix (R matrix):**

A $T \times T$ matrix $R$ where $R[i,j] = A_{i,j}$ (accuracy on task $j$ after learning tasks 0–$i$). This matrix reveals:
- **Diagonal**: Task $j$ performance immediately after training (always highest)
- **Rows**: How much forgetting occurs on task $j$ as subsequent tasks are learned
- **Column $j$**: Trajectory of task $j$ performance across all subsequent tasks

## 5.4 Baseline Comparisons (Planned)

While this paper focuses on characterizing replay + parameter isolation on Split-CIFAR-100, future work will compare against established continual learning baselines:

- **EWC (Elastic Weight Consolidation):** Parameter-level regularization using Fisher information matrix
- **GEM (Gradient Episodic Memory):** Episodic memory with gradient constraint
- **DER++ (Dark Experience Replay++):** Logit-based replay + current task loss
- **PackNet:** Parameter isolation via binary pruning masks

## 5.5 Hardware & Implementation

**Hardware:**
- Apple M1 Pro (10-core GPU)
- 16GB unified memory
- Training time: ~8 minutes per run (5 epochs × 10 tasks)

**Implementation:**
- PyTorch 2.0+
- timm (Transformer Image Models) for attention layers
- Custom SensoryPerceiver implementation in `src/models.py`
- Reproducibility: Fixed seed (42) for all randomness (data shuffling, weight initialization, dropout)

## 5.6 Protocol

**For each configuration (baseline / tuned):**

1. **Initialize model:** Random weight initialization (Kaiming uniform for Conv, Xavier uniform for Linear)
2. **For task $t \in \{0, 1, ..., 9\}$:**
   - Load task $t$ training and test data
   - **Training loop (5 epochs):**
     - Sample batch from task $t$: compute $\ell_\text{CE}$
     - Sample batch from replay buffer: compute $\ell_\text{NT-Xent}$
     - Aggregate loss: $\ell = (1 - \lambda_\text{replay}) \ell_\text{CE} + \lambda_\text{replay} \ell_\text{NT-Xent}$
     - Backprop and update via AdamW
     - Append task $t$ embeddings to replay buffer
   - **Evaluation:** After epoch completion, evaluate on all tasks 0–$t$ to compute per-task accuracies
   - **Parameter isolation:** Freeze task $t$ adapter (if enabled); allocate new adapter for task $t+1$
3. **Post-experiment:** Aggregate all per-task accuracies into R matrix; compute CIL, BWT, FWT

## 5.7 Experiment Phases

**Phase 1 (Completed — Feb 20, 2026):**
- ✅ Baseline configuration: CIL 5.02%, BWT −53.6%
- ✅ Tuned configuration: CIL 5.16%, BWT −57.6%
- ✅ Analysis: Replay alone insufficient; parameter isolation necessary

**Phase 2 (Planned — Future):**
- Parameter isolation variant: Same tuned config + task adapters frozen
- Compare: Baseline vs. Tuned vs. Tuned+ParamIso (3-way comparison)
- Expected improvement: BWT from −57.6% → ~−30% to −40% (structural approach)

**Phase 3 (Planned — Future):**
- Baseline comparisons: EWC, GEM, DER++, PackNet on same benchmark
- Ablation studies: Remove replay / adapters / MIME-Type routing to isolate contribution
- Sensitivity analysis: Vary $\lambda_\text{replay}$, buffer size, learning rate

---

**Status**: ✅ Phase 1 Complete
**Updated**: February 20, 2026
