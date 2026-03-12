# Abstract

We present **Sensory Perceiver**, a multimodal architecture that combines modality-specific towers with shared cross-modal attention for continual learning settings. The system uses explicit modality identity and canonical MIME-style input case definitions for routing semantics, with a runtime registry for `sensor_streaming` modalities.

We evaluate continual learning behavior on Split-CIFAR-100 using a code-aligned progression of replay-centric experiments. In **Phase 1** (feature-only replay), the model exhibits severe forgetting (CIL 4.86%, BWT −53.80%). In **Phase 2** (raw-input replay with Reservoir Sampling), retention improves substantially (CIL 19.54%, BWT −34.91%), including Task 0 retention recovery from 0.6% to 33.0%. In the current tuned-core configuration, a 6-seed Titan Node run improves the mean result further to CIL 23.96% and BWT −30.39%. Together, these results indicate that replay signals must reach shared feature extractors, not only task heads, and that replay-path design dominates the current retention story.

The implementation includes a modernized runtime/service architecture with typed contracts, structured observability, and explicit modality registries. MIME semantics are enforced at runtime-contract boundaries for modality validation and setup derivation, while model forward dispatch remains modality-key based.

Parameter-isolation primitives (task adapters with freeze/rollover APIs) are implemented in the model and validated in targeted tests. Sequential activation is available as an opt-in service path, but current preliminary BWT sweeps with EWC and SI regularization have not yet surpassed the tuned-core replay result.

Current multimodal support includes proxy audiovisual and `sensor_streaming` datasets. The active perception-side evidence ladder now evaluates canonical typed sensor routing against a merged generic stream, followed by leave-one-out, corruption, noise, and capacity-matched controls. Real benchmark ingestion (for example AudioSet/Ego4D) and additional modality towers remain future-stage work.

**Keywords:** multimodal perception, continual learning, catastrophic forgetting, reservoir sampling, MIME-type routing, parameter isolation
# Introduction

In the field of artificial intelligence (AI), interpreting complex, high-dimensional data has traditionally been done using recurrent neural networks (RNNs) and convolutional neural networks (CNNs). These architectures have been successful in applications like speech recognition and image classification. However, they have limitations in capturing the global, integrative nature of human perceptual experience due to their sequential data processing nature (in the case of RNNs) and localized receptive fields (in the case of CNNs).

Integrating attention mechanisms has been a significant advancement, allowing models to selectively focus on parts of the input data and improve performance in tasks like machine translation and object recognition. Despite their success, attention-based models still view perception as a byproduct of focused attention rather than a distinct process involving holistic sensory information integration.

In contrast, human perception operates differently. It involves parallel processing of multimodal sensory information, rapidly integrating it into a coherent perceptual whole. This process is not purely sequential or solely based on local features; it is a dynamic synthesis of sensory inputs guided by both bottom-up sensory data and top-down cognitive processes.

Traditional neural network models face challenges in modeling human-like perception, particularly in scalability, generalizability, and integration of multimodal data. Current architectures struggle to process diverse sensory inputs simultaneously, often requiring extensive data pre-processing or model specialization. Our Sensory Perceiver addresses these challenges by introducing **MIME-style modality semantics** as a routing primitive with runtime-contract validation and setup derivation.

In the present implementation, Sensory Perceiver should be understood as a machine-perception framework with two inseparable goals: (1) to represent heterogeneous sensory streams as explicit, typed modality cases that can be validated and configured at runtime, and (2) to learn over time without destroying earlier perceptual structure. The first goal motivates the modality-specific towers, shared cross-modal integration, and canonical `sensor_streaming` MIME registry; the second motivates the replay-centric continual-learning experiments that currently provide the strongest evidence base. In other words, continual learning is not separate from the project’s notion of perception, but the temporal-stability requirement inside it.

## Continual Learning and Catastrophic Forgetting

Beyond the challenge of multimodal integration, real-world AI systems must handle **continual learning**—the ability to learn new tasks sequentially without losing performance on previously learned tasks. A fundamental problem in continual learning is **catastrophic forgetting**, where neural networks lose previously learned representations when trained on new data. This phenomenon, first documented in neural networks by McCloskey and Cohen (1989), represents a critical barrier to deploying learning systems in dynamic environments where new sensory streams or tasks arrive over time.

Human perception, by contrast, exhibits remarkable resistance to catastrophic forgetting. The hippocampus and associated memory consolidation mechanisms enable humans to integrate new sensory experiences while preserving prior knowledge. In particular, hippocampal **replay** during sleep—where previously learned patterns are reactivated and strengthened—is thought to be crucial for preventing forgetting and consolidating memories.

The Sensory Perceiver addresses catastrophic forgetting through replay-centric mechanisms in the current runtime:

1. **Experience Replay with Reservoir Sampling**: We employ a bounded replay buffer with reservoir updates, so early and late task samples remain represented over long training horizons.

2. **Replay Mode Variants**: The training pipeline supports (a) standard replay with cross-entropy on replayed samples and (b) contrastive replay based on NT-Xent-style feature alignment.

Task-adapter parameter-isolation APIs are implemented in the model and tested in isolated benchmarks. Sequential rollover is available in the service path as an opt-in configuration, but the strongest current evidence remains replay-centric rather than isolation-centric.

## Contributions and Scope

Our contributions in this paper are fourfold:

1. **Architectural Innovation**: We introduce a modular PerceiverTower + cross-modal attention design with explicit modality identity and MIME-style routing semantics.

2. **Replay Mechanism Analysis**: We empirically characterize feature-only replay failure, raw-input replay recovery, and the current tuned-core continual-learning regime on Split-CIFAR-100.

3. **Modern Runtime Implementation**: We provide typed runtime contracts, structured errors, service-layer orchestration, and observability instrumentation suitable for reproducible experimentation.

4. **Canonical `sensor_streaming` MIME Registry and Evidence Ladder**: We formalize `sensor_streaming` input cases (modality, MIME type, feature dimensionality) as a single runtime source of truth and expose a bounded perception-side experiment ladder comparing typed routing against merged routing and subsequent stress tests.

The remainder of this paper is organized as follows: §3 provides related work. §4 details architecture and implementation-aligned methods. §5 describes experimental setup and implementation status. §6 presents the empirical progression from failure mode to current best result. §7 discusses implications, gaps, and next-stage roadmap. §8 concludes.

**Scope Note**: We acknowledge that the name "Perceiver" was independently used by Jaegle et al. (2021) for a different architecture. While both architectures share the goal of general-purpose multimodal perception, our current implementation focuses on continual-learning behavior, MIME-style modality semantics, and service-oriented runtime architecture. A detailed comparison is provided in §3 (Related Work).
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
# Methods (Code-Aligned)

## 4.1 System Architecture

The implemented architecture consists of:

1. **Modality-specific PerceiverTowers** for local feature extraction.
2. **Shared cross-modal attention** over concatenated modality embeddings.
3. **Optional task adapters** for parameter isolation primitives.
4. **Replay buffer** with reservoir updates for continual learning.

Core model implementation: `src/models/core.py`.

## 4.2 MIME-Style Input Cases and Routing Semantics

The runtime formalizes `sensor_streaming` modality cases as a canonical registry:

- `smartwatch` → `application/vnd.sensory.smartwatch.timeseries+tensor` (7 features)
- `room_env` → `application/vnd.sensory.environment.timeseries+tensor` (4 features)
- `robot_arm` → `application/vnd.sensory.robot-arm.timeseries+tensor` (9 features)

Registry implementation: `src/runtime/mime_cases.py`.

Important implementation detail: MIME semantics are enforced at runtime-contract boundaries for modality validation and setup derivation, while the model forward path dispatches by **modality keys** in the input dictionary.

## 4.3 Continual Learning Replay

### 4.3.1 Reservoir Sampling Buffer Management

For each sample index \(t\) with buffer capacity \(B\):
- if buffer not full, append;
- else replace a random slot with probability \(B/(t+1)\).

This ensures each observed sample has equal retention probability over time.

### 4.3.2 Replay Modes

The training service exposes three replay modes:

- `er`: cross-entropy on replayed raw samples.
- `contrastive`: NT-Xent-style feature alignment between current and replayed features.
- `feature`: classifier-level replay from a feature store (ablation baseline).

Replay is computed from raw modality inputs, so gradients can traverse the backbone when replayed batches are processed.

## 4.4 Parameter Isolation: Implemented API vs Active Pipeline

Implemented model capabilities:
- `TaskAdapter.freeze()`
- `SensoryPerceiver.freeze_task()` (freeze current adapter + allocate next)
- `get_trainable_parameters()`, `get_task_coverage()`

Current status:
- APIs are implemented and test-validated.
- Sequential activation is available in the service path via `adapter_rollover` with configurable `continual_task_count`.

## 4.5 Runtime and Services

The codebase uses a service-oriented runtime:

- `training_service.py`
- `evaluation_service.py`
- `orchestration_service.py`

with:
- typed runtime contracts (`TrainRuntimeConfig`, `EvalRuntimeConfig`, `OrchestratorRuntimeConfig`),
- typed error codes (`AppError`, `ErrorCode`),
- schema validation for persisted training config,
- structured JSON events + metrics snapshots.

## 4.6 Datasets in Current Implementation

Implemented in dataloader factory:
- `continual` (Split-CIFAR / Permuted-MNIST)
- `sensor_streaming` (generated multistream sensor dataset)
- `multimodal` (audio-visual proxy dataset)

Real benchmark ingestion pipelines (e.g., AudioSet/Ego4D) remain future-stage work and are not yet in the active runtime path.
# Experiments

## 5.1 Experimental Scope

This paper’s empirical claims are grounded in **Split-CIFAR-100 continual learning** with a code-aligned progression of experiments:

- **Phase 1**: feature-only replay (insufficient).
- **Phase 2**: raw-input replay + reservoir sampling (improved stability).
- **Current tuned core**: the paper-primary `scripts/run_local.py --tuned` configuration on Titan Node.
- **Preliminary BWT sweep**: replay-buffer, EWC, and SI follow-on ablations, interpreted as exploratory rather than final paper-primary evidence.

## 5.2 Runtime and Reproducibility Setup

- Runtime entrypoints: `src.train`, `src.eval`, `src.experiments.experiment_orchestrator`
- Service layer: `src/services/*`
- Runtime contracts + typed errors: `src/runtime/*`
- Observability: structured JSON events and metrics snapshots in `src/observability/*`

Primary reproducibility controls:
- fixed seed configuration,
- persisted training config schema,
- deterministic experiment/dataset validation via runtime contracts.
- Titan Node execution with separate core and baseline environments as documented in `docs/training-guide.md`.

## 5.3 Datasets and Modalities

### Active benchmark path (used for reported quantitative CL results)
- Split-CIFAR-100 (`experiment=continual`, `dataset=split_cifar`)

### Implemented but not the primary source of the reported CL table
- `sensor_streaming` generated dataset (`experiment=sensor_streaming`, `dataset=sensor_streaming`)
- Audio-visual proxy dataset (`experiment=multimodal`, `dataset=audiovisual`)

### Active perception-side benchmark ladder
- `sensor_streaming` canonical-vs-merged routing ablation
  - `canonical`: typed sensor modalities (`smartwatch`, `room_env`, `robot_arm`) derived from the MIME registry
  - `merged`: a single generic 20-feature sensor stream without typed modality separation
  - generator properties:
    - compositional class codes rather than one globally unique signal per modality
    - shared nuisance structure across samples
    - shifted validation regime rather than exact train-distribution reuse
- follow-on sensor stress tests:
  - leave-one-modality-out (`drop_smartwatch`, `drop_room_env`, `drop_robot_arm`)
  - MIME-binding corruption (`corrupt_bindings`)
  - single-modality noise robustness (`noise_smartwatch`, `noise_room_env`, `noise_robot_arm`)
  - capacity-matched canonical-vs-merged control

This is the most direct experiment for testing whether the typed modality semantics materially matter beyond runtime correctness. The follow-on sequence can now be driven by a bounded Titan-local supervisor, which waits for the baseline family to finish, aggregates the comparison, and then launches stress tests in a fixed order rather than an open-ended search.

## 5.4 Evaluation Metrics

Primary continual-learning metrics:
- **CIL** (final mean accuracy across tasks)
- **BWT** (backward transfer / forgetting)
- Task-retention traces (e.g., Task 0 retention after final task)

For current Titan campaigns, we also use:
- multiseed mean comparison for the tuned-core result,
- ablation ranking by final BWT for follow-on sweeps.

## 5.5 Implementation Status Constraints

To avoid over-claiming:

1. Task-adapter rollover is implemented in the service path and can be enabled via `adapter_rollover` with `continual_task_count > 1`.
2. MIME routing is implemented as canonical modality-case registry + validation, while model dispatch uses modality keys in input dicts.
3. Real multimodal ingestion pipelines (AudioSet/Ego4D-scale training integration) remain future work.
4. Preliminary BWT sweeps are interpreted as follow-on evidence only if they beat the tuned-core continual baseline; otherwise they are diagnostic evidence about what does not improve forgetting.
5. The sensor/MIME lane is bounded by explicit decision states (`canonical-dominant`, `capacity-confounded`, `competitive`, `merged-favored`) and should not be described as an unconstrained autonomous search for perfect results.

These constraints are intentional and reflected in the Results/Discussion interpretation.
# Results

We report a code-aligned continual-learning result progression on Split-CIFAR-100.

Generated paper assets for this section are packaged under `paper/figures/generated/results_package/`:

- `table_results_overview.png`
- `figure_phase_progression.png`
- `figure_tuned_core_multiseed.png`
- `figure_bwt_sweep_ranked.png`
- `captions.md`
- `latex_results_package.tex`
- `package_manifest.json`

At present, the packaged quantitative story is still centered on the continual-learning lane. The active `sensor_streaming` routing ladder is the next perception-side evidence path, but those results are not claimed here until the baseline aggregation and stress-test sequence complete.

## 6.1 Phase 1: Feature-Only Replay (Failure Mode)

Observed metrics:
- CIL: **4.86%**
- BWT: **−53.80%**
- Task 0 retention after final task: **0.6%**

Interpretation:
- Replay signals constrained to higher layers are insufficient to preserve stable backbone representations.
- The result is near-total forgetting for early tasks.

## 6.2 Phase 2: Raw-Input Replay + Reservoir Sampling

Observed metrics:
- CIL: **19.54%**
- BWT: **−34.91%**
- Task 0 retention after final task: **33.0%**

Relative change from Phase 1:
- CIL: +14.68 pp
- BWT: +18.89 pp (less negative)
- Task 0 retention: +32.4 pp

Interpretation:
- Replaying raw inputs restores gradient flow through shared feature extractors and reduces representation drift.
- Reservoir sampling improves long-horizon retention fairness across earlier and later tasks.

## 6.3 Current Tuned-Core Continual Result

We next evaluate the current paper-primary continual-learning path on Titan Node using `scripts/run_local.py --tuned` over 6 seeds.

Observed multiseed result:
- CIL mean: **23.96%**
- BWT mean: **−30.39%**
- FWT mean: **0.00%**
- Task 0 retention mean: **25.1%**

Per-seed values:
- seed 47: CIL **23.60%**, BWT **−29.44%**
- seed 48: CIL **22.98%**, BWT **−32.39%**
- seed 49: CIL **23.95%**, BWT **−30.54%**
- seed 50: CIL **24.16%**, BWT **−31.15%**
- seed 51: CIL **24.32%**, BWT **−30.31%**
- seed 52: CIL **24.74%**, BWT **−28.49%**

Interpretation:
- The tuned-core path materially improves on the earlier raw-input replay phase result.
- Variance across seeds is modest, which strengthens the implementation-level credibility of the result.
- Forgetting remains substantial, but the strongest current evidence in this code branch is the tuned replay path rather than an isolation- or regularization-heavy alternative.

## 6.4 Preliminary BWT Improvement Sweep

We ran a preliminary follow-on sweep to test whether stronger replay settings or modest regularization terms improve backward transfer beyond the tuned-core result.

Completed exploratory runs include:
- EWC 1.0: CIL **23.03%**, BWT **−35.20%**
- replay ER, larger buffer: CIL **23.09%**, BWT **−35.23%**
- EWC 2.0: CIL **22.96%**, BWT **−35.37%**
- SI 0.1: CIL **23.06%**, BWT **−35.41%**
- weaker replay ER baseline: CIL **21.17%**, BWT **−40.84%**

Interpretation:
- These exploratory settings do not yet surpass the tuned-core mean BWT (**−30.39%**).
- The current evidence therefore favors replay-path design and tuned core configuration over naive strengthening of EWC, SI, or buffer size.
- This sweep is useful negative evidence: not every additional regularizer improves forgetting in the present architecture/runtime regime.

## 6.5 Consolidated Table

| Phase | Replay design | CIL | BWT | Task 0 retention |
|---|---|---:|---:|---:|
| 1 | Feature-only replay | 4.86% | −53.80% | 0.6% |
| 2 | Raw-input + reservoir sampling | 19.54% | −34.91% | 33.0% |
| 3 | Tuned core continual path (6-seed mean) | 23.96% | −30.39% | 25.1% |

## 6.6 Implementation-Alignment Notes

The reported phase contrast is aligned with current code capabilities:

- Reservoir sampling and replay modes are implemented in model/service code.
- Raw-input replay path is present in active training logic.
- Task-adapter isolation APIs are implemented, with sequential rollover available as an opt-in service path.
- The current tuned-core benchmark path is `scripts/run_local.py --tuned` rather than the older phase-only service benchmark narrative.

Therefore, this section claims a robust replay-centric continual result, not a solved replay+isolation combined benchmark result.

## 6.7 Limits of Current Evidence

1. Current quantitative claim is based on a narrow benchmark setting (Split-CIFAR-100 continual learning).
2. The tuned-core result has multiseed coverage, but confidence intervals and broader statistical treatment are still needed.
3. Preliminary BWT sweep results are diagnostic and not yet strong enough to replace the tuned-core result as the main reported configuration.
4. The active sensor/MIME routing ladder is not yet complete, so the present quantitative section should still be read as the continual-learning evidence core rather than the final perception-side evidence picture.
5. Real multimodal dataset ingestion (AudioSet/Ego4D-scale) is not yet an active training/eval path in this code branch.
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

## 7.6 What The Active Sensor Ladder Adds

The next decisive question is no longer "can we tweak BWT a little more with nearby regularizers?" It is:

- does typed sensor routing beat a merged generic stream?
- if so, does that advantage survive leave-one-out, binding-corruption, and noise stress tests?

This matters because it is the shortest path from architectural argument to measured perception-side evidence. The bounded Titan-local supervisor is useful here not as an optimization oracle, but as a way to execute that evidence ladder in a fixed, interpretable order with explicit stop states.

## 7.7 Next Experimental Priorities

1. Complete and aggregate the active `sensor_streaming` canonical-vs-merged routing family.
2. Interpret the follow-on sensor stress tests together: leave-one-out, corruption, and noise robustness.
3. Run a capacity-matched canonical-vs-merged control if the baseline gap is small enough that capacity confounds remain plausible.
4. Add confidence intervals and fuller statistical treatment to the tuned-core continual benchmark.
5. Benchmark sequential adapter rollover against the tuned-core result, not just against older phase baselines.
6. Add real multimodal ingestion/evaluation paths and report them separately from proxy datasets.
# Conclusion

This work establishes a code-aligned continual-learning result progression for Sensory Perceiver:

- Phase 1 feature-only replay fails with severe forgetting.
- Phase 2 raw-input replay with reservoir sampling materially improves retention.
- The current tuned-core continual path improves the result further to a 6-seed mean of CIL 23.96% and BWT −30.39%.

Beyond model behavior, the codebase now provides a production-style research runtime: typed contracts, explicit error semantics, service-layer training/evaluation/orchestration, and structured observability.

We also formalize modality protocol semantics through canonical `sensor_streaming` MIME input cases and runtime-contract validation for modality setup derivation, while model forward dispatch remains modality-key based. The next evidence-bearing stage is an active canonical-vs-merged sensor routing ladder with bounded follow-on stress tests rather than an open-ended search loop.

At the same time, two boundaries remain explicit:

1. Parameter-isolation APIs and opt-in service-path activation are implemented, but current exploratory EWC/SI-style follow-on sweeps have not yet surpassed the tuned-core replay baseline.
2. Real multimodal benchmark ingestion (beyond proxy datasets) remains future-stage work.

The immediate path forward is therefore clear: complete the sensor routing ladder and use it to either justify or narrow the broader perception claim, strengthen the tuned-core result with fuller statistical treatment, test whether structural isolation can beat that baseline rather than merely exist in code, and extend from proxy multimodal data to real large-scale multimodal datasets.
# Implementation Status Matrix (Code-Aligned)

This matrix maps paper claims to current implementation status in the codebase.

## Status Legend
- **Implemented**: Present in production path and testable from current code.
- **Partial**: Present as scaffolding/proxy, but not complete for paper-level claim.
- **Planned**: Not yet implemented in active runtime/training path.

## Architecture and Runtime

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| Package refactor (`models`, `runtime`, `services`, `observability`) | Implemented | `src/models/core.py`, `src/runtime/`, `src/services/`, `src/observability/` | Replaces monolithic structure. |
| Typed runtime contracts + error codes + schema validation | Implemented | `src/runtime/contracts.py`, `src/runtime/errors.py`, `src/runtime/schema.py` | Includes config/dataclass validation and typed error classification. |
| Service-layer training/evaluation/orchestration | Implemented | `src/services/training_service.py`, `src/services/evaluation_service.py`, `src/services/orchestration_service.py` | Active entrypoints call services. |
| Structured observability and metrics snapshots | Implemented | `src/observability/structured.py`, `src/observability/metrics.py` | JSON events + metric snapshots. |

## Continual Learning Mechanics

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| TaskAdapter module exists | Implemented | `src/models/core.py` (`class TaskAdapter`) | Includes `freeze()` method. |
| Adapter rollover per task (`freeze_task`) | Implemented (model API) | `src/models/core.py` (`freeze_task`) | Freezes current adapter and allocates next adapter. |
| Parameter-isolation telemetry (`get_trainable_parameters`, `get_task_coverage`) | Implemented | `src/models/core.py` | Available for reporting/analysis. |
| Parameter isolation active in service pipeline (opt-in) | Implemented | `src/services/training_service.py`, `src/train.py` | Enable with `--adapter_rollover --continual_task_count <N>`. |
| Reservoir sampling replay buffer | Implemented | `src/models/core.py` (`_update_replay_buffer`) | Integrated in model buffer updates. |
| Replay mode suite (`feature`, `er`, `contrastive`) | Implemented | `src/services/training_service.py`, `src/train.py`, `src/runtime/contracts.py` | Includes feature-only replay baseline and raw-input replay modes. |
| External classifier head (separate from backbone) | Implemented | `src/services/training_service.py` | Head is `nn.Linear` outside `SensoryPerceiver`. |

## MIME and Modality Routing

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| Canonical `sensor_streaming` experiment ID | Implemented | `src/runtime/mime_cases.py` (`SENSOR_STREAMING_EXPERIMENT_ID`) | Canonical ID is `sensor_streaming`. |
| Canonical `sensor_streaming` MIME input cases | Implemented | `src/runtime/mime_cases.py` (`SENSOR_STREAMING_INPUT_CASES`) | Cases for `smartwatch`, `room_env`, `robot_arm`. |
| Runtime modality setup derived from MIME registry | Implemented | `src/runtime/contracts.py` + `get_sensor_streaming_modality_setup()` | Single source of truth for dims/config. |
| MIME validation function for `sensor_streaming` bindings | Implemented | `validate_sensor_streaming_mime_bindings()` | Validates modality↔MIME pairs. |
| Runtime forward pass dispatches towers by MIME strings directly | **Partial** | `src/models/core.py` (`forward`) | Routing is by modality keys in input dict; MIME metadata is validated at runtime boundary, not parsed inside model forward. |

## Dataset/Benchmark Coverage

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| Sensor streaming dataset in dataloader factory | Implemented | `src/data/loaders.py` (`SensorStreamingDataset`, `create_dataloaders`) | Uses generated streams. |
| Audio-visual multimodal support in dataloader factory | Implemented (proxy) | `src/data/loaders.py` (`AudioVisualProxyDataset`) | Proxy dataset, not production ingestion. |
| Real AudioSet/Ego4D training pipeline integration | Planned | N/A in active runtime code | Not present in current loaders/services. |
| Additional IMU/text towers in active model config | Planned | N/A in `resolve_modality_setup` | Current multimodal setup includes vision + audio only. |

## Paper Alignment Notes

1. **Accurate to claim now**:
   - Phase-structured replay narrative and reservoir mechanism are reflected in model/service code.
   - Runtime modernization and typed contract system are present.

2. **Must be phrased carefully in paper**:
   - Parameter isolation is integrated as an **opt-in service path** (not always-on by default), and current exploratory Titan results do not yet show it outperforming the tuned replay-centric baseline.
   - MIME routing is currently a **canonical registry + boundary validation** plus modality-key dispatch, not raw MIME parsing inside `forward()`.
   - Multimodal real-benchmark ingestion (AudioSet/Ego4D) remains future/ongoing work.

3. **Canonical naming**:
   - Use `sensor_streaming` consistently (experiment ID, dataset ID, docs, scripts).
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
| Real multimodal benchmark ingestion (AudioSet/Ego4D scale) | Future work, not claimed as active benchmark path | Absent from active loaders/services; status already constrained in `05-experiments.md` and status matrix | Planned | Medium |

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
- **Multimodal claim**: “Proxy multimodal support is implemented; real large-scale multimodal ingestion (e.g., AudioSet/Ego4D) remains future-stage work.”

## 10.4 Immediate Paper Synchronization Actions

1. Keep `00-implementation-status-matrix.md` and this section synchronized whenever implementation status changes.
2. Avoid unsupported novelty claims unless backed by explicit comparative evidence in-scope.
3. Keep benchmark evidence statements scoped to datasets actually exercised in current code paths.
