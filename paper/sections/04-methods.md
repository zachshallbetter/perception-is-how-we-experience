# Methods (Code-Aligned)

## 4.1 System Architecture

The architectural position in the literature is:
- [Transformer](https://arxiv.org/abs/1706.03762): attention as the core sequence primitive.
- [Perceiver](https://arxiv.org/abs/2103.03206): latent-bottleneck perception over arbitrary input arrays.
- [Perceiver IO](https://arxiv.org/abs/2107.14795): structured query-based outputs over that latent core.
- Sensory Perceiver: typed routing, synchronization, bounded adaptation, and replay-backed operation layered on top of that lineage.

The implemented architecture consists of:

1. **Modality-specific PerceiverTowers** for local feature extraction.
2. **Shared cross-modal attention** over concatenated modality embeddings.
3. **Optional task adapters** for parameter isolation primitives.
4. **Replay buffer** with reservoir updates for continual learning.

Towers should be read as **modality-specific encoders for related but semantically distinct sensory views**. The point of the split is not to claim the modalities are unrelated. It is the opposite: they are often different observations of the same world-state, user state, environment, or task context, but they should not be flattened together before the model has a chance to preserve their local structure and learn how they interact.

Typical tower groupings could include:

- camera + microphone + pose
- wearable + room sensors + medical device telemetry
- robot state + environment sensors + operator signals

The general architectural claim is therefore:

- each tower should honor the semantics of its own modality,
- the shared core should learn cross-modal relationships,
- and replay should re-enter through the towers so those semantics do not drift during continual learning.

Broader domain-specific tower examples and future live-system specializations are tracked outside the canonical paper lane under [reference/future-concepts/README.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/README.md). They should be read as **Conceptual** unless and until they are implemented and evidenced in this repository.

Important implementation boundary:
- in the current codebase, these towers share one `PerceiverTower` family with modality-specific configuration
- the paper's claim is therefore **semantic distinction first**, not that each modality already uses a completely different tower class

Conceptually, a more mature live deployment could specialize tower internals further:
- vision streams may want stronger spatial-temporal bias
- audio streams may want stronger time-frequency bias
- wearable or telemetry streams may want stronger temporal or structured-state bias

That specialization is part of the architectural direction, but it is not the current implemented claim.

### 4.1.1 Current Synthetic Benchmark Instance

The repository's current synthetic perception-side benchmark, `sensor_streaming`, instantiates the general tower pattern with three canonical modalities:

- **smartwatch**: a 7-feature wearable stream representing body-side signals such as heart-rate and motion-derived measurements
- **room_env**: a 4-feature ambient stream representing environmental state such as temperature, humidity, CO\textsubscript{2}, and light
- **robot_arm**: a 9-feature actuation/kinematics stream representing joint and force/torque state

Those three modalities are one benchmark instance, not the universal definition of a Sensory Perceiver deployment.

Core model implementation: `src/models/core.py`.

Source files:
- [core.py](/Users/zachshallbetter/Projects/Perception/src/models/core.py)
- [mime_cases.py](/Users/zachshallbetter/Projects/Perception/src/runtime/mime_cases.py)
- [contracts.py](/Users/zachshallbetter/Projects/Perception/src/runtime/contracts.py)
- [training_service.py](/Users/zachshallbetter/Projects/Perception/src/services/training_service.py)

```mermaid
flowchart LR
    T["Transformer<br/>tokens → encoder/decoder → token prediction"] --> P["Perceiver / Perceiver IO<br/>arbitrary array → latent bottleneck → query output"]
    P --> S["Sensory Perceiver<br/>typed envelopes → sync → towers + shared core → user outputs + adaptation gate"]
```

```mermaid
flowchart LR
    A["Modality A<br/>camera / wearable / device"] --> B["PerceiverTower<br/>modality-specific projection + MLP + self-attention"]
    C["Modality B<br/>audio / ambient / room"] --> D["PerceiverTower<br/>modality-specific projection + MLP + self-attention"]
    E["Modality C<br/>pose / telemetry / actuation"] --> F["PerceiverTower<br/>modality-specific projection + MLP + self-attention"]
    B --> G["Masked shared cross-modal attention"]
    D --> G
    F --> G
    G --> H["Task adapters (optional)"]
    H --> I["Classifier / output head"]
    J["Replay buffer<br/>raw modality inputs"] -.-> G
```

```mermaid
flowchart LR
    A["Speech Commands<br/>https://huggingface.co/datasets/google/speech_commands"] --> G["Bootstrap / canonical evidence gate"]
    B["WISDM<br/>https://archive.ics.uci.edu/dataset/507/wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset"] --> G
    C["ESC-50<br/>https://github.com/karolpiczak/ESC-50"] --> G
    D["UrbanSound8K<br/>https://urbansounddataset.weebly.com/urbansound8k.html"] --> G
    E["FSD50K<br/>https://zenodo.org/records/4060432"] --> G
    F["RealWorld HAR<br/>https://huggingface.co/datasets/sha-ce/SensorData"] --> G
    G --> H["Bootstrap evidence"]
    G --> I["Canonical integrated evidence"]
```

## 4.2 MIME-Style Input Cases and Routing Semantics

The runtime formalizes `sensor_streaming` modality cases as a canonical registry:

- `smartwatch` → `application/vnd.sensory.smartwatch.timeseries+tensor` (7 features)
- `room_env` → `application/vnd.sensory.environment.timeseries+tensor` (4 features)
- `robot_arm` → `application/vnd.sensory.robot-arm.timeseries+tensor` (9 features)

Registry implementation: `src/runtime/mime_cases.py`.

Important implementation detail: MIME semantics are enforced at runtime-contract boundaries for modality validation and setup derivation, while the model forward path dispatches by **modality keys** in the input dictionary.

This means the canonical MIME-style registry is the source of truth for:

- what each modality is,
- how wide its feature tensor is,
- and which tower it should route into.

It is not a claim that MIME strings themselves are parsed deep inside the model. The registry constrains the boundary; the forward path uses the canonical modality keys that boundary resolution produced.

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

Real benchmark ingestion is now bounded explicitly. Bootstrap real-data paths exist for WISDM and Speech Commands, while integrated canonical sequences remain pending. The active bounded lineups now extend beyond those bootstrap runs to include HF-backed replacements `fsd50k` and `realworld_har`, while the older `audioset` and `ubiqlog` options remain deferred outside the active chain.

External benchmark sources are tracked in [real-sensor-datasets.md](/Users/zachshallbetter/Projects/Perception/docs/real-sensor-datasets.md), including:
- [WISDM](https://archive.ics.uci.edu/dataset/507/wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset)
- [Speech Commands](https://huggingface.co/datasets/google/speech_commands)
- [ESC-50](https://github.com/karolpiczak/ESC-50)
- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- [FSD50K](https://zenodo.org/records/4060432)
- [RealWorld HAR via sha-ce/SensorData](https://huggingface.co/datasets/sha-ce/SensorData)
