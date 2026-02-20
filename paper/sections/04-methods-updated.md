# Methods (Code-Aligned)

## 4.1 System Architecture

The implemented architecture consists of:

1. **Modality-specific PerceiverTowers** for local feature extraction.
2. **Shared cross-modal attention** over concatenated modality embeddings.
3. **Optional task adapters** for parameter isolation primitives.
4. **Replay buffer** with reservoir updates for continual learning.

Core model implementation: `src/models/core.py`.

## 4.2 MIME-Style Input Cases and Routing Semantics

The runtime formalizes sensor-streaming modality cases as a canonical registry:

- `smartwatch` → `application/vnd.sensory.smartwatch.timeseries+tensor` (7 features)
- `room_env` → `application/vnd.sensory.environment.timeseries+tensor` (4 features)
- `robot_arm` → `application/vnd.sensory.robot-arm.timeseries+tensor` (9 features)

Registry implementation: `src/runtime/mime_cases.py`.

Important implementation detail: the model forward path dispatches by **modality keys** in the input dictionary, while MIME validation and modality-shape canonicalization are enforced at runtime-contract boundaries.

## 4.3 Continual Learning Replay

### 4.3.1 Reservoir Sampling Buffer Management

For each sample index \(t\) with buffer capacity \(B\):
- if buffer not full, append;
- else replace a random slot with probability \(B/(t+1)\).

This ensures each observed sample has equal retention probability over time.

### 4.3.2 Replay Modes

The training service exposes two replay modes:

- `er`: cross-entropy on replayed raw samples.
- `contrastive`: NT-Xent-style feature alignment between current and replayed features.

Replay is computed from raw modality inputs, so gradients can traverse the backbone when replayed batches are processed.

## 4.4 Parameter Isolation: Implemented API vs Active Pipeline

Implemented model capabilities:
- `TaskAdapter.freeze()`
- `SensoryPerceiver.freeze_task()` (freeze current adapter + allocate next)
- `get_trainable_parameters()`, `get_task_coverage()`

Current status:
- APIs are implemented and test-validated.
- Full sequential activation of adapter rollover in the default service training loop is not yet wired as the primary path.

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

Real benchmark ingestion pipelines (e.g., AudioSet/Ego4D) are not yet in the active runtime path.
