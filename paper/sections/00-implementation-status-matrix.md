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
| Real-data bootstrap evidence on Speech Commands and WISDM | Implemented | `results/real_audio_sequence_summaries/real-audio-seq-001-report.json`, `results/real_sensor_sequence_summaries/real-seq-004-report.json` | Bootstrap evidence only; not the canonical integrated package. |
| Canonical integrated real-data sequences | Partial / Pending | `real-audio-seq-002`, `real-sensor-seq-001` supervisor reports | Promoted only after bounded sequence completion. |
| AudioSet media-backed training pipeline integration | Blocked | Metadata staged locally; no local audio media in active runtime path | Preserve blocked state explicitly. |
| UbiqLog supervised benchmark integration | Blocked | Raw logs present, but no canonical supervised task definition | Preserve blocked state explicitly. |
| Additional IMU/text towers in active model config | Planned | N/A in `resolve_modality_setup` | Current active multimodal setup is still bounded and dataset-specific. |

## Paper Alignment Notes

1. **Accurate to claim now**:
   - Phase-structured replay narrative and reservoir mechanism are reflected in model/service code.
   - Runtime modernization and typed contract system are present.

2. **Must be phrased carefully in paper**:
   - Parameter isolation is integrated as an **opt-in service path** (not always-on by default), and current exploratory Titan results do not yet show it outperforming the tuned replay-centric baseline.
   - MIME routing is currently a **canonical registry + boundary validation** plus modality-key dispatch, not raw MIME parsing inside `forward()`.
   - Real-data evidence is now bootstrap-plus-canonical staged; blocked datasets remain explicit rather than being described vaguely as future work.

3. **Canonical naming**:
   - Use `sensor_streaming` consistently (experiment ID, dataset ID, docs, scripts).
