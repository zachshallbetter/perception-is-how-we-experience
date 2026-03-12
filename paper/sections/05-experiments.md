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

The practical reason for this ladder is straightforward: if typed routing only wins on an easy baseline and collapses under capacity matching or stress, it should not be used as the basis for stronger product or paper claims. The bounded supervisor is therefore part of the evidence discipline, not just an automation convenience.

### Real-data progression path

In parallel with the hardened synthetic ladder, the repository now has an explicit real-data progression path through the same `sensor_streaming` runtime:

- integrated real-sensor datasets:
  - `wisdm`
  - `pamap2`
  - `opportunity`
  - `uci_har`
  - `hhar`
  - `mhealth`
  - `selfback`
  - `geomagnetic_wlan`
- integrated real-audio datasets:
  - `speech_commands`
  - `esc50`
  - `urbansound8k`
- explicitly blocked:
  - `ubiqlog`, because the repository does not yet define a canonical supervised task target for those raw logs
  - `audioset`, because only metadata CSVs are staged locally and no audio media has been acquired into the prepared-audio path

This path is intentionally conservative. Real-data routing experiments are only launched once prepared manifests and canonical `train.pt` / `val.pt` splits exist. A bounded Titan-local real-data supervisor is now available to:

- auto-prepare integrated datasets if raw files are present,
- launch canonical-vs-merged routing comparisons for prepared datasets,
- and record blocked datasets explicitly rather than silently treating them as runnable evidence.

Audio is now part of the runnable real-data path, but only in a narrow first form. The repository now defines:

- one prepared audio contract under `data/real-audio/`,
- working parsers for `speech_commands`, `esc50`, and `urbansound8k`,
- and one bounded Titan-local audio supervisor for launching prepared audio runs.

These real-data integrations are not yet part of the main paper-primary results. They exist to extend the empirical coverage beyond synthetic routing and to make the next evidence steps executable on Titan without inventing a second runtime or supervision stack.

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
