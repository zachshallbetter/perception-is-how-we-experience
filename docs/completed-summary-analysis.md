# Completed Summary Analysis

This document summarizes the completed bounded sequence reports and packaged evidence that are already safe to interpret. It does **not** promote any sequence that is still running, blocked, or failed.

Current authority boundary:

- **Canonical paper evidence** lives under [paper/sections](file:///Users/zachshallbetter/Projects/Perception/paper/sections).
- **Packaged result artifacts** live under [paper/figures/generated/results_package](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package).
- **This file** is an operator-facing synthesis of completed summary files, not a replacement for the canonical Results section.

## Summary State

As of the current completed reports:

- strongest validated quantitative result remains the replay-centric continual-learning lane
- strongest completed perception-side result remains the synthetic sensor routing ladder
- first promotable canonical integrated real-audio sequence now exists
- first promotable canonical integrated real-sensor sequence now exists
- the current paper-grade experiment program is now complete enough to support the canonical manuscript without waiting on more routine training runs

## 1. Continual-Learning Core

Canonical packaged continual-learning evidence remains centered on the tuned 6-seed Titan result described in:

- [paper/sections/06-results.md](file:///Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md)
- [paper/figures/generated/results_package/real_data/real_data_status.json](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package/real_data/real_data_status.json)

Current paper-primary multiseed result:

- CIL mean: `23.96%`
- BWT mean: `-30.39%`
- FWT mean: `0.00%`

Interpretation:

- replay-path design remains the strongest supported mechanistic result
- no later regularizer-heavy sweep has displaced the tuned replay-centric path

## 2. Completed Synthetic Sensor Ladder

The completed synthetic sensor routing ladder remains the strongest perception-side completed evidence path. The final ladder status was:

- `canonical-dominant`

Key completed branch gaps from [paper/sections/06-results.md](file:///Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md):

- baseline: `+0.4583`
- capacity-matched baseline: `+0.3767`
- corrupt bindings: `+0.2717`
- drop smartwatch: `+0.3317`
- drop room environment: `+0.7950`
- drop robot arm: `+0.6333`
- noise smartwatch: `+0.0633`
- noise room environment: `+0.0617`
- noise robot arm: `-0.0933`

Interpretation:

- typed routing is strongly favorable overall on the hardened synthetic benchmark
- the result survives capacity matching and multiple stress conditions
- the advantage is not unconditional; `noise_robot_arm` is the notable counterexample

This result is completed and reportable, but still synthetic. It is not the same as broad real-domain proof.

## 3. Bootstrap Real-Data Evidence

### 3.1 Bootstrap Audio

Bootstrap audio evidence comes from:

- [results/real_audio_sequence_summaries/real-audio-seq-001-report.json](file:///Users/zachshallbetter/Projects/Perception/results/real_audio_sequence_summaries/real-audio-seq-001-report.json)

Completed result:

- `speech_commands`: mean accuracy `88.32%`

Interpretation:

- this established that the first real audio path worked end to end
- it is still bootstrap evidence, not the canonical integrated audio sequence

### 3.2 Bootstrap Sensor

Bootstrap sensor evidence comes from:

- [results/real_sensor_sequence_summaries/real-seq-004-report.json](file:///Users/zachshallbetter/Projects/Perception/results/real_sensor_sequence_summaries/real-seq-004-report.json)

Completed result:

- `wisdm`: canonical-minus-merged `+0.0419`

Interpretation:

- typed routing remained directionally favorable on the first real wearable/body-sensing path
- the gap is much smaller than on the synthetic ladder, which is exactly why the real-data path matters

## 4. Canonical Integrated Real-Audio Evidence

The first promotable integrated real-audio sequence is:

- [results/real_audio_sequence_summaries/real-audio-seq-003-report.json](file:///Users/zachshallbetter/Projects/Perception/results/real_audio_sequence_summaries/real-audio-seq-003-report.json)

This sequence is:

- `completed`
- unblocked
- artifact-complete

Completed datasets:

- `speech_commands`: `88.32%`
- `esc50`: `41.83%`
- `urbansound8k`: `60.57%`
- `fsd50k`: `9.74%`

Interpretation:

- this is the first **promotable canonical integrated audio sequence**
- it should supersede older partially blocked audio attempts for paper packaging
- the dataset spread is informative:
  - `speech_commands` is stable and strong
  - `esc50` and `urbansound8k` show the path extends beyond command recognition
  - `fsd50k` is currently weak and should be treated as a difficult regime, not as a success story

Operationally, this means the real-audio path is no longer only scaffold or bootstrap. It is now part of the canonical evidence package.

## 5. Canonical Integrated Real-Sensor Evidence

The completed integrated sensor sequence is:

- `real-sensor-seq-002`

This sequence is:

- `completed`
- unblocked
- artifact-complete

Completed datasets:

- `wisdm`: canonical-minus-merged `+0.0419`
- `pamap2`: canonical-minus-merged `+0.1820`
- `opportunity`: canonical-minus-merged `+0.0087`
- `uci_har`: canonical-minus-merged `-0.0033`
- `hhar`: canonical-minus-merged `-0.0006`
- `mhealth`: canonical-minus-merged `-0.0938`
- `selfback`: canonical-minus-merged `+0.0215`
- `geomagnetic_wlan`: canonical-minus-merged `+0.0000`
- `realworld_har`: canonical-minus-merged `+0.0238`

Interpretation:

- this is now the first **promotable canonical integrated sensor sequence**
- the integrated real-sensor line is materially beyond the original WISDM bootstrap
- canonical routing is positive on five of the nine integrated sensor datasets, tied on one, and negative on three; the overall story is therefore mixed but completed rather than pending

## 6. Promotion Rules

For clarity, the completed summary files should be interpreted with these gates:

- **Promotable now**
  - tuned continual-learning lane
  - completed synthetic sensor ladder
  - `real-audio-seq-003`
  - bootstrap `wisdm`
  - bootstrap `speech_commands`
- **Historical only**
  - earlier blocked or superseded audio attempts such as `real-audio-seq-002`
- **Promotable now**
  - `real-sensor-seq-002`

If a sequence is still `running`, `failed`, or `blocked`, it belongs in:

- sequence summary/status files
- generated package status
- operational docs if necessary

It does **not** belong in the canonical paper as completed evidence.

## 7. Practical Takeaways

What the completed summaries currently support:

- replay-path placement is still the strongest validated mechanistic finding
- typed routing has strong completed synthetic support
- real audio is now an integrated evidentiary lane rather than just a bootstrap path
- real sensor evidence is now a completed integrated evidentiary lane, but its empirical story is mixed rather than uniformly dominant

What they do **not** yet support:

- broad claims that real multimodal perception is fully validated across the integrated sensor lineup
- broad claims that the completed integrated sensor lineup is uniformly canonical-dominant across every dataset

## 8. Program Completion State

For the current canonical paper package, the core experiment program should now be treated as:

- **completed** for the continual-learning lane
- **completed** for the synthetic routing ladder
- **completed** for the canonical integrated audio lane
- **completed** for the canonical integrated sensor lane

This means further routine Titan runs are no longer required to justify the present manuscript. Additional runs should now be treated as one of:

- strategic strengthening
- targeted ablation
- replication
- future-domain expansion

They are not baseline completion work anymore.

## 9. Relevant Source Files

- [paper/sections/06-results.md](file:///Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md)
- [paper/sections/07-discussion.md](file:///Users/zachshallbetter/Projects/Perception/paper/sections/07-discussion.md)
- [paper/figures/generated/results_package/real_data/real_data_status.json](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package/real_data/real_data_status.json)
- [results/real_audio_sequence_summaries/real-audio-seq-001-report.json](file:///Users/zachshallbetter/Projects/Perception/results/real_audio_sequence_summaries/real-audio-seq-001-report.json)
- [results/real_audio_sequence_summaries/real-audio-seq-003-report.json](file:///Users/zachshallbetter/Projects/Perception/results/real_audio_sequence_summaries/real-audio-seq-003-report.json)
- [results/real_sensor_sequence_summaries/real-seq-004-report.json](file:///Users/zachshallbetter/Projects/Perception/results/real_sensor_sequence_summaries/real-seq-004-report.json)
