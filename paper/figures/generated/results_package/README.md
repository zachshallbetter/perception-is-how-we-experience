# Paper Results Package

This package is generated from the validated evidence used by the canonical paper sections.

## Sources

- Phase 1 uses the canonical legacy metrics from `paper/sections/06-results.md` because the original feature-only run directory is not currently tracked as a local results artifact.
- Phase 2 uses `results/cl_baseline_reservoir_sampling/results.json`.
- Phase 3 uses the discovered tuned-core family `20260311_140818` with 6 synchronized Titan runs.
- The BWT sweep uses the completed `results/bwt_*/results.json` runs synchronized from Titan.
- The completed synthetic sensor routing ladder is reported from canonical section text and synchronized sequence summaries.
- Real-data evidence is split into bootstrap (`real-audio-seq-001`, `real-seq-004`) and canonical integrated (`real-audio-seq-003`, `real-sensor-seq-002`) stages.
- Canonical promotion requires a sequence to complete with all expected datasets present and no blocked entries. `real-audio-seq-003` is currently promotable, and `real-sensor-seq-002` is currently not promotable.
- The active integrated lineups now extend to `fsd50k` and `realworld_har`; `audioset` and `ubiqlog` remain deferred legacy sources outside the bounded chain.

## Included artifacts

- `phase_progression.csv`
- `tuned_core_multiseed.csv`
- `bwt_sweep_ranked.csv`
- `table_results_overview.{png,pdf,svg}`
- `figure_phase_progression.{png,pdf,svg}`
- `figure_tuned_core_multiseed.{png,pdf,svg}`
- `figure_bwt_sweep_ranked.{png,pdf,svg}`
- `figure_real_data_evidence.{png,pdf,svg}`
- `package_manifest.json`
- `real_data/real_data_status.json`
- `real_data/real_audio_evidence.csv` (when local reports exist)
- `real_data/real_sensor_evidence.csv` (when local reports exist)
- `source_matrices/*.png`

## Build

```bash
python3 paper/figures/build_results_package.py
```
