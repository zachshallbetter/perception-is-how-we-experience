# Paper Results Package

This package is generated from validated Split-CIFAR-100 results used by the canonical paper sections.

## Sources

- Phase 1 uses the canonical legacy metrics from `paper/sections/06-results.md` because the original feature-only run directory is not currently tracked as a local results artifact.
- Phase 2 uses `results/cl_baseline_reservoir_sampling/results.json`.
- Phase 3 uses the 3 tuned-core Titan runs under `results/paper_runlocal_tuned_20t_seed*_20260310_091001_recover/`.
- The BWT sweep uses the completed `results/bwt_*/results.json` runs synchronized from Titan.

## Included artifacts

- `phase_progression.csv`
- `tuned_core_multiseed.csv`
- `bwt_sweep_ranked.csv`
- `table_results_overview.{png,pdf,svg}`
- `figure_phase_progression.{png,pdf,svg}`
- `figure_tuned_core_multiseed.{png,pdf,svg}`
- `figure_bwt_sweep_ranked.{png,pdf,svg}`
- `package_manifest.json`
- `source_matrices/*.png`

## Build

```bash
python3 paper/figures/build_results_package.py
```
