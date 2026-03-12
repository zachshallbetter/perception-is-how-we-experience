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
