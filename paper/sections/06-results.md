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

At present, the packaged quantitative story is still centered on the continual-learning lane. The completed `sensor_streaming` routing ladder is the current perception-side evidence path. It now includes the baseline, capacity-matched control, leave-one-out branches, corruption branch, and all three single-modality noise branches.

For an applied reader, the present Results section should be read as establishing one validated operational rule: if a perception system must keep learning over time, replay design at the shared backbone matters more than simply adding lightweight regularizers around the edges.

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
4. The synthetic sensor/MIME routing ladder is complete and reportable, but it remains controlled synthetic evidence rather than the final real-domain perception claim.
5. Real-data evidence is now split into bootstrap and canonical integrated stages. Bootstrap WISDM and Speech Commands runs are includable; the bounded integrated sequences remain pending until their reports complete successfully.

## 6.8 Completed Sensor/MIME Routing Summary

The completed `sensor_streaming` ladder now provides the current perception-side result on the hardened synthetic sensor benchmark. We compare:

- `canonical`: typed modality routing (`smartwatch`, `room_env`, `robot_arm`) with modality-specific towers
- `merged`: one generic sensor stream without explicit typed modality separation

Completed paired gaps so far are:

| Variant | Canonical mean acc. | Merged mean acc. | Canonical - Merged |
|---|---:|---:|---:|
| baseline | 1.0000 | 0.5417 | +0.4583 |
| capacity-matched baseline | 1.0000 | 0.6233 | +0.3767 |
| drop smartwatch | 1.0000 | 0.6683 | +0.3317 |
| drop room environment | 1.0000 | 0.2050 | +0.7950 |
| drop robot arm | 0.8967 | 0.2633 | +0.6333 |
| corrupt bindings | 1.0000 | 0.7283 | +0.2717 |
| noise smartwatch | 1.0000 | 0.9367 | +0.0633 |
| noise room environment | 1.0000 | 0.9383 | +0.0617 |
| noise robot arm | 0.8817 | 0.9750 | -0.0933 |

Interpretation:

- Typed canonical routing is ahead on the baseline, capacity-matched, leave-one-out, corruption, and two of the three noise branches.
- The capacity-matched control remains clearly positive (**+0.3767**), which reduces the risk that the baseline result is only a front-end capacity artifact.
- Leave-one-out and corruption conditions still favor canonical routing strongly, which suggests the result is not limited to the easiest baseline comparison.
- The smallest margins occur under single-modality noise, where merged routing narrows the gap substantially.
- `noise_robot_arm` is the only completed branch where merged routing wins (**-0.0933**), so the full ladder should be described as strongly favorable to canonical routing overall, but not uniformly dominant under every stress condition.

Overall supervisor state for this ladder was **`canonical-dominant`**. This should still be treated as **completed synthetic evidence** rather than a final broad perception claim. However, it is now the strongest direct support so far that typed routing matters for more than runtime hygiene on the current controlled sensor task.

## 6.9 Real-Data Evidence Packaging

The real-data lane is now packaged with an explicit bootstrap-versus-canonical split so the paper does not overread early runs:

- **Bootstrap audio evidence**: `real-audio-seq-001` on `speech_commands`
- **Canonical audio evidence**: `real-audio-seq-002` on `speech_commands`, `esc50`, and `urbansound8k`
- **Bootstrap sensor evidence**: `real-seq-004` on `wisdm`
- **Canonical sensor evidence**: `real-sensor-seq-001` on the integrated real-sensor lineup

Completed bootstrap real-data results are:

- `real-audio-seq-001` (`speech_commands`): mean accuracy **88.32%**
- `real-seq-004` (`wisdm`): canonical-minus-merged routing gap **+4.19 pp**

Interpretation:

- These bootstrap runs establish that the first real audio and real sensor paths work end to end on Titan.
- They are useful evidence, but they are not the canonical integrated comparison set for the next paper update.
- The canonical real-data evidence package is promoted only from the bounded integrated sequences (`real-audio-seq-002` and `real-sensor-seq-001`) once their sequence reports complete successfully.

Operationally, the generated package under `paper/figures/generated/results_package/real_data/` is the source of truth for current real-data status. It separates:

- bootstrap evidence
- canonical integrated evidence
- deferred legacy sources outside the active bounded chain

This keeps historical bootstrap reports from being mistaken for the current integrated claim surface.
