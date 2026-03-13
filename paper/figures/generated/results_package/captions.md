# Figure And Table Captions

## Table 1
Split-CIFAR-100 continual-learning progression. Phase 1 is a legacy feature-only replay result sourced from the canonical paper section because the original run directory is not currently tracked as a local artifact. Phase 2 and Phase 3 are artifact-backed, with Phase 3 reporting the mean over 6 tuned-core Titan runs from family `20260311_140818`.

## Figure 1
CIL and BWT progression from feature-only replay to raw-input replay with reservoir sampling and then the tuned-core continual configuration. The figure emphasizes that replay-path design, not just stronger regularization, drove the major retention gains.

## Figure 2
Tuned-core multiseed stability on Titan Node. Each point is one full 20-task Split-CIFAR-100 run, with dashed lines indicating the multiseed means. Variance is modest relative to the mean effect size.

## Figure 3
Exploratory BWT sweep ranked by final BWT. The dashed reference line marks the tuned-core multiseed mean. None of the completed EWC, SI, larger-buffer ER, or contrastive replay runs surpass the tuned-core baseline.

## Figure 4
Real-data evidence overview. Bootstrap evidence comes from `real-audio-seq-001` on Speech Commands and `real-seq-004` on WISDM. Canonical evidence is promoted only when bounded integrated sequences finish with all expected datasets present and no blocked entries. `real-audio-seq-003` is the current canonical audio sequence, while `real-sensor-seq-002` remains pending promotion until it completes cleanly.

## Source Matrices
The copied accuracy-matrix images provide the underlying per-task retention structure for the reported Phase 2 raw-input replay run and the three tuned-core Titan runs.
