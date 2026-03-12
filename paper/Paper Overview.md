# Paper

This overview tracks the authoritative paper corpus under `paper/sections/` and the generated results package under `paper/figures/generated/results_package/`.

## Canonical Authority Model

Authoritative claim files:
- `paper/sections/00-implementation-status-matrix.md`
- `paper/sections/01-abstract.md`
- `paper/sections/02-introduction.md`
- `paper/sections/03-related-work.md`
- `paper/sections/04-methods.md`
- `paper/sections/05-experiments.md`
- `paper/sections/06-results.md`
- `paper/sections/07-discussion.md`
- `paper/sections/08-conclusion.md`
- `paper/sections/10-concept-implementation-audit.md`

`paper/sections/09-examples.md` is illustrative and non-empirical.

## Scope Baseline

- Empirical benchmark claims are bounded to currently reported results and active code paths.
- MIME semantics are framed as runtime-contract validation and modality setup derivation.
- Model forward dispatch is framed as modality-key based.
- Parameter isolation is framed as implemented model capability with opt-in service-path integration.
- The current paper-primary quantitative evidence is the 6-seed tuned-core Split-CIFAR-100 result.
- The current perception-side evidence path is the completed synthetic `sensor_streaming` canonical-vs-merged ladder plus bounded real-data follow-on sequences.

## Reader Map

If you want to understand the paper quickly, read:

1. `paper/sections/01-abstract.md`
2. `paper/sections/02-introduction.md`
3. `paper/sections/06-results.md`
4. `paper/sections/07-discussion.md`

If you want to understand what a user or product team can actually do with the results, pair the paper with:

- `README.md`
- `docs/productization-roadmap.md`

This is important because the current strongest evidence remains replay-centric continual learning, while the broader perception-side sensor/MIME claim is now supported by a completed synthetic routing ladder and only partially supported by bootstrap real-data evidence.

## Practical Reader Takeaway

The practical takeaway should be read at two levels:

1. **What the project currently proves**
   - replay placement matters for continual retention
   - typed routing has meaningful synthetic perception-side support
   - real wearable and audio evidence is now moving from scaffold to executable benchmark path

2. **What a future user would actually get if the architecture continues to validate**
   - a perception system that can combine multiple sensory channels,
   - stay coherent when one modality is noisy or missing,
   - and keep adapting to new environments or users without full retraining.

That is the right user-facing interpretation of the paper. The current value lies in the architecture and update discipline, not in claiming a finished universal perception product.

## What Currently Counts As Evidence

Current strong evidence:

- Phase 1 vs Phase 2 replay-path progression on Split-CIFAR-100
- tuned-core 6-seed Titan result
- negative evidence from the BWT follow-on sweep

Current completed perception-side evidence:

- `sensor_streaming` canonical-vs-merged routing
- leave-one-out, corruption, and noise stress tests
- capacity-matched routing control

Current active but not yet promotable evidence:

- `real-audio-seq-002`
- `real-sensor-seq-001`

## Monolith Policy

`paper/Perception Is How We Experience.md` is a generated monolith assembled from canonical section files.
Do not hand-edit the monolith as a source of truth.

## Current Paper Package

The paper-ready tables, plots, captions, and provenance manifest are generated under:

- `paper/figures/generated/results_package/`

These assets are the canonical figure/table bundle for the current results section and should stay synchronized with `paper/sections/06-results.md`.

## Supplementary Tracks

- Research drafts: `paper/supplementary/research-papers/`
- Patent drafts: `paper/patents/`

These tracks are non-canonical and must stay synchronized with implementation status constraints in `paper/sections/10-concept-implementation-audit.md`.
