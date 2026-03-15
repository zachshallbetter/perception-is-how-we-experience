# Concept Verification Audit

This document verifies the major concept families in the repository against the current evidence base and identifies the shortest improvement path for each one.

Use it with:
- [Paper Overview.md](/Users/zachshallbetter/Projects/Perception/paper/Paper%20Overview.md)
- [00-implementation-status-matrix.md](/Users/zachshallbetter/Projects/Perception/paper/sections/00-implementation-status-matrix.md)
- [10-concept-implementation-audit.md](/Users/zachshallbetter/Projects/Perception/paper/sections/10-concept-implementation-audit.md)
- [reference/future-concepts/README.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/README.md)
- [paper/patents/README.md](/Users/zachshallbetter/Projects/Perception/paper/patents/README.md)

## Labels

- **Implemented**: present in the current repo code path
- **Evidenced**: backed by packaged results or completed reports
- **Conceptual**: architecture idea or explanatory lens not yet promoted as canonical fact
- **Patent candidate**: novelty-oriented idea tracked in the patent lane
- **Blocked / open problem**: not yet explained, validated, or mature enough to promote

## Verification Matrix

| Concept | Primary Lane | Current Truth Level | Verification Basis | Main Limitation | Best Improvement |
|---|---|---|---|---|---|
| Replay-through-backbone matters more than head-only replay | Canonical paper | Implemented + Evidenced | Split-CIFAR Phase 1 -> Phase 2 -> tuned-core results; replay branches in `src/services/training_service.py` | Evidence is strongest on continual learning, not every downstream domain | Keep as the central mechanistic claim; add tighter confidence framing, not broader scope |
| Typed routing beats premature merged routing in the synthetic ladder | Canonical paper | Implemented + Evidenced | completed `sensor_streaming` canonical-vs-merged ladder and packaged results | Synthetic support does not automatically generalize to every real dataset | Keep scoped as synthetic perception-side evidence |
| Completed integrated audio lane is canonical evidence | Canonical paper | Implemented + Evidenced | `real-audio-seq-003` completed unblocked and packaged | Audio lane is classification-focused, not yet a richer live multimodal deployment | Use as canonical integrated evidence, not as proof of every audio concept |
| Completed integrated sensor lane is canonical evidence | Canonical paper | Implemented + Evidenced | `real-sensor-seq-002` completed unblocked and packaged | Real-sensor gains are mixed, not uniformly positive | Use as canonical evidence with explicit positive/neutral/negative breakdown |
| Tower-first multimodal architecture | Canonical paper | Implemented + Partially evidenced | `src/models/core.py`, synthetic routing ladder, bounded integrated sequences | Current implementation uses one `PerceiverTower` family, not fully specialized tower classes | Keep the claim at semantic distinction first, not tower-class specialization |
| MIME-style runtime contracts as boundary truth | Canonical paper | Implemented | `src/runtime/mime_cases.py`, `src/runtime/contracts.py` | Forward pass still dispatches by modality key, not raw MIME string | Keep boundary-level claim explicit and avoid overstating in-model MIME routing |
| Task-adapter isolation as an active superiority claim | Canonical paper | Implemented but not evidenced as best result | model/service APIs plus preliminary sweeps | It is not the strongest benchmark result | Keep it opt-in and exploratory in paper language |
| Distinction-first hypothesis | Future concept | Conceptual with partial evidence | replay result, synthetic routing ladder, completed real-audio and real-sensor sequences, plus the bounded distinction-probe design | It is an explanatory hypothesis, not a closed law | Finish the positive/neutral/negative real-sensor explanation and keep falsification criteria explicit |
| Distinction predicts when typed routing helps | Future concept | Blocked / open problem | real-sensor buckets plus the eval-only distinction probes on `pamap2`, `mhealth`, `opportunity`, `uci_har`, and `hhar` | Structural explanation is not complete yet | Finish probe campaign and relate gains to heterogeneity, nuisance separation, and redundancy |
| Hierarchical modality grouping | Future concept / patent | Conceptual + Patent candidate | concept notes and patent note only | not in canonical runtime path | Promote only after implementation and bounded evidence |
| Dynamic modality composition | Future concept / patent | Conceptual + Patent candidate | concept notes and patent note only | not implemented in active runtime | Keep out of canonical paper until implemented and tested |
| Async multimodal synchronization | Future concept / patent | Conceptual + partially aligned scaffolding | live-system docs and patent note | no canonical empirical validation | Treat as live-system design guidance and patent-track material |
| Modality importance and runtime explanation | Future concept / patent | Conceptual + Patent candidate | concept notes and patent note only | no implemented runtime explanation surface in canonical path | Keep in concept/patent lanes |
| Perception-to-action loops | Future concept / patent | Conceptual + Patent candidate | concept notes and patent note only | no bounded canonical action-loop evidence | Keep non-canonical |
| Supplementary domain tracks in robotics, medical, smart home | Supplementary | Conceptual / planned | supplementary papers and application blueprints | they are protocols, not executed canonical evidence | Keep as promotion-ready blueprints, not benchmark evidence |

## Strong Concepts

These currently survive verification well:

- replay-through-backbone as the central continual-learning mechanism
- typed routing as a real perception-side design win in the synthetic ladder
- bounded sequence governance and promotion gates
- the claim-discipline split between canonical paper, docs, future concepts, and patents

## Concepts That Need Tighter Language

- tower specialization:
  - current code supports semantic tower separation with one shared tower family
  - do not imply every modality already has a fully different encoder class
- MIME routing:
  - current truth is contract-boundary validation and setup derivation
  - not raw MIME parsing inside `forward()`
- real-data multimodal strength:
  - completed integrated audio and sensor sequences now exist
  - but the sensor lane is mixed, so the theory should stay conditional

## Concepts That Need Explanation, Not More Branding

- distinction
- why `pamap2` is strongly positive
- why `mhealth` is negative
- why `opportunity`, `uci_har`, and `hhar` are near-zero

These do not mainly need more slogans. They need explanatory analysis.

## Improvement Priorities

1. Finish the bounded distinction probes and summarize them against the real-sensor buckets.
2. Tighten canonical language whenever a concept is implemented but not yet evidenced as dominant.
3. Keep future architecture and patent notes separate from present-tense empirical claims.
4. Use the integrated audio and sensor completions as evidence of bounded real-data viability, not as blanket validation for all future concepts.

## Bottom Line

The repository now has a clean core:

- a strong mechanistic continual-learning claim,
- a real synthetic perception-side routing result,
- and completed bounded integrated real-data lanes for audio and sensors.

The next verification work is no longer “does this run?” It is:

- which concepts are already true enough to defend,
- which are still explanatory hypotheses,
- and which are invention or future-use material that should remain outside the canonical claim surface.
