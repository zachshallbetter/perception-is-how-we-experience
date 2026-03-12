# Reviewer Rebuttal Plan

This document translates the likely reviewer attacks into concise rebuttal positions and concrete supporting evidence.

## 1. "This is just a continual-learning paper."

Rebuttal:

The current strongest quantitative evidence is indeed replay-centric continual learning, and the paper now says so directly. However, the system itself is not a single-stream CL baseline. It combines:

- explicit typed modality contracts,
- modality-specific towers,
- shared cross-modal integration,
- and temporal adaptation mechanisms.

The correct interpretation is that continual learning is the strongest validated evidence path inside a broader perception architecture, not that the broader architecture is absent.

Evidence:

- `src/runtime/mime_cases.py`
- `src/runtime/contracts.py`
- `src/models/core.py`
- `paper/sections/02-introduction.md`
- `paper/sections/10-concept-implementation-audit.md`

## 2. "MIME routing is not really demonstrated."

Rebuttal:

We agree that the previous version risked making the MIME/typed-modality story more conceptual than measured. The repository now includes a direct sensor-streaming ablation path:

- canonical typed modality routing via the sensor MIME registry
- merged generic-stream routing as the comparison condition

This is the right experiment to evaluate whether explicit typed modality structure matters empirically on the implemented sensor benchmark path.

Evidence path:

- `src/data/loaders.py`
- `src/runtime/contracts.py`
- `scripts/launch_sensor_mime_ablation_titan.sh`
- `docs/sensor-mime-experiment-plan.md`

## 3. "Your newer anti-forgetting mechanisms did not improve the headline result."

Rebuttal:

Correct, and that is explicitly reflected in the paper. The point of the follow-on sweeps is not that every regularizer helps, but that the validated gain currently comes from replay-path design. The exploratory sweeps serve as negative evidence against simplistic “just add EWC/SI” narratives in this architecture.

Evidence:

- `paper/sections/06-results.md`
- `paper/sections/07-discussion.md`
- generated package under `paper/figures/generated/results_package/`

## 4. "The broad thesis is still ahead of the evidence."

Rebuttal:

The paper now distinguishes:

- validated current result,
- implemented architecture,
- future-scale agenda.

That separation is deliberate. We are not claiming the full multimodal perception agenda is already fully benchmarked; we are claiming the system architecture exists, the runtime semantics are explicit, and the replay-centric retention result is the strongest current evidence.

## 5. What materially strengthens the paper next

The highest-value next result is:

- a clean canonical-vs-merged `sensor_streaming` ablation with matched seeds and reported accuracy/latency.

If canonical typed routing wins, the broader paper identity becomes substantially more defensible.
