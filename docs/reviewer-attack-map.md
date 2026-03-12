# Reviewer Attack Map

This document is intentionally blunt. It captures the most likely reviewer attacks against the current paper and how we should interpret them.

## Attack 1: "This is a continual-learning paper dressed up as a perception paper."

Why they would say it:

- The strongest quantitative evidence is still Split-CIFAR-100 continual learning.
- The broader perception language includes MIME semantics, sensor streams, and multimodal integration, but those have had less empirical weight than the CL runs.

Status:

- Fair criticism if we overclaim.
- Unfair if we explicitly present continual learning as the strongest current evidence path inside a broader perception architecture.

Response:

- Narrow the headline claim to what is proven.
- Make the sensor/MIME ablation a first-class experiment, not an appendix idea.

## Attack 2: "MIME routing is mostly a contract-layer concept, not a demonstrated model-level necessity."

Why they would say it:

- Runtime MIME registry and validation are implemented.
- Model forward dispatch still uses modality keys, not literal MIME strings.
- If we do not show a concrete canonical-vs-generic sensor result, the reviewer can dismiss the MIME story as governance rather than science.

Status:

- Strong attack if we keep the current state purely conceptual.
- Resolved materially only if we show a typed-modality vs merged-stream experiment.

Response:

- Run `sensor_streaming` with:
  - `modality_routing_mode=canonical`
  - `modality_routing_mode=merged`
- Compare accuracy/latency across matched seeds.

## Attack 3: "The paper’s claim surface is bigger than its evidence surface."

Why they would say it:

- The architecture story spans modality typing, shared integration, replay, service/runtime semantics, and future multimodal scaling.
- The measurements are strongest on one benchmark family.

Status:

- True in the current state if the paper is written too broadly.

Response:

- Separate:
  - validated mechanism claim,
  - implemented architecture claim,
  - future-scale agenda.
- Do not blur them into one undifferentiated “we solved perception” story.

## Attack 4: "The newer mechanism work is negative or inconclusive."

Why they would say it:

- EWC/SI/buffer sweeps did not beat the tuned replay-centric baseline.
- Distillation replay currently looks worse than the tuned ER controls.

Status:

- Factually correct.

Response:

- Treat these as diagnostic evidence, not failed hype.
- The result is still useful:
  - replay-path design is robust,
  - nearby anti-forgetting knobs are not automatically better,
  - new mechanism classes must clear a higher bar.

## Attack 5: "The project is interesting, but the broad thesis is still aspirational."

Why they would say it:

- The codebase is coherent and serious.
- The broad multimodal-perception thesis still needs one more clean experimental pillar.

Status:

- Also fair.

Response:

- The next decisive result should be the sensor/MIME ablation.
- If that lands cleanly, the broader architecture thesis becomes much easier to defend.

## Current Bottom Line

Today, the most defensible framing is:

- Sensory Perceiver is a machine-perception architecture with typed modality semantics and shared integration.
- Its strongest current empirical evidence is replay-centric continual learning.
- The next experiment that most changes the paper is the sensor/MIME canonical-vs-merged ablation.
