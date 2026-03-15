# Strategic Next Questions

This document captures what is worth doing **after** the canonical experiment program has completed.

It is intentionally not a backlog of routine benchmark completion work. The completed paper-grade lanes already exist:

- replay-centric continual learning
- synthetic sensor routing ladder
- canonical integrated real-audio sequence
- canonical integrated real-sensor sequence

## What Counts As Worthwhile Next Work

A next experiment should now answer a sharper question than "does the pipeline run?"

Good next targets:

- **theory strengthening**
  - test whether the distinction pattern holds under a more adversarial multimodal setting
- **replication**
  - rerun the strongest canonical results under a second environment or alternate seed family
- **boundary finding**
  - identify where typed routing stops helping and why
- **domain expansion**
  - move one future-concept domain into implemented + evidenced status
- **productization relevance**
  - validate one live-system operational claim with bounded evidence

## Highest-Value Next Questions

1. **When does distinction stop helping?**
   - The integrated sensor sequence is mixed, not uniformly dominant.
   - The next high-value experiment is not another broad sweep, but the targeted distinction-probe analysis of the negative or neutral datasets:
     - `uci_har`
     - `hhar`
     - `mhealth`
     - `geomagnetic_wlan`

2. **What exact properties predict typed-routing gains?**
   - Candidate factors:
     - non-overlapping information
     - modality heterogeneity
     - nuisance overlap
     - temporal granularity mismatch
     - class structure complexity
     - synthetic vs real-domain entropy
   - Practical rule:
     - prioritize domains where modality families arise from different physical mechanisms and operate on different nuisance or temporal regimes

3. **Can one future-use domain be promoted cleanly?**
   - The strongest current candidates, ranked by expected scientific value, are:
     - audio + video recognition
     - healthcare waveforms + physiological context
     - robotics with state + environment sensing
   - Reason:
     - these domains stress clearly different modality families rather than weakly separated variants of the same signal
     - they are the best test of whether typed routing really benefits heterogeneous modalities
   - The requirement is unchanged: implemented plus evidenced.

4. **How stable is the current strongest result?**
   - Replication under a second seed family or alternate hardware would strengthen the paper more than random exploratory expansion.

## What Should Not Drive The Next Cycle

- routine dataset-integration cleanup for the current paper
- broad undirected hyperparameter search
- adding more conceptual examples to the canonical paper
- expanding the claim surface before a new evidence gate clears

## Working Rule

From this point forward:

- routine experiment completion is done
- new runs must justify themselves as theory, replication, boundary, or domain-promotion work

That is the cleanest way to keep the project from turning completed evidence into an open-ended benchmark treadmill.

## Preferred Domain-Expansion Order

If the next cycle includes domain promotion, the cleanest order is:

1. **Audio + video**
   - fastest path to a clearly heterogeneous real multimodal case
   - strongest test of visual versus auditory distinction
2. **Healthcare waveforms + slower physiological/context streams**
   - strong test of multi-timescale physiological distinction
   - high application credibility
3. **Robotics**
   - natural fit for the architecture
   - higher integration complexity, but very strong if it lands cleanly

Domains that are still useful but not first-choice should stay secondary until one of the above is implemented and evidenced.
