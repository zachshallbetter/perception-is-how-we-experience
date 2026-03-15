# Real-Sensor Distinction Probes

This document defines the bounded follow-up experiments for the sharpest open explanatory question in the current sensor evidence:

- why is `pamap2` strongly positive,
- why is `mhealth` negative,
- and why are `opportunity`, `uci_har`, and `hhar` near-zero?

## Purpose

The completed integrated real-sensor sequence gives a directional outcome per dataset. It does not, by itself, explain those outcomes.

These probes are designed to test whether the observed routing gaps are associated with:

- stronger dependence on a particular projected modality
- distributed signal across multiple projected modalities
- asymmetry under modality-specific noise
- different brittleness patterns between canonical and merged routing
- underlying non-overlapping information versus redundancy

## Core Hypothesis

The working hypothesis is:

- typed routing helps when the projected modalities carry meaningfully non-overlapping task-relevant or nuisance structure
- typed routing helps less, or can hurt, when the projected modalities are highly redundant, weakly informative, or poorly aligned with the imposed split

This is intentionally stronger than "different modality = separate."

Two signals from the same broad sensory family can still deserve separation if they provide different information:

- left/right auditory channels for localization
- left/right visual channels for depth

And two nominally different sensors can still behave as redundancy if they mostly measure the same phenomenon.

In practical terms:

- `pamap2` is the strongest positive case and should look like a genuinely heterogeneous dataset
- `mhealth` is the strongest negative case and should look more redundant or asymmetrically mismatched to the current split
- `opportunity`, `uci_har`, and `hhar` should clarify whether near-zero means weak structure, mixed structure, or practical indifference

## Design

This is an **eval-only** campaign.

It reuses the completed checkpoints from:

- `real-sensor-seq-002`

and evaluates them under existing `sensor_variant` stress conditions:

- `drop_smartwatch`
- `drop_room_env`
- `drop_robot_arm`
- `noise_smartwatch`
- `noise_room_env`
- `noise_robot_arm`

Target datasets:

- `pamap2`
- `mhealth`
- `opportunity`
- `uci_har`
- `hhar`

Target routings:

- `canonical`
- `merged`

Seeds:

- `42`
- `43`
- `44`

## Embedded Predictions

### Positive-case prediction: `pamap2`

If `pamap2` is positive because the current split preserves meaningful distinctions, then:

- dropping `smartwatch` or `robot_arm` should hurt accuracy materially
- canonical routing should either degrade less gracefully than merged under noise, or retain a positive canonical-minus-merged gap even when stressed
- the result should suggest that multiple projected modalities are carrying non-redundant signal
- the positive effect should look like complementary information, not just “more sensors”

### Negative-case prediction: `mhealth`

If `mhealth` is negative because the current split is redundant or mismatched, then:

- merged routing should remain competitive or superior under drop/noise probes
- canonical routing may degrade more than merged when a single modality is impaired
- the negative baseline gap may persist or widen under stress
- dropping any single projected stream may hurt both routings similarly if the dataset is effectively redundant sensor soup

### Near-zero prediction: `opportunity`, `uci_har`, `hhar`

If these datasets are genuinely near the boundary, then:

- some modality probes may be informative while others are not
- canonical-minus-merged may stay near zero, flip sign, or vary by modality
- the result should distinguish weakly helpful structure from true indifference

## Why Eval-Only

The scientific question is explanatory, not baseline-completion work.

Because the completed checkpoint families already exist, the cleanest next step is to:

- hold training fixed
- perturb modality availability or quality at evaluation
- compare how canonical and merged routing degrade

This avoids another large training campaign when the explanatory question can be answered with a cheaper bounded probe.

## Commands

Launch:

```bash
PYTHON_BIN=/ai/envs/perception/bin/python \
SOURCE_SEQUENCE_ID=real-sensor-seq-002 \
bash scripts/launch_real_sensor_distinction_probe_titan.sh
```

Summarize:

```bash
python3 scripts/summarize_real_sensor_distinction_probes.py \
  --campaign real-sensor-distinction-YYYYMMDD_HHMMSS \
  --source-sequence-id real-sensor-seq-002
```

## Reading The Outputs

The summary should be read in three layers:

1. **Per routing / variant accuracy**
   - how well the checkpoint performs under each modality stress
2. **Delta vs baseline**
   - how much each routing degrades relative to its completed baseline checkpoint
3. **Canonical minus merged under stress**
   - whether the routing gap grows, shrinks, or flips under specific modality failures

## Outcome Interpretation

### Strong support for the distinction hypothesis

This looks like:

- `pamap2` stays positive across the probe family
- its degradation pattern implies non-redundant signal across multiple projected modalities
- `mhealth` stays negative and looks more redundant or more brittle under the current split

That would upgrade the result from:

- typed routing sometimes helps

to:

- typed routing helps when inputs carry meaningfully non-overlapping information and can backfire when the split is redundant or mismatched

### Mixed support

This looks like:

- `pamap2` stays positive, but only for certain modality probes
- `mhealth` is negative, but only under a subset of perturbations
- the near-zero datasets split into weakly positive and weakly negative subcases

That would still be valuable. It would imply the distinction hypothesis is real but more local and modality-specific than the current wording suggests.

### Weak support or failure

This looks like:

- all datasets degrade similarly regardless of routing
- canonical-minus-merged does not track modality stress in an interpretable way
- positive, neutral, and negative datasets do not separate structurally

That would not invalidate the existing empirical results, but it would weaken the explanatory version of the distinction hypothesis and push the next step toward other predictors such as task complexity, class structure, or training dynamics.

## Intended Scientific Payoff

If a positive dataset stays positive and shows interpretable degradation when a projected modality is removed, that supports the idea that distinction is helping because the streams carry meaningfully different structure.

If a near-zero dataset remains near-zero across all variants, that suggests the current decomposition may be weak, redundant, or practically irrelevant for that task.

If a negative dataset like `mhealth` remains negative or gets worse under the same probes, it becomes a stronger counterexample to any overbroad distinction-first claim.

## Scientific Value

This is good follow-up work because it is:

- explanatory rather than just empirical
- bounded rather than exploratory sprawl
- mechanistically motivated
- cheap relative to new training campaigns
- falsifiable in clear ways

If it succeeds, the distinction story becomes sharper and more defensible.
If it fails, the project still learns something real: the mixed real-sensor outcome is not explained by simple modality heterogeneity alone.
