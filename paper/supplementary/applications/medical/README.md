# Application Domain: Medical Monitoring and Diagnosis

## Scope

This document defines the medical-domain evaluation blueprint for multimodal monitoring/diagnostic support studies.
It is not a canonical source for main-paper empirical claims.

## Problem Framing

Primary objective: reliable multimodal inference under missingness, calibration requirements, and interpretability constraints.

Key modality classes:
- physiological time-series,
- imaging modalities,
- structured clinical signals,
- text/context channels where available.

## Evaluation Blueprint

### Core Experiments

1. Multimodal baseline with complete modality availability.
2. Missing-modality stress tests.
3. Calibration and threshold sensitivity analyses.
4. Interpretability and error-review protocol runs.

### Metrics

- discriminative performance,
- calibration reliability,
- false-alarm burden,
- robustness to incomplete modality sets.

## Risk Register

- overconfident incorrect predictions,
- poor behavior under sparse or delayed modalities,
- explanations that are not clinically actionable.

## Promotion Criteria

A medical finding can be promoted to canonical sections only if:
1. the run path is reproducible in active runtime/service flow,
2. evaluation protocol and safety constraints are explicit,
3. evidence and limitation boundaries are documented together.
