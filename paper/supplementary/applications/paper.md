# Applications of Sensory Perceiver: Robotics, Medical, and Smart Homes

## Scope and Claim Boundaries

This supplementary paper defines domain transfer frameworks and validation pathways for three application tracks.
It does not supersede canonical benchmark claims.

Canonical benchmark authority remains:
- `paper/sections/05-experiments.md`
- `paper/sections/06-results.md`
- `paper/sections/00-implementation-status-matrix.md`

## Abstract

We provide a domain-oriented blueprint for evaluating Sensory Perceiver in robotics, medical monitoring, and smart-home automation settings. The emphasis is on protocol quality, risk containment, and evidence promotion criteria rather than canonical performance reporting.

## 1. Domain Transfer Framework

Each domain track should be specified through five layers:
1. **Operational objective** (what decision/action the system supports),
2. **Modality contract** (which modalities, shape assumptions, missingness behavior),
3. **Evaluation protocol** (metrics, baselines, stratification),
4. **Failure model** (expected breakdown patterns),
5. **Promotion gate** (conditions to move findings to canonical sections).

## 2. Robotics Track

### Objective

Evaluate multimodal control under asynchronous sensing and strict decision-latency constraints.

### Protocol Essentials

- Define control tasks and scenario stratification.
- Define latency budget and miss conditions.
- Include missing/noisy sensor stress tests.

### Risk Focus

- control instability under modality dropouts,
- synchronization drift,
- replay effects on policy retention.

## 3. Medical Track

### Objective

Evaluate multimodal monitoring/diagnostic assistance under missingness, calibration, and interpretability constraints.

### Protocol Essentials

- Define outcome tasks and risk stratification.
- Include calibration and false-alarm analysis.
- Document explanation and safety review process.

### Risk Focus

- overconfident errors,
- drift under incomplete modalities,
- non-actionable explanations.

## 4. Smart-Home Track

### Objective

Evaluate concurrent activity recognition under sparse and heterogeneous sensor streams with edge constraints.

### Protocol Essentials

- Define multi-label activity taxonomy.
- Include sensor sparsity and outage scenarios.
- Report latency/resource profile for deployment realism.

### Risk Focus

- collapse in sparse-signal periods,
- false positives in overlapping activities,
- edge-resource instability.

## 5. Cross-Domain Evaluation Governance

### Shared Requirements

1. aligned metric definitions where comparable,
2. explicit data partition and leakage controls,
3. reproducibility package for each domain run,
4. distinction between domain-specific and architecture-global conclusions.

### Evidence Labels

Use one of:
- `observed` (reproducible and bounded),
- `pilot` (partial controls),
- `planned` (design not yet executed).

## 6. Promotion Rules to Canonical Sections

A domain finding is promotable when:
1. the path runs through active runtime/service code,
2. evaluation protocol is fully specified and reproducible,
3. limitation analysis is present,
4. canonical sections are updated together (methods + experiments + results + status matrix).

## 7. Conclusion

This supplement provides implementation-aware domain blueprints and governance for future application studies while preserving strict canonical claim boundaries.
