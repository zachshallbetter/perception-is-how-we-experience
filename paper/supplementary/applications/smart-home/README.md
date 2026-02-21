# Application Domain: Smart Home Automation

## Scope

This document defines the smart-home evaluation blueprint for multimodal activity-recognition and automation-assist studies.
It is not a canonical source for main-paper empirical claims.

## Problem Framing

Primary objective: reliable concurrent activity recognition under sparse, heterogeneous sensor streams with edge-resource constraints.

Key modality classes:
- ambient audio,
- motion and occupancy signals,
- environmental/contextual telemetry,
- temporal context features.

## Evaluation Blueprint

### Core Experiments

1. Multi-activity baseline across full sensor configuration.
2. Sensor sparsity and outage robustness runs.
3. Concurrent-activity disambiguation runs.
4. Edge-latency/resource sensitivity runs.

### Metrics

- multi-label recognition quality,
- robustness to sensor sparsity,
- false-activation burden,
- edge latency and resource profile.

## Risk Register

- confusion under overlapping activities,
- degraded accuracy in sparse-signal windows,
- edge runtime instability.

## Promotion Criteria

A smart-home finding can be promoted to canonical sections only if:
1. reproducible active runtime path exists,
2. protocol and metric definitions are explicit,
3. resource and limitation constraints are documented with the finding.
