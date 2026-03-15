# Compiled Results

This document concatenates the current completed result lanes into one operator-facing summary.

It is a convenience artifact, not a new authority source. Canonical claim files remain:

- [paper/sections/06-results.md](/Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md)
- [paper/figures/generated/results_package](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package)
- [docs/completed-summary-analysis.md](/Users/zachshallbetter/Projects/Perception/docs/completed-summary-analysis.md)

## Summary

The current paper-grade experiment program is complete across four lanes:

- replay-centric continual learning
- completed synthetic sensor routing ladder
- canonical integrated real-audio sequence
- canonical integrated real-sensor sequence

## 1. Continual-Learning Results

### Phase Progression

| Phase | Replay design | CIL | BWT | Task 0 retention |
|---|---|---:|---:|---:|
| 1 | Feature-only replay | 4.86% | -53.80% | 0.6% |
| 2 | Raw-input + reservoir sampling | 19.54% | -34.91% | 33.0% |
| 3 | Tuned core continual path (6-seed mean) | 23.96% | -30.39% | 25.1% |

### Tuned-Core Multiseed

| Seed | CIL | BWT |
|---|---:|---:|
| 47 | 23.60% | -29.44% |
| 48 | 22.98% | -32.39% |
| 49 | 23.95% | -30.54% |
| 50 | 24.16% | -31.15% |
| 51 | 24.32% | -30.31% |
| 52 | 24.74% | -28.49% |

### Exploratory BWT Sweep

| Variant | CIL | BWT |
|---|---:|---:|
| tuned core multiseed mean | 23.96% | -30.39% |
| EWC 1.0 | 23.03% | -35.20% |
| replay ER, larger buffer | 23.09% | -35.23% |
| EWC 2.0 | 22.96% | -35.37% |
| SI 0.1 | 23.06% | -35.41% |
| weaker replay ER baseline | 21.17% | -40.84% |

Interpretation:

- the strongest mechanistic result remains replay-path placement
- tuned backbone replay remains stronger than the completed regularizer-heavy follow-ons

## 2. Synthetic Sensor Routing Ladder

| Variant | Canonical mean acc. | Merged mean acc. | Canonical - Merged |
|---|---:|---:|---:|
| baseline | 1.0000 | 0.5417 | +0.4583 |
| capacity-matched baseline | 1.0000 | 0.6233 | +0.3767 |
| drop smartwatch | 1.0000 | 0.6683 | +0.3317 |
| drop room environment | 1.0000 | 0.2050 | +0.7950 |
| drop robot arm | 0.8967 | 0.2633 | +0.6333 |
| corrupt bindings | 1.0000 | 0.7283 | +0.2717 |
| noise smartwatch | 1.0000 | 0.9367 | +0.0633 |
| noise room environment | 1.0000 | 0.9383 | +0.0617 |
| noise robot arm | 0.8817 | 0.9750 | -0.0933 |

Supervisor summary:

- final ladder status: `canonical-dominant`

Interpretation:

- typed routing wins strongly overall on the hardened synthetic benchmark
- the result is not unconditional; `noise_robot_arm` is the clear counterexample

## 3. Real Audio Results

### Bootstrap Audio

| Sequence | Dataset | Metric |
|---|---|---:|
| `real-audio-seq-001` | `speech_commands` | 88.32% mean accuracy |

### Canonical Integrated Audio

| Sequence | Dataset | Mean accuracy |
|---|---|---:|
| `real-audio-seq-003` | `speech_commands` | 88.32% |
| `real-audio-seq-003` | `esc50` | 41.83% |
| `real-audio-seq-003` | `urbansound8k` | 60.57% |
| `real-audio-seq-003` | `fsd50k` | 9.74% |

Status:

- `real-audio-seq-003`: `completed`
- blocked count: `0`
- canonical integrated audio lane: **promotable**

Interpretation:

- the real-audio path is no longer only bootstrap scaffolding
- `fsd50k` remains the hardest current audio regime

## 4. Real Sensor Results

### Bootstrap Sensor

| Sequence | Dataset | Canonical - Merged |
|---|---|---:|
| `real-seq-004` | `wisdm` | +0.0419 |

### Canonical Integrated Sensor

| Sequence | Dataset | Canonical - Merged |
|---|---|---:|
| `real-sensor-seq-002` | `wisdm` | +0.0419 |
| `real-sensor-seq-002` | `pamap2` | +0.1820 |
| `real-sensor-seq-002` | `opportunity` | +0.0087 |
| `real-sensor-seq-002` | `uci_har` | -0.0033 |
| `real-sensor-seq-002` | `hhar` | -0.0006 |
| `real-sensor-seq-002` | `mhealth` | -0.0938 |
| `real-sensor-seq-002` | `selfback` | +0.0215 |
| `real-sensor-seq-002` | `geomagnetic_wlan` | +0.0000 |
| `real-sensor-seq-002` | `realworld_har` | +0.0238 |

Status:

- `real-sensor-seq-002`: `completed`
- blocked count: `0`
- canonical integrated sensor lane: **promotable**

Interpretation:

- the integrated real-sensor story is mixed rather than uniformly canonical-dominant
- that is scientifically useful because it gives both supporting and limiting cases for the distinction hypothesis

## 5. Real-Sensor Buckets

Analysis convention used here:

- positive: `> +0.01`
- neutral: `-0.01 <= gap <= +0.01`
- negative: `< -0.01`

### Positive

- `pamap2`: `+0.1820`
- `wisdm`: `+0.0419`
- `realworld_har`: `+0.0238`
- `selfback`: `+0.0215`

### Neutral

- `opportunity`: `+0.0087`
- `geomagnetic_wlan`: `+0.0000`
- `uci_har`: `-0.0033`
- `hhar`: `-0.0006`

### Negative

- `mhealth`: `-0.0938`

## 6. Current Scientific Read

Across the completed program:

- replay-through-backbone is still the clearest mechanistic win
- typed routing is strongly supported in the synthetic ladder
- typed routing is partially supported and bounded in integrated real-sensor data
- distinction currently looks like the best synthesis lens, but not yet a universal law

## 7. Source Files

- [paper/sections/06-results.md](/Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md)
- [docs/completed-summary-analysis.md](/Users/zachshallbetter/Projects/Perception/docs/completed-summary-analysis.md)
- [paper/figures/generated/results_package/real_data/real_data_status.json](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package/real_data/real_data_status.json)
- [paper/figures/generated/results_package/real_data/real_audio_evidence.csv](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package/real_data/real_audio_evidence.csv)
- [paper/figures/generated/results_package/real_data/real_sensor_evidence.csv](file:///Users/zachshallbetter/Projects/Perception/paper/figures/generated/results_package/real_data/real_sensor_evidence.csv)
