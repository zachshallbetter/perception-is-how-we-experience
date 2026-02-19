# Application Domain: Smart Home Automation

**Challenge**: Activity recognition with multiple concurrent events and variable sensor configurations
**Modalities**: Audio (microphone) + Motion (accelerometer, gyro) + Environmental (temperature, light, occupancy) + Temporal (time of day, day of week)
**Benchmark**: Synthetic smart home activity recognition dataset
**Status**: 📋 Planned - experiments TBD after main paper

## Domain-Specific Challenges

1. **Concurrent Activities**: Multiple simultaneous activities (cooking + watching TV, etc.)
2. **Sparse Events**: Many time periods have no interesting activity
3. **Variable Modalities**: Not all homes have all sensors
4. **Scalability**: Can system handle 10+ modalities?
5. **Power Constraints**: Embedded devices have limited compute

## Benchmark Design

### Dataset
- Synthetic smart home with 500+ house-days of activity
- Activities: cooking, cleaning, eating, sleeping, entertainment, work, exercise
- Sensors: microphone, motion (3 rooms), temperature, light, occupancy
- Goal: Predict all concurrent activities in 5-minute windows
- Train/test split: 400 / 100 days

### Metrics
- Multi-label accuracy (Hamming loss)
- Per-activity F1 score
- Modality robustness (performance with missing sensors)
- Energy efficiency (inference power consumption)
- Inference latency (ms)

### Baselines
- Perceiver IO baseline
- Per-room activity classifiers
- Concatenation + MLP
- Hidden Markov Model (standard baseline)

## Research Questions

1. **Can we recognize multiple concurrent activities?** (Expected: 70-80% Hamming accuracy)
2. **What's the minimum sensor set needed?** (Expected: audio + motion sufficient)
3. **Does system scale to 10+ modalities?** (Expected: graceful degradation)
4. **Can we achieve real-time inference?** (Expected: <100ms on ARM)

## Paper Outline

1. **Introduction** - Smart home automation needs
2. **Related Work** - Activity recognition, IoT sensor fusion
3. **Method** - Lightweight architecture for embedded systems
4. **Experiments** - Sensor configurations, baselines, benchmarks
5. **Results** - Accuracy, robustness, efficiency
6. **Discussion** - Real-world deployment challenges
7. **Conclusion** - Future work in edge AI

## Timeline

- **Week 4**: Activity taxonomy, dataset synthesis
- **Week 5**: Model training, efficiency profiling
- **Week 6**: Robustness analysis, embedded testing
- **Week 7-8**: Paper finalization, submission prep

---

See parent directory (`../README.md`) for applications portfolio.
See `../../README.md` for supplementary papers strategy.
