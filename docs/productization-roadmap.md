# Productization Roadmap

This document explains how the current research result can become a usable system.

## What A User Can Actually Take From The Current Results

Assuming the current paper claims hold up, the most practical outputs are:

1. **An architecture pattern**
   - keep modality identity explicit
   - use modality-specific towers
   - integrate through a shared perceptual core rather than collapsing all streams immediately

2. **A continual-learning rule**
   - if a system must adapt over time, replay must reach the shared feature stack
   - replay that only touches late heads is not enough

3. **A runtime-contract rule**
   - define modalities canonically at the boundary
   - derive routing/setup from that canonical registry
   - treat the registry as operational infrastructure, not just metadata

## Who Should Actually Use This Now

The present state is appropriate for:

- research teams evaluating multimodal routing and continual-learning design choices
- applied ML teams building a narrow domain-specific perception prototype
- platform teams that need a disciplined update loop before they need a broad product surface

The present state is not appropriate for:

- teams expecting a turnkey general-purpose multimodal foundation model
- teams that need a fully validated large-scale benchmark suite before adopting any part of the architecture
- teams that cannot tolerate synthetic-benchmark evidence in the early prototype stage

## Productization Stages

### Stage 1: Research-Grade Internal Tool

Goal:
- make the repository usable by a small internal team without paper-specific tribal knowledge

Requirements:
- stable environment bootstrap
- one canonical launch command per experiment family
- durable result packaging
- bounded automation with readable stop states

This stage is largely in place now.

### Stage 2: Domain-Specific Perception Prototype

Goal:
- target one real sensor domain rather than broad synthetic scope

Recommended domains:
- wearable + room sensing
- robot arm + ambient sensing
- industrial equipment monitoring

Requirements:
- replace synthetic `sensor_streaming` with real or recorded domain data
- define missing-data and sensor-failure behavior
- add latency, throughput, and calibration metrics
- evaluate whether typed routing still beats merged routing under realistic noise

This is the highest-value next practical step.

Suggested first proof domains:

1. wearable + room sensing for context/activity inference
2. robot arm + ambient sensing for failure or state detection
3. industrial equipment monitoring with multi-sensor anomaly detection

### Stage 3: Continual Update Service

Goal:
- move from offline experiments to incremental model updates in operation

Requirements:
- persisted replay store with retention policy
- model/version registry
- dataset and modality lineage
- safe incremental retraining loop
- rollback and evaluation gates before promotion

At this stage, the replay findings become operational rather than purely academic.

### Stage 4: User-Facing Product Surface

Goal:
- expose a useful application on top of the perception stack

Example surfaces:
- anomaly detection dashboard
- activity recognition service
- context-aware robotics controller
- multimodal monitoring assistant

Requirements:
- product-specific outputs, not just benchmark metrics
- clear explanations of sensor health and modality confidence
- privacy and retention policy for collected signals
- failure-state behavior when one modality becomes unavailable

## What Must Be True Before Strong Product Claims

Before claiming more than “research prototype,” the following should be true:

1. `sensor_streaming`-style evidence is replaced or backed by real-domain benchmarks.
2. The capacity-matched routing comparison still supports typed routing.
3. The system handles missing sensors, noisy sensors, and calibration drift cleanly.
4. Continual updates are measured for both retention and operational safety.
5. Result packaging includes user-facing metrics, not only paper metrics.

## Recommended Near-Term Path

1. Finish the active bounded sensor evidence ladder.
2. Decide whether typed routing remains favorable after the capacity-matched and stress-test gates.
3. Choose one real sensor domain and build a narrow prototype there.
4. Treat the current continual-learning result as the stability mechanism inside that prototype.

That path turns the current work from “interesting research platform” into “credible domain-specific perception system.”

## Minimum Productization Checklist

Before turning this repository into a real deployed service, the minimum checklist should include:

1. one chosen sensor domain with a stable data contract
2. real or recorded benchmark data replacing synthetic-only evaluation
3. a replay-store policy with retention and rollback rules
4. capacity-matched routing evidence if canonical routing is being used as a product rationale
5. user-facing metrics that describe system usefulness, not only paper metrics like CIL/BWT
