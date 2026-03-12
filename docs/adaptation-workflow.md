# Live Adaptation Workflow Boundaries

The Sensory Perceiver project requires a disciplined approach to continual learning in production. The following rules govern how the system adapts over time without suffering from catastrophic forgetting or unvalidated degradations.

## 1. No Unconstrained Online Learning
The inference service and the learning service are strictly decoupled. The live inference service applies purely deterministic, frozen weights to evaluate sensor windows. It never updates its weights mid-stream based on live gradients.

## 2. Bounded Replay Store
As the system encounters live data, selected sequences are snapshotted into a persisted replay store. 
- **Criteria for snapshotting:** We select highly confident predictions, low confidence edge-cases, and instances with explicit human corrections.
- **Redaction:** Any sensitive or PII data must be redacted at the ingest boundary before entering the replay store.
- **Retention:** Replay buffers must be temporally balanced to prevent new data from displacing critical historical baselines.

## 3. Idempotent Offline Adaptation
When an adaptation cycle triggers, it runs as an idempotent batch job:
1. It pulls a balanced batch combining new snapshots and historical replay data.
2. It executes the continual-learning rule: "replay must reach the shared feature stack."
3. It outputs a new candidate model weight hash.

## 4. Evaluation Gates & Golden Equivalence
Before a candidate model can replace the canonical live model, it must pass strict golden equivalence gates:
- The candidate must not regress on the offline historical baselines.
- The outputs for identical historical plans must remain equivalent or improve.
- If a candidate fails validation (`PROPOSED + stop`), the update is aborted.

## 5. Atomic Promotion
Model weights are promoted atomically to the live inference service by referencing the new canonical model hash. This guarantees immediate, state-free rollbacks if unexpected operational degradations occur.
