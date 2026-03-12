# Live System Runtime Contract

This document formalizes the runtime constraints placed on the live perceptual system, guaranteeing safety and determinism in production operations.

## 1. Schema Strictness & Discriminators
All incoming multimodal streaming data is constrained by a strict set of discriminators defined in the SSoT (Single Source of Truth). Any `FrameEnvelope` with an unrecognized modality discriminator will trigger an immediate hard error (`SCHEMA_CHANGED` or equivalent). There is no "best effort" mapping. 

## 2. No Silent Fallbacks (NoFallback)
A `MISSING` state for a sensor must be expressly declared in the input event schema or mapped explicitly through the stream synchronizer. There are no implicit defaults. If an input is `DEGRADED`, the model handles it explicitly. If a modality essential to the safety policy is missing, the perceptual core must refuse prediction with explicit error codes instead of guessing.

## 3. Pure Deterministic Processing (Determinism & Canon)
The inference step mapping an input window into a predicted output acts exactly like a pure function:
`resolveInferencePlan(frozen_window, canonical_model_weights) -> InferencePlan`

We strictly ban the usage of `Date.now()`, random operations, or unordered dictionary iterations inside synchronization logic or the inference core. Timestamp buffering relies entirely on client timestamps within discrete buckets to guarantee repeatable and testable inputs.

## 4. Bounded Streaming Outputs 
Models yield typed payload structures (e.g. `InferencePlan`), capturing not just logit prediction, but an actionable `is_safe` confidence flag and `error_code` if applicable.

When models fail early (e.g. missing required modality), they return an error code safely instead of bringing down the system process. Downstream webhooks and controllers operate exclusively over the frozen inference plan.
