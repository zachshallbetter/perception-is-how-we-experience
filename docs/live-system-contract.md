# Live System Runtime Contract

This document formalizes the runtime constraints placed on the live perceptual system, guaranteeing safety and determinism in production operations.

## 1. Schema Strictness & Discriminators
All incoming multimodal streaming data is constrained by a strict set of discriminators defined in the SSoT (`src/runtime/live_schema.py`). Any `FrameEnvelope` with an unrecognized modality discriminator triggers an immediate `AppError` (`SCHEMA_VALIDATION_FAILED` or equivalent). There is no "best effort" mapping. 

## 2. Explicit Missing Data (NoFallback)
A `MISSING` state for a sensor is mapped explicitly through the stream synchronizer. The synchronizer outputs a deterministic `valid_mask` tensor alongside the data tensor rather than relying on silent zero-fills. If an input is `DEGRADED`, the live inference boundary passes that mask through to the model in the canonical per-modality payload instead of discarding it.

## 3. Pure Deterministic Processing (Determinism & Canon)
The inference step mapping an input window into a predicted output acts exactly like a pure function:
`resolveInferencePlan(frozen_window, canonical_model_weights) -> InferencePlan`

We strictly ban the usage of `Date.now()`, random operations, or unordered dictionary iterations inside synchronization logic or the inference core. Timestamp buffering relies entirely on client timestamps within discrete buckets to guarantee repeatable and testable inputs.

## 4. Strict Failure Boundaries
When models fail due to corrupted weights, runtime errors, or unhandled exceptions during execution, the inference core does not silently swallow the error into a "safe" object. It immediately delegates the error to the strict `AppError` (fail + TODO + stop) flow.

Live inference also requires an explicit model capability declaration (`supports_live_valid_mask = True`). The preferred path is a native mask-aware model that consumes the canonical per-modality payload directly. If the underlying model is still tensor-only, it must be wrapped by an explicit live adapter that consumes the canonical payload and applies deterministic masking at the boundary. The inference service does not silently drop the mask or revert to the old tensor-only contract.

When known logical failures occur (such as missing required modalities), the core can return a safe prediction plan with a descriptive error code `MISSING_REQUIRED_MODALITIES` without crashing the downstream controllers.
