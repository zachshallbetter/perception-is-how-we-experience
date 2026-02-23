# Patent Track Draft

This draft is incubated from docs and maintained in the paper corpus for filing workflow alignment.

# Candidate: Canonical MIME-Case Runtime Contract for Multimodal ML

Status: filing-ready draft
Priority: P0

## Problem

Multimodal systems drift when modality identity, schema width, and experiment routing are configured in multiple disconnected places.

## Proposed invention

A canonical runtime contract where each modality case is defined as:

`InputCase = {modality, mime_type, feature_dim, description}`

and runtime behavior is derived only from this registry for:

1. Modality setup resolution.
2. MIME binding validation.
3. Experiment-level dataset compatibility checks.

## Code evidence

- Canonical input cases: `src/runtime/mime_cases.py`
- Runtime contract enforcement: `src/runtime/contracts.py`
- Documentation alignment: `docs/sensor-streaming-mime-cases.md`

## Claim skeleton

1. A runtime governance method for multimodal machine learning comprising canonical MIME-case registry definition and contract-bound derivation of modality setup and validation logic.

## Enablement package

- Contract conformance tests for all registry entries.
- Negative tests for unknown modality and MIME mismatch.
- End-to-end training/evaluation boot validations.

## TODO

- [ ] Add registry schema versioning and migration policy.
- [ ] Add strict deny-by-default for unregistered modality keys at model boundary.
- [ ] Emit structured contract version in training artifacts.
