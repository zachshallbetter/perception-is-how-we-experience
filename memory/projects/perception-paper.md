# Project: Perception Is How We Experience

**Status:** Active — in review/revision phase
**Author:** Zach
**Target:** Academic journal + IP documentation + exploratory system implementation
**Discipline:** Computer Science — Machine Learning (Deep Learning), NLP
**Last reviewed:** February 19, 2026 (comprehensive review by Claude)

---

## What the Paper Proposes

A neural network architecture called **"the Perceiver"** designed to process multimodal
sensory data (vision, audio, touch, smell, taste, vestibular, proprioception, interoception,
and more) in a unified framework inspired by human perception.

Core ideas:
- **MIME-Type unified encoding** — classifying sensor streams by MIME content-type as
  an input routing and encoding strategy (most original contribution)
- **PerceiverTower architecture** — modality-specific processing towers feeding a shared
  cross-modal attention layer
- **Replay mechanism** — biologically-inspired memory consolidation (analogous to
  hippocampal replay during REM sleep) to prevent catastrophic forgetting
- **Sensory tokenization** — a token memory bank for storing and retrieving sensory features
- **8+ sensory modalities** — extends beyond vision+language to include proprioception,
  interoception, vestibular sense, etc.

---

## Document Inventory

| File | Role |
|------|------|
| `Perception Is How We Experience.md` | Main paper body (87KB — primary document) |
| `Architecture.md` | Architecture detail — 7-component breakdown |
| `Sensation.md` | Neural coding background; 8+ sensory input types |
| `Interpretation.md` | Application examples (from Zephyr Beta 7b) |
| `Paper Overview.md` | High-level summary of all sections |
| `Protocols.md` | IoT communication protocols (Bluetooth, Zigbee, Wi-Fi, etc.) |
| `Reference.md` | Hebbian learning reference |
| `Recall versus Replay.md` | Clarifies replay ≠ recall (important for M4 fix) |
| `Todo.md` | Original future-work notes from paper development |
| `Code/Dynamic Weighting.md` | PerceiverTower implementation (Python + pseudocode) |
| `Examples/Autonomous Driving...` | Case study 1 |
| `Examples/Patient Monitoring...` | Case study 2 |
| `Literature/*.pdf` | 3 reference PDFs (Vaswani 2017, Annotated Transformer, Guide) |
| `Sensor Types/Sensor References/*.md` | 14 modality deep-dives |
| `Review/...Review Report.docx` | Comprehensive review (Feb 19 2026) — 7 parts |

---

## Architecture Summary

The proposed PerceiverTower has 7 components (A–G + feedback):
- **A** Init params: `num_inputs`, `dim=768`, `depth=24`, `heads=8`, `mlp_dim=1024`
- **B** Conv2d — initial feature extraction *(contradicts "no convolutions" claim)*
- **C** MLP — `Linear(dim*heads, mlp_dim)` → ReLU → DropPath *(Linear dim is wrong)*
- **D** MultiheadAttention — QKV scaled dot-product *(CDP formula missing `1/√d_k`)*
- **E** Fully connected output layers
- **F** LayerNorm
- **G** Tower assembly as ModuleList *(forward() not implemented — needs Sequential)*

---

## Review Findings Summary (Feb 19 2026)

### Critical (5) — blocks submission
1. **C1 — Naming conflict:** "Perceiver" = Jaegle et al. (2021) ICML. Must cite + differentiate or rename.
2. **C2 — Fabricated experiments:** Abstract: "Our experiments demonstrate/show..." — no experiments exist.
3. **C3 — Conv contradiction:** Intro says no convolutions; code uses `Conv2d`.
4. **C4 — CDP formula:** `softmax(qTk)*v` is missing `1/√d_k`. Correct: `softmax(QKᵀ/√d_k)*V`.
5. **C5 — Incomplete background:** "Perceptron:" and "CNN:" are empty placeholder headers.

### Major (6) — required for journal
- No experiments/benchmarks (suggest: AudioSet, MSCOCO, ModelNet40, Split-CIFAR)
- `PerceiverTower.forward()` tensor shape pipeline broken (Conv2d 4D → MHA expects 3D)
- `DropPath`, `@register_model` not imported (need `from timm.models.layers import ...`)
- Hebbian learning (unsupervised, no backprop) conflated with contrastive loss (self-supervised)
- MIME-Type encoding needs full implementation spec (routing table, embedding bridge, unlabeled streams)
- Code artifacts: `pythonCopy code` prefix on several blocks; `BatchNorm2d` imported but unused

### Strengths to preserve
- MIME-Type sensory routing (genuinely novel)
- 8+ modality coverage with detailed sensor references
- Replay/memory consolidation mechanism
- Sensory tokenization / token memory bank
- IoT sensor integration perspective (Protocols.md)

---

## Revision Roadmap (Phases)

### Phase 1 — Critical fixes (do first)
- [x] C1: Renamed to "Sensory Perceiver"; Jaegle [2, 3] cited; Related Work section with comparison table added
- [x] C2: All fabricated experimental claims removed — replaced with hypothesis/proposal language
- [x] C3: Abstract and Introduction updated: Conv2d front-end acknowledged; "no convolutions" claim scoped accurately to global integration layer
- [x] C4: CDP formula corrected to `softmax(QKᵀ/√d_k)·V`; `d_k` explained; Vaswani [1] cited; Dynamic Weighting.md pseudocode updated
- [x] C5: Perceptron (Rosenblatt 1958, McCulloch & Pitts 1943) and CNN (LeCun et al. 1998) background subsections written
- [ ] M2: Fix `PerceiverTower.forward()` reshape pipeline (Conv2d 4D → MHA 3D)
- [x] M3: timm imports added (`DropPath`, `register_model`); unused `BatchNorm2d` removed

### Phase 2 — Journal-ready
- [ ] Add ~18 missing citations (full list in review doc Part III)
- [ ] Resolve Hebbian vs. contrastive paradigm
- [ ] Develop MIME-Type encoding with equations + routing table
- [ ] Restructure sections: Intro → Related Work → Method → Experiments → Discussion
- [ ] Remove Notion aside blocks and code copy artifacts

### Phase 3 — Top-tier venue
- [ ] Run experiments on ≥2 multimodal benchmarks
- [ ] Add ablation study on replay mechanism
- [ ] Write Related Work: vs. PerceiverIO, ViLT, CLIP, FLAVA
- [ ] Formalize MIME-Type routing with algorithm pseudocode

---

## Session History

| Date | What happened |
|------|---------------|
| Feb 19 2026 | Full comprehensive review completed → Review Report (.docx) generated |
| Feb 19 2026 | All files renamed (Notion UUIDs stripped); Commands → Literature; review → Review/ |
| Feb 19 2026 | Memory initialized (this file) |
| Feb 19 2026 | Phase 1 critical fixes applied: C1 (rename + Related Work), C2 (abstract claims), C3 (convolution), C4 (CDP formula), C5 (background), M3 (imports) |
