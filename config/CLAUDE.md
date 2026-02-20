# Memory — Perception Is How We Experience

## Me
Zach. Sole author of this paper. Goal: refine, validate, and publish.

## This Project
| Term | Meaning |
|------|---------|
| **the paper** | "Perception Is How We Experience" — ML/NLP deep learning architecture proposal |
| **the Perceiver** | The proposed architecture in the paper (naming conflict — see Issues) |
| **the review** | Comprehensive Review Report in `Review/` — generated Feb 19 2026 |
| **the review doc** | `Review/Perception - Comprehensive Review Report.docx` |

## Key Terms
| Term | Meaning |
|------|---------|
| **CDP** | Composite Dot Product — the paper's attention formula (has bug: missing `1/√d_k`) |
| **MIME encoding** | Novel unified sensory encoding via MIME-Types — most original contribution |
| **replay** | Biologically-inspired memory consolidation mechanism (see Recall vs Replay.md) |
| **PerceiverTower** | Modality-specific processing module in the proposed architecture |
| **Jaegle et al.** | DeepMind's published Perceiver (2021, ICML) — naming conflict with this paper |
| **sensory tokenization** | Token memory bank concept for storing sensory features |

## Active Issues (from review)
| # | Severity | Status | Issue |
|---|----------|--------|-------|
| C1 | CRITICAL | ✅ FIXED (Feb 19) | Architecture name "Perceiver" conflicts with Jaegle et al. (2021) — renamed to "Sensory Perceiver"; Related Work section added with comparison table; citations [2, 3] added |
| C2 | CRITICAL | ✅ FIXED (Feb 19) | Abstract claims experimental results that don't exist — replaced with hypothesis/proposal language throughout |
| C3 | CRITICAL | ✅ FIXED (Feb 19) | Paper says no convolutions; code uses `Conv2d` — Abstract and Introduction updated to accurately describe Conv2d front-end + attention-based global integration |
| C4 | CRITICAL | ✅ FIXED (Feb 19) | CDP formula missing `1/√d_k` — corrected to `softmax(QKᵀ/√d_k)·V`; explanation and Vaswani [1] citation added; Dynamic Weighting.md pseudocode updated |
| C5 | CRITICAL | ✅ FIXED (Feb 19) | Background placeholders filled: Perceptron (Rosenblatt 1958 [8], McCulloch & Pitts 1943 [20]) and CNN (LeCun et al. 1998 [6]) subsections written |
| M1 | MAJOR | ⬜ OPEN | No experiments, no datasets, no benchmarks — needed for any publication |
| M2 | MAJOR | ⬜ OPEN | `PerceiverTower.forward()` has incompatible tensor shapes (Conv2d 4D → MHA 3D) |
| M3 | MAJOR | ✅ FIXED (Feb 19) | `DropPath` and `@register_model` — added `from timm.models.layers import DropPath` and `from timm.models.registry import register_model`; removed unused `BatchNorm2d` import |
| M4 | MAJOR | ⬜ OPEN | Hebbian learning and contrastive loss conflated — pick one paradigm |
| M5 | MAJOR | ⬜ OPEN | MIME-Type encoding is 1 paragraph with no implementation — needs full spec |

## File Structure (after reorganization Feb 19 2026)
```
perception-is-how-we-experience/
├── Perception Is How We Experience.md  ← main paper
├── Architecture.md
├── Sensation.md
├── Interpretation.md
├── Paper Overview.md
├── Protocols.md
├── Reference.md
├── Recall versus Replay.md
├── Todo.md
├── Sensor Types.md
├── Code.md
├── Examples.md
├── Literature.md
├── Code/
│   └── Dynamic Weighting.md
├── Examples/
│   ├── Autonomous Driving in Heavy Traffic.md
│   └── Patient Monitoring System.md
├── Literature/
│   ├── Attention Is All You Need (Vaswani et al., 2017).pdf
│   ├── The Annotated Transformer.pdf
│   └── The Ultimate Guide to Transformer Deep Learning.pdf
├── Review/
│   └── Perception - Comprehensive Review Report.docx
└── Sensor Types/
    └── Sensor References/
        └── [14 modality files: Auditory, Visual, Tactile, Olfactory,
            Gustatory, Vestibular, Proprioception, Interoception,
            Chronoception, Equilibrioception, Magnetoception,
            Nociception, Thermoception, Sensor Overlap]
```

## Preferences
- Decisive, concise responses — no fluff
- Proceed when context is sufficient; ask only when it changes the outcome
- Never fabricate citations, results, or experimental data
- Full comprehensive pass preferred over iterative partial passes
→ Full project context: memory/projects/perception-paper.md
→ Full glossary + citations: memory/context/citations.md
