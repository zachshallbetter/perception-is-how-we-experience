# Memory — Perception Is How We Experience

## Me
Zach. Sole author of this paper. Goal: refine, validate, and publish.

## This Project
| Term | Meaning |
|------|---------|
| **the paper** | "Perception Is How We Experience" — ML/NLP deep learning architecture proposal |
| **Sensory Perceiver** | The proposed architecture (renamed from "Perceiver" to avoid conflict with Jaegle et al.) |
| **the review** | Comprehensive Review Report in `review/` — generated Feb 19 2026 |
| **the review doc** | `review/Perception - Comprehensive Review Report.docx` |

## Key Terms
| Term | Meaning |
|------|---------|
| **CDP** | Composite Dot Product — the paper's attention formula; corrected to include `1/√d_k` |
| **MIME encoding** | Novel unified sensory encoding via MIME-Types — most original contribution |
| **replay** | Experience replay phase for continual learning; uses NT-Xent contrastive loss |
| **PerceiverTower** | Modality-specific processing module (Conv2d front-end → flatten → MLP → 768-d embedding) |
| **Jaegle et al.** | DeepMind's published Perceiver (2021, ICML) — compared in Related Work section |
| **sensory tokenization** | Token memory bank concept for storing sensory features |

## Active Issues (from review — all CF/MJ/MN fixed Feb 20 2026)
| # | Severity | Status | Issue |
|---|----------|--------|-------|
| C1 | CRITICAL | ✅ FIXED | Architecture name — renamed to "Sensory Perceiver"; Related Work + comparison table added |
| C2 | CRITICAL | ✅ FIXED | Abstract overclaimed results — replaced with hypothesis/proposal + synthetic validation language |
| C3 | CRITICAL | ✅ FIXED | Conv2d contradiction — Abstract/Intro updated to accurately describe Conv2d front-end |
| C4 | CRITICAL | ✅ FIXED | CDP missing `1/√d_k` — corrected; Hebbian/sleep conflation removed; committed to NT-Xent |
| C5 | CRITICAL | ✅ FIXED | Background placeholders filled (Rosenblatt 1958, LeCun 1998, GWT) |
| M1 | MAJOR | 🟡 PARTIAL | Real results: baseline (CIL 5.02%, BWT −53.6%) + tuned (CIL 5.16%, BWT −57.6%) on Split-CIFAR-100. Full benchmarking vs. EWC/GEM/PackNet on complete 50-task split required before submission |
| M2 | MAJOR | ✅ FIXED | PerceiverTower tensor shape — added `flatten(start_dim=1)` before MLP |
| M3 | MAJOR | ✅ FIXED | Missing imports (`DropPath`, `register_model`) added; unused imports removed |
| M4 | MAJOR | ✅ FIXED | Hebbian/contrastive conflation resolved — committed to NT-Xent throughout |
| M5 | MAJOR | ✅ FIXED | MIME-Type Extension Protocol specification added (3-step registration, conflict resolution) |

## File Structure (after reorganization Feb 20 2026)
```
perception-is-how-we-experience/
├── CLAUDE.md                     ← this file
├── README.md
├── paper/                        ← canonical paper
│   ├── Perception Is How We Experience.md   ← main paper (105k chars)
│   ├── Paper Overview.md
│   ├── sections/                 ← section drafts (01–09)
│   └── figures/                  ← figure generation scripts
├── src/                          ← model implementation
├── baselines/                    ← Perceiver IO baseline code
├── results/                      ← synthetic results + figures
│   └── figures/                  ← 10 generated figure PNGs
├── supplementary/                ← supplementary papers
│   ├── architecture-paper/
│   ├── orchestrator-paper/
│   └── applications/
├── literature/                   ← reference PDFs + markdown summaries
│   ├── *.pdf                     ← 22 papers
│   └── markdown/                 ← converted summaries
├── reference/                    ← supporting reference material
│   ├── sensor-types/             ← 14 modality files
│   │   └── Sensor References/   ← Auditory, Visual, Tactile, … (14 files)
│   └── code-snippets/
├── review/                       ← review documents
│   └── Perception - Comprehensive Review Report.docx
├── memory/                       ← project memory
│   ├── projects/perception-paper.md
│   └── context/citations.md
├── config/                       ← config + requirements
├── scripts/                      ← utility scripts
└── archive/                      ← all old versions + session artifacts
    ├── session-files/            ← audit docs, progress files, etc.
    ├── 00-START / 01-PLANNING / 02-PAPER / 03-RESEARCH / Code/
    └── …
```

## Preferences
- Decisive, concise responses — no fluff
- Proceed when context is sufficient; ask only when it changes the outcome
- Never fabricate citations, results, or experimental data
- Full comprehensive pass preferred over iterative partial passes
→ Full project context: memory/projects/perception-paper.md
→ Full glossary + citations: memory/context/citations.md
