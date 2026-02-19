# Perception Is How We Experience — Research Portfolio

A comprehensive research project on multimodal learning with the Sensory Perceiver architecture.

**Current Status**: Planning complete, ready for Week 1 experiments

---

## Quick Navigation

### 🚀 Getting Started
- **New to this project?** → [`00-START/START_HERE.md`](00-START/START_HERE.md)
- **Project summary** → [`00-START/COMPLETE_DELIVERY_SUMMARY.md`](00-START/COMPLETE_DELIVERY_SUMMARY.md)
- **Latest updates** → [`00-START/SESSION_SUMMARY.md`](00-START/SESSION_SUMMARY.md)

### 📋 Planning & Execution
- **Master execution plan** (10 weeks) → [`01-PLANNING/MASTER_EXECUTION_PLAN.md`](01-PLANNING/MASTER_EXECUTION_PLAN.md)
- **Week-by-week ready reference** → [`01-PLANNING/EXECUTION_READY.md`](01-PLANNING/EXECUTION_READY.md)
- **Publication strategy** → [`01-PLANNING/PUBLICATION_STRATEGY.md`](01-PLANNING/PUBLICATION_STRATEGY.md)
- **6-paper portfolio strategy** → [`01-PLANNING/MULTI_PAPER_STRATEGY.md`](01-PLANNING/MULTI_PAPER_STRATEGY.md)

### 📄 Paper & Research
- **Main paper** → [`02-PAPER/Perception Is How We Experience.md`](02-PAPER/Perception%20Is%20How%20We%20Experience.md)
- **Paper overview & structure** → [`02-PAPER/Paper Overview.md`](02-PAPER/Paper%20Overview.md)
- **Architecture details** → [`02-PAPER/Architecture.md`](02-PAPER/Architecture.md)
- **Sensation & perception mechanisms** → [`02-PAPER/Sensation.md`](02-PAPER/Sensation.md)
- **All referenced literature** → [`02-PAPER/Literature.md`](02-PAPER/Literature.md)

### 🔬 Research Planning
- **Paper outlines** (all 6 papers) → [`03-RESEARCH/PAPER_OUTLINES.md`](03-RESEARCH/PAPER_OUTLINES.md)
- **Publication-quality figures** → [`03-RESEARCH/FIGURE_TEMPLATES.md`](03-RESEARCH/FIGURE_TEMPLATES.md)
- **Figure integration status** → [`03-RESEARCH/FIGURES_INTEGRATION_COMPLETE.md`](03-RESEARCH/FIGURES_INTEGRATION_COMPLETE.md)
- **Application domain frameworks** → [`03-RESEARCH/APPLICATION_FRAMEWORKS.md`](03-RESEARCH/APPLICATION_FRAMEWORKS.md)

### 💻 Code & Implementation
All code organized in `code/` directory:
- **Models** → `code/models.py` (SensoryPerceiver + Perceiver IO architectures)
- **Training** → `code/train.py` (main trainer), `code/train_perceiver_io.py` (baseline)
- **Experiments** → `code/experiment_orchestrator.py` (GPU coordination)
- **Data** → `code/data_loader.py` (dataset loading)
- **Figure templates** → `figures/` (7 publication-quality figure templates)

### ⚙️ Configuration & Setup
- **Project configuration** → `config/CLAUDE.md` (project instructions)
- **Requirements** → `config/requirements.txt` (Python dependencies)
- **Setup guide** → `config/SETUP_GUIDE.md` (environment setup)
- **Experiment config** → `config/experiment_config.yaml` (hyperparameters)

### 🔧 Scripts & Execution
- **Run experiments** → `scripts/run_experiments.ps1` (Windows) or `scripts/run_experiments.bat`
- **Validate setup** → `scripts/validate_setup.py`

---

## Directory Structure

```
perception-is-how-we-experience/
│
├── 00-START/                          ← Entry points (start here!)
│   ├── START_HERE.md
│   ├── COMPLETE_DELIVERY_SUMMARY.md
│   └── SESSION_SUMMARY.md
│
├── 01-PLANNING/                       ← Strategy & execution plans
│   ├── MASTER_EXECUTION_PLAN.md       ← Week-by-week timeline
│   ├── EXECUTION_READY.md
│   ├── MULTI_PAPER_STRATEGY.md
│   ├── PUBLICATION_STRATEGY.md
│   └── [4 other strategy docs]
│
├── 02-PAPER/                          ← Main paper & content
│   ├── Perception Is How We Experience.md
│   ├── Paper Overview.md
│   ├── Architecture.md
│   ├── Sensation.md
│   ├── Interpretation.md
│   ├── Reference.md
│   └── [other paper sections]
│
├── 03-RESEARCH/                       ← Research planning & analysis
│   ├── PAPER_OUTLINES.md              ← Outlines for all 6 papers
│   ├── FIGURE_TEMPLATES.md
│   ├── FIGURES_INTEGRATION_COMPLETE.md
│   └── APPLICATION_FRAMEWORKS.md
│
├── code/                              ← Python implementations
│   ├── models.py                      ← SensoryPerceiver architecture
│   ├── train.py                       ← Training pipeline
│   ├── experiment_orchestrator.py     ← GPU coordination (3 GPUs)
│   ├── perceiver_io_baseline.py       ← Baseline implementation
│   └── [other code files]
│
├── figures/                           ← Publication-quality figures
│   ├── __init__.py
│   ├── README.md
│   ├── accuracy_curves.py             ← Template 1
│   ├── ablation_results.py            ← Template 2
│   ├── efficiency_frontier.py         ← Template 3
│   ├── comparison_table.py            ← Template 4
│   ├── architecture_diagram.py        ← Template 5 (graphviz)
│   ├── attention_heatmap.py           ← Template 6
│   └── radar_comparison.py            ← Template 7
│
├── config/                            ← Configuration & setup
│   ├── CLAUDE.md                      ← Project instructions
│   ├── SETUP_GUIDE.md
│   ├── requirements.txt
│   └── experiment_config.yaml
│
├── scripts/                           ← Executable scripts
│   ├── run_experiments.ps1            ← Windows execution
│   ├── run_experiments.bat
│   └── validate_setup.py              ← Environment validation
│
├── support/                           ← Supporting materials
│   ├── README.md
│   └── Todo.md
│
└── [existing directories]
    ├── Code/                          ← Code snippets & details
    ├── Examples/                      ← Example implementations
    ├── Literature/                    ← Reference papers (PDF)
    ├── Review/                        ← Comprehensive review
    ├── Sensor Types/                  ← Sensor documentation
    └── results/                       ← Experiment outputs (created at runtime)
```

---

## Key Documents by Phase

### Phase 1: Setup & Planning (Week 0 - COMPLETE)
- ✅ Comprehensive review and critical issue fixes (C1-C5, M1-M3 fixed)
- ✅ Perceiver IO baseline implementation
- ✅ Full 10-week execution plan
- ✅ 6-paper portfolio strategy
- ✅ Publication-quality figure templates (7 templates, 800+ lines of code)

### Phase 2: Week 1 - GPU Experiments
**Status**: Ready to execute
- Run validation: `python scripts/validate_setup.py --verbose`
- Launch experiments: `.\scripts\run_experiments.ps1 -Mode full -Parallel`
- GPUs: 3× Parallel (SensoryPerceiver, Perceiver IO, Ablations)
- Runtime: ~18 hours
- Outputs saved to `results/summary.csv`

### Phase 3: Week 2-4 - Main Paper
- Create publication figures (7 templates)
- Write Methods/Results/Discussion
- Generate comparative analysis
- Submit to NeurIPS (May 31 deadline)

### Phase 4: Week 5-10 - Remaining Papers
- Architecture paper (refactored code)
- Orchestrator paper
- 3 application papers (Robotics, Medical, Smart Home)
- Publish to arXiv as completed

---

## Getting Started (Tomorrow)

```bash
# 1. Validate environment
python scripts/validate_setup.py --verbose

# 2. Review execution plan
cat 01-PLANNING/EXECUTION_READY.md

# 3. Launch experiments
.\scripts\run_experiments.ps1 -Mode full -Parallel

# 4. Monitor GPU execution
watch -n 5 nvidia-smi

# 5. In Week 2, create figures
python figures/accuracy_curves.py
python figures/ablation_results.py
# ... (create all 7 figures)
```

---

## Key Files by Purpose

**If you need to...**

| Task | File |
|------|------|
| Understand the project | `00-START/START_HERE.md` |
| See what's been done | `00-START/SESSION_SUMMARY.md` |
| Plan Week 1 | `01-PLANNING/EXECUTION_READY.md` |
| Write the main paper | `02-PAPER/Paper Overview.md` |
| Create publication figures | `figures/README.md` |
| Run experiments | `scripts/run_experiments.ps1` |
| Check dependencies | `config/requirements.txt` |
| Understand architecture | `02-PAPER/Architecture.md` |
| Submit to NeurIPS | `01-PLANNING/PUBLICATION_STRATEGY.md` |

---

## Status Summary

| Component | Status | Details |
|-----------|--------|---------|
| Paper | ✅ Draft complete | All sections written, critical issues fixed |
| Architecture | ✅ Implemented | SensoryPerceiver + Perceiver IO baselines |
| Planning | ✅ Complete | 10-week timeline, 6-paper portfolio |
| Figures | ✅ Ready | 7 templates, 800 lines of code |
| Code | ✅ Ready | GPU orchestration, baselines, trainers |
| Experiments | 🚀 Ready to go | Validation script ready, GPU setup confirmed |

---

## Project Structure Notes

**Organization Principles**:
- **00-START**: Entry points and status (read first)
- **01-PLANNING**: Strategy and execution (follow these)
- **02-PAPER**: Paper content (write these)
- **03-RESEARCH**: Research planning (reference these)
- **code/**: Python implementations (execute these)
- **config/**: Setup and configuration (configure first)
- **scripts/**: Automation and workflows (run as needed)
- **support/**: Supporting materials (reference as needed)
- **figures/**: Publication-quality visualization templates
- **Legacy dirs**: Code/, Examples/, Literature/, Review/, Sensor Types/ (preserved for reference)

---

## Questions?

- **How do I start?** → `00-START/START_HERE.md`
- **What's the timeline?** → `01-PLANNING/MASTER_EXECUTION_PLAN.md`
- **How do I run experiments?** → `scripts/run_experiments.ps1`
- **How do I create figures?** → `figures/README.md`
- **What's the architecture?** → `02-PAPER/Architecture.md`

---

**Last Updated**: February 19, 2026
**Status**: Organized and ready for Week 1 execution
