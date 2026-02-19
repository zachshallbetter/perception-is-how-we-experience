# Sensory Perceiver: Multimodal Perception Through MIME-Type Unified Encoding

A scalable neural architecture for multimodal learning with modality-specific feature extraction, cross-modal attention fusion, and continual learning capabilities.

## Quick Links

- 🚀 **[QUICK START](QUICK_START.md)** - Get up and running in 5 minutes
- 📈 **[PROGRESS](PROGRESS.md)** - Current status and timeline
- 📋 **[PLANNING](planning/EXECUTION_READY.md)** - Executive summary and phase timeline

## Directory Structure

```
perception-is-how-we-experience/
├── main-paper/              ← Primary paper and figures
├── src/                     ← Production code (models, training, evaluation)
├── baselines/               ← Comparison implementations (Perceiver IO)
├── planning/                ← Strategy docs, execution plans, timelines
├── research/                ← Analysis frameworks, paper outlines, validation
├── reference/               ← Legacy code, examples, literature
├── config/                  ← Configuration files
├── scripts/                 ← Utility scripts for experiments and setup
├── docs/                    ← Documentation (architecture, guides, FAQs)
└── README.md, QUICK_START.md, PROGRESS.md
```

## Project Status: 85% Ready for Publication

✅ **Completed (Phases 1-3)**:
- Core architecture formalized (MIME-Type encoding, PerceiverTower modules)
- Mathematical formulations added
- 14 foundational citations integrated
- Code-paper alignment validated
- All internal contradictions resolved

⏳ **Pending (Phase 4: M1 - Validation Experiments)**:
- Run GPU experiments on multimodal benchmarks
- Generate quantitative validation results
- Complete papers and figures

## Architecture Overview

**Sensory Perceiver** combines:
1. **MIME-Type Unified Sensory Encoding**: Modality-specific feature extraction with explicit modality tokens
2. **PerceiverTower Modules**: Lightweight towers for each sensory modality
3. **Cross-Modal Attention Fusion**: Information exchange across modality boundaries
4. **Experience Replay**: Mitigation of catastrophic forgetting in continual learning

## Key Files

### Core Implementation
- `src/models.py` - SensoryPerceiver and PerceiverTower architecture
- `src/train.py` - Training pipeline with continual learning support
- `src/eval.py` - Evaluation metrics and analysis
- `src/experiments/experiment_orchestrator.py` - Multi-GPU coordination

### Paper & Documentation
- `main-paper/Perception Is How We Experience.md` - Main paper (102K words)
- `main-paper/Paper Overview.md` - High-level summary
- `planning/EXECUTION_READY.md` - Quick reference for Phase 4
- `research/FIGURE_TEMPLATES.md` - 7 publication-quality figure templates

### Configuration
- `config/experiment_config.yaml` - Hyperparameters
- `config/requirements.txt` - Python dependencies
- `config/SETUP_GUIDE.md` - Environment setup

## Publication Timeline

**Week 0 (This week)**: Directory reorganization ← **YOU ARE HERE**
**Week 1**: GPU experiments + metric validation (18 GPU hours + 30 human hours)
**Week 2-3**: Paper finalization + figure generation
**Week 4**: Submission to NeurIPS/ICLR

## Next Steps

1. **Verify new structure**: Check all imports work with new paths
2. **Run M1 experiments**: Execute GPU validation experiments
3. **Generate results**: Compute metrics and create figures
4. **Finalize paper**: Write results section, add findings to discussion

For detailed information, see [QUICK_START.md](QUICK_START.md) and [PROGRESS.md](PROGRESS.md).

---

**Project Owner**: Zach (zachshallbetter@icloud.com)
**Last Updated**: February 19, 2026
**Status**: 85% publication-ready (awaiting M1 validation experiments)
