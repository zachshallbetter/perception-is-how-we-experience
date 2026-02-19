# Sensory Perceiver: Architecture Overview

## System Design

Sensory Perceiver implements a unified multimodal learning architecture combining:

### 1. MIME-Type Unified Sensory Encoding
Explicit modality identification and routing:
- One-hot modality tokens for each sensor type
- Enables learned modality-specific processing
- Allows information flow between modalities

### 2. PerceiverTower Modules
Lightweight modality-specific feature extractors:
- Conv1D blocks for initial feature extraction
- Learnable positional encodings
- Layer normalization and dropout
- Output: Fixed-size hidden representations

### 3. Cross-Modal Attention Fusion
Information integration across modality boundaries:
- Concatenate modality representations
- Compute attention over all modalities
- Learn which modalities to combine

### 4. Experience Replay
Continual learning without catastrophic forgetting:
- Maintain replay buffer of past experiences
- Sample during training on new tasks
- Inspired by continual learning literature (Lopez-Paz 2017)

## Code Organization

```
src/
├── models.py              ← Architecture definitions
├── train.py              ← Training pipeline
├── eval.py               ← Evaluation metrics
├── data/loaders.py       ← Data loading
├── experiments/
│   ├── orchestrator.py   ← Multi-GPU coordination
│   └── configs.py        ← Experiment configurations
└── utils/                ← Metric computation, visualization
```

## Key Design Decisions

1. **Modality-Specific Towers**: Inductive bias toward learning modality-specific representations before fusion
2. **Cross-Modal Attention**: Learned mechanism for information integration (not fixed rules)
3. **Experience Replay**: Simple yet effective approach to continual learning
4. **Explicit Modality Tokens**: Enables model to route information based on sensor type

## Mathematical Formulation

See `main-paper/sections/04-methods.md` for detailed mathematical specifications.

## Performance Characteristics

- **Parameters**: ~5M baseline (scales with modality count)
- **Throughput**: 1000+ samples/sec on TitanX
- **Memory**: ~8GB for batch size 32
- **Scalability**: Linear with number of modalities

## Comparison with Baselines

| Aspect | Perceiver IO | Sensory Perceiver |
|--------|-------------|-------------------|
| Modality handling | Symmetric | Explicit towers |
| Fusion mechanism | Global attention | Cross-modal attention |
| Continual learning | None | Experience replay |
| Extensibility | To new modalities | Easy (add tower) |

For detailed comparison, see `planning/MULTI_PAPER_STRATEGY.md`.
