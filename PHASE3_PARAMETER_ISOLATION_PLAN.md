# Phase 3: Parameter Isolation + Raw-Pixel Replay

**Objective**: Combine Phase 2 robust raw-pixel replay with task-specific parameter isolation to achieve SOTA-competitive continual learning performance.

**Expected Outcomes**: CIL 30–35% (vs. 19.54% Phase 2), BWT −25% (vs. −34.91% Phase 2), Task 0 retention 40–50%

**Timeline**: 2–3 weeks

---

## Overview: Why Phase 3?

Phase 2 (19.54% CIL) achieved meaningful improvement over Phase 1 (4.86%) by ensuring gradient signals flow to PerceiverTower feature extractors via raw-pixel replay. However, the towers remain **task-coupled**: weight updates from new tasks still displace old-task information.

Phase 3 adds **structural isolation**: task-specific adapters become frozen after each task, preventing any weight changes on past tasks once they're learned. The towers remain task-agnostic and shared (with replay stabilization from Phase 2), while task-specific output mappings are protected via freezing.

**Hypothesis**: Combining replay-based stability (Phase 2) with structural isolation (Phase 3) should achieve ~30–35% CIL, competitive with EWC and GEM baselines.

---

## Architecture: Phase 2 + Task Adapters

### Core Components

```
Input (raw pixels) → PerceiverTowers (Conv2d + MHA)
                  ↓ (shared, updated every task with replay)
            Cross-Modal Attention
                  ↓
            TaskAdapter_t (1 per task, frozen after training)
                  ↓
            Classification Head (updated per task)
                  ↓
            Output (class logits)
```

**Key differences from Phase 1**:
- Phase 1: TaskAdapter exists but only adapts for new tasks (isolation not enforced)
- Phase 3: TaskAdapter frozen immediately after task training, new task gets fresh adapter

### TaskAdapter Implementation

Each TaskAdapter is a 2-layer MLP residual connection:

```python
class TaskAdapter(nn.Module):
    def __init__(self, d_model=768, d_hidden=512, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return x + self.mlp(self.ln(x))  # Residual connection

class SensoryPerceiverWithIsolation(nn.Module):
    def __init__(self, num_tasks=10, **tower_kwargs):
        super().__init__()
        self.towers = build_modality_towers(**tower_kwargs)
        self.cross_attention = CrossModalAttention()

        # Task-specific adapters (one per task)
        self.adapters = nn.ModuleDict({
            f'task_{t}': TaskAdapter() for t in range(num_tasks)
        })

        self.current_task = 0
        self.frozen_tasks = set()  # Track which adapters are frozen

    def freeze_task(self, task_id):
        """Freeze adapter for completed task."""
        for param in self.adapters[f'task_{task_id}'].parameters():
            param.requires_grad = False
        self.frozen_tasks.add(task_id)

    def activate_task(self, task_id):
        """Activate adapter for new task."""
        self.current_task = task_id
        for param in self.adapters[f'task_{task_id}'].parameters():
            param.requires_grad = True

    def forward(self, x, task_id=None):
        task_id = task_id or self.current_task

        # Shared processing (updated with replay every task)
        z = self.towers(x)  # Modality-specific extraction
        attended = self.cross_attention(z)  # Cross-modal integration

        # Task-specific adaptation (frozen after task training)
        adapted = self.adapters[f'task_{task_id}'](attended)

        # Task-specific classification
        logits = self.task_heads[f'task_{task_id}'](adapted)
        return logits
```

---

## Training Procedure

### Phase 3 Training Loop

```python
def train_phase3_with_isolation_and_replay(
    model, tasks, num_tasks=10, buffer_size=5000,
    replay_weight=0.5, lr=1e-3, epochs_per_task=5
):
    """
    Train with both Phase 2 replay and Phase 3 isolation.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    replay_buffer = ReservoirSamplingBuffer(buffer_size)

    for task_id in range(num_tasks):
        # Activate fresh adapter for this task
        model.activate_task(task_id)

        # Initialize new classification head
        model.add_task_head(task_id)

        # Get trainable parameters (current adapter + towers + head)
        trainable_params = list(model.towers.parameters()) + \
                          list(model.adapters[f'task_{task_id}'].parameters()) + \
                          list(model.task_heads[f'task_{task_id}'].parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        print(f"\n=== Task {task_id} ===")
        print(f"Trainable adapters: 1 (frozen: {len(model.frozen_tasks)})")
        print(f"Frozen adapters: {model.frozen_tasks}")

        for epoch in range(epochs_per_task):
            # Phase 3a: Learn current task
            for batch_idx, (X_current, y_current) in enumerate(tasks[task_id]):
                logits = model(X_current, task_id=task_id)
                ce_loss = F.cross_entropy(logits, y_current)

                # Phase 3b: Replay old tasks
                if len(replay_buffer) > 0:
                    X_replay = replay_buffer.sample_batch(batch_size=32)
                    z_current = model.towers(X_current)
                    z_replay = model.towers(X_replay)

                    # NT-Xent on tower outputs (Phase 2 mechanism)
                    replay_loss = nt_xent_loss(z_current, z_replay, tau=0.07)
                else:
                    replay_loss = 0

                # Combined loss
                total_loss = (1 - replay_weight) * ce_loss + \
                            replay_weight * replay_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | " +
                          f"CE: {ce_loss:.4f} | Replay: {replay_loss:.4f}")

            # Add Task t data to replay buffer
            for X, y in tasks[task_id]:
                replay_buffer.add_batch(X, task_id)

        # CRITICAL: Freeze this task's adapter
        model.freeze_task(task_id)
        print(f"✓ Froze adapter for Task {task_id}")

        # Evaluate on all tasks seen so far
        cil, bwt = evaluate_continual_learning(model, tasks[:task_id+1])
        print(f"CIL: {cil:.2%} | BWT: {bwt:.2%}")
```

---

## Key Hyperparameters

| Parameter | Phase 2 | Phase 3 | Rationale |
|-----------|---------|---------|-----------|
| Learning rate | 1e-3 | 1e-3 | Same as Phase 2 baseline |
| Replay weight | 0.5 | 0.5 | Balance new task + old task gradients |
| Buffer size | 5000 | 5000 | Maintain task-fair representation |
| Adapter hidden dim | — | 512 | ~256K parameters per adapter |
| Epochs per task | 5 | 5 | Standard for Split-CIFAR-100 |

---

## Implementation Checklist

### Week 1: Implementation

- [ ] **Task 1.1**: Extend `SensoryPerceiverWithIsolation` class with adapter freezing
  - Add `freeze_task()` and `activate_task()` methods
  - Implement per-task classification heads
  - Ensure gradients don't flow to frozen adapters

- [ ] **Task 1.2**: Verify gradient flow in training loop
  - Add assertions checking `requires_grad=True` for current adapter, `False` for frozen
  - Print parameter counts at each task: trainable vs. frozen

- [ ] **Task 1.3**: Run Phase 3 training on Split-CIFAR-100 (5 seeds)
  - Use seed 42 as primary (compare to Phase 2)
  - Seeds 123, 456, 789, 999 for confidence intervals
  - Track wall time and memory usage

- [ ] **Task 1.4**: Logging and checkpointing
  - Save model at each task boundary
  - Log CIL/BWT/task-retention metrics after each task
  - Save confusion matrix (10×10 grid) for final analysis

### Week 2: Benchmarking & Comparison

- [ ] **Task 2.1**: Baseline comparison
  - Run PackNet (parameter isolation only, no replay)
  - Run EWC (regularization, no replay)
  - Run Simple ER (replay only, no isolation)
  - Compare to Phase 3 results

- [ ] **Task 2.2**: Ablation: Isolation vs. Replay
  - Run Phase 3 without freezing (replay only) → expect ~19% CIL
  - Run Phase 3 without replay (isolation only) → expect ~15% CIL
  - Run Phase 3 with both → expect ~30–35% CIL

- [ ] **Task 2.3**: Hyperparameter sensitivity
  - Vary adapter_hidden: 256, 512, 1024
  - Vary replay_weight: 0.3, 0.5, 0.7
  - Vary buffer_size: 2500, 5000, 10000
  - Report performance vs. hyperparameters

- [ ] **Task 2.4**: Task-level analysis
  - Focus on Task 9 performance (Phase 2 bottleneck)
  - Does isolation help Task 9 learn better?
  - Report Task 9 CIL with Phase 3 vs. Phase 2

### Week 3: Finalization & Paper Update

- [ ] **Task 3.1**: Generate figures
  - Figure 7: Confusion matrix (Phase 2 vs Phase 3)
  - Figure 8: BWT improvement over baselines (error bars)
  - Figure 9: Hyperparameter sensitivity (heatmaps)

- [ ] **Task 3.2**: Write Results §6.3 (Phase 3 empirical results)
  - Update paper with Phase 3 findings
  - Add 5-seed confidence intervals
  - Compare to EWC/GEM baselines

- [ ] **Task 3.3**: Update Discussion §7.9
  - Move "Parameter Isolation + Replay" from future to completed
  - Discuss Phase 3 insights (e.g., does adapter freezing help Task 9?)
  - List Phase 4 (multimodal validation) as next priority

---

## Expected Results & Success Criteria

### Minimum Success (Phase 3 working)
- CIL ≥ 25% (vs. 19.54% Phase 2)
- BWT ≥ −30% (vs. −34.91% Phase 2)
- Task 0 retention ≥ 35% (vs. 33.0% Phase 2)
- Task 9 accuracy ≥ 5% (vs. 3.0% Phase 2)

### Target Success (SOTA-competitive)
- CIL 30–35% (match EWC/GEM range)
- BWT −20% to −25% (substantial forgetting reduction)
- Task 0 retention ≥ 40%
- Task 9 accuracy ≥ 8%

### Stretch Goals (Novel contribution)
- CIL ≥ 35% (approach PackNet range)
- BWT ≥ −20% (better than GEM)
- Demonstrate first architecture combining raw-pixel replay + parameter isolation

---

## Fallback Plans

If Phase 3 doesn't reach target (CIL < 30%):

### Option A: Logit Replay (higher priority)
Add output logits to replay buffer. Store $(X, y, \text{logits}_t)$ where logits are soft targets from original task learning. During replay, use MSE loss on logits alongside NT-Xent on embeddings.

**Expected boost**: +3–5% CIL.

### Option B: Task-Specific Learning Rates
Higher LR for later tasks to overcome weight inertia. Implement scheduler:
```
lr(task_t) = base_lr * (1.0 + 0.1 * t)
```
Helps Task 9 convergence.

**Expected boost**: +2–3% CIL.

### Option C: Larger Buffer + More Epochs
Increase buffer to 10000 and epochs to 10 per task. Trade computation for accuracy.

**Expected boost**: +2–4% CIL, ~2x training time.

---

## Integration with Paper

Once Phase 3 completes, paper updates:

1. **Abstract**: Update with Phase 3 results (CIL 30–35%)
2. **Methods §4.5**: Move from "Phase 3 — Future" to "Phase 3 — Implemented & Validated"
3. **Results §6.3**: New subsection with Phase 3 empirical results
4. **Results §6.4**: Summary table comparing Phase 1, 2, 3, and baselines
5. **Discussion §7.1**: Discuss how isolation + replay work synergistically
6. **Discussion §7.9**: Move parameter isolation to completed work

---

## Code Preparation Status

**Already exists in codebase**:
- `src/models.py`: `TaskAdapter` class implemented (lines 45–78)
- `scripts/run_local.py`: Reservoir Sampling buffer implemented
- `test_*.py`: Test suite for parameter isolation (not yet executed)

**Action items before Phase 3 starts**:
- [ ] Run existing tests: `pytest test_parameter_isolation.py`
- [ ] Verify `freeze_task()` prevents gradient flow
- [ ] Verify confusion matrix is generated correctly

---

## Timeline & Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Implementation | Phase 3 training script ready + seed 42 run complete |
| 2 | Benchmarking | 5-seed Phase 3 complete + baseline comparisons ready |
| 3 | Analysis & Paper | Results written into paper + figures generated |

**Target completion**: End of Week 3 (10 days from start)

