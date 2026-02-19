# Sensory Perceiver — Automated Experiment Workflow

Complete workflow for running all experiments on your 3x TitanX GPU Windows machine.

## 🚀 Quick Start (5 minutes)

```powershell
# PowerShell (recommended for Windows)
.\run_experiments.ps1 -Mode quick

# OR batch file
run_experiments.bat quick
```

This will:
1. Generate synthetic IoT data (if not present)
2. Train on synthetic data (10 epochs, ~4 hours)
3. Evaluate and save results
4. Generate `monitor_dashboard.html` for live progress tracking

## 📋 Workflow Modes

### Quick Test (`-Mode quick`)
**Duration**: ~4 hours | **GPU**: 1 (GPU 0)

- Synthetic IoT dataset (generated on-the-fly)
- 10 epochs for fast validation
- Tests full pipeline without long wait

```powershell
.\run_experiments.ps1 -Mode quick
```

**Use case**: First-time setup, debugging, quick validation

---

### Full Benchmark (`-Mode full`)
**Duration**: ~40+ hours (can be parallelized to 14 hours on 3 GPUs) | **GPUs**: 1-3

Runs all 3 datasets × 3 seeds = 9 experiments:
- Synthetic IoT (3 seeds, 50 epochs each)
- Split-CIFAR-100 (3 seeds, 50 epochs each)
- Permuted MNIST (3 seeds, 50 epochs each)

```powershell
# Sequential (uses 1 GPU, ~40 hours)
.\run_experiments.ps1 -Mode full

# Parallel (uses all 3 GPUs, ~14 hours)
.\run_experiments.ps1 -Mode full -Parallel

# Sequential + auto-push to GitHub
.\run_experiments.ps1 -Mode full -Push
```

**Use case**: Final publication results, comprehensive validation

---

### Monitor Dashboard (`-Mode monitor`)
Generate live progress dashboard without running experiments:

```powershell
.\run_experiments.ps1 -Mode monitor
```

Opens `monitor_dashboard.html` showing:
- Experiment status (pending/running/completed/failed)
- Progress bar
- Per-experiment accuracy and runtime
- Auto-refreshes every 30 seconds

---

## 📊 Monitoring

### Browser Dashboard
Open `monitor_dashboard.html` in your browser:
- Live progress tracking
- GPU allocation
- Estimated time remaining
- Per-experiment metrics

Auto-refreshes every 30 seconds.

### Log File
Real-time logs in `experiment_orchestrator.log`:
```bash
tail -f experiment_orchestrator.log  # Unix
Get-Content -Wait experiment_orchestrator.log  # PowerShell
```

### Results JSON
Structured results saved to `results/experiment_log.json`:
```json
[
  {
    "name": "synthetic_iot_seed0",
    "status": "completed",
    "duration_hours": 4.2,
    "result": {
      "overall_accuracy": 0.92,
      "latency_mean": 15.3
    }
  }
]
```

---

## 🔧 Configuration

### Custom Experiment Config
Edit `experiment_orchestrator.py` to customize:

```python
orchestrator = ExperimentOrchestrator(n_gpus=3)

# Queue custom experiments
orchestrator.add_experiment(
    name='custom_exp_1',
    dataset='split_cifar',
    gpu_id=0,
    epochs=100
)
```

### Hyperparameters
Modify defaults in command line:
```powershell
python train.py \
    --d_model 1024 \
    --num_heads 16 \
    --depth 8 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --output_dir results/custom
```

---

## 💾 GPU Memory & Performance

### TitanX (Maxwell/Pascal, 12GB each) Optimization

**Current defaults** (should fit comfortably):
- `--batch_size 32`
- `--d_model 768`
- `--num_heads 8`

**If you get OOM errors:**
```powershell
# Reduce batch size
python train.py --batch_size 16

# Reduce model size
python train.py --d_model 512 --num_heads 6

# Both
python train.py --batch_size 16 --d_model 512
```

**GPU Usage Monitor** (Windows):
```powershell
# PowerShell: Check GPU usage during training
while($true) {
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader
    sleep 5
}
```

---

## 📤 GitHub Integration

### Auto-commit Results
Push results to GitHub after each experiment:

```powershell
.\run_experiments.ps1 -Mode full -Push
```

This will:
1. Run all experiments
2. Auto-commit `results/` + `monitor_dashboard.html` after completion
3. Push to your repo's `main` branch

**Requirements:**
- Git repo initialized (`git init`)
- GitHub remote configured (`git remote add origin ...`)
- Git credentials configured (SSH keys or PAT)

### Manual Push
```powershell
python scripts/push_results.py --message "Add full benchmark results"
```

---

## 🔍 Results Analysis

### Quick Stats
```powershell
# Aggregate results across all runs
python scripts/aggregate_results.py --results_dir results --output_file results/summary.csv

# Generate visualizations
python scripts/plot_results.py --results_dir results --output_dir results/plots
```

### CSV Summary
`results/summary.csv`:
```
experiment,mean_accuracy,std_accuracy,n_runs,seeds
continual_cifar_seed_0,0.9234,0.0012,1,0
continual_mnist_seed_0,0.9122,0.0008,1,0
iot_streaming_seed_0,0.8891,0.0015,1,0
```

### Detailed Results
`results/*/eval_results.json`:
- Per-class accuracy
- Confusion matrix
- Latency statistics
- Embedding projections (if saved)

---

## 🐛 Troubleshooting

### GPU Not Detected
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

python -c "import torch; print(torch.cuda.device_count())"
# Should print: 3 (for 3 TitanX GPUs)
```

**Fix**: Ensure NVIDIA drivers and CUDA toolkit are installed:
```powershell
nvidia-smi  # Should show your GPUs
nvcc --version  # Should show CUDA version
```

### Out of Memory (OOM)
Reduce batch size or model size:
```powershell
python train.py --batch_size 16 --d_model 512
```

### Experiments Stuck
Check logs:
```powershell
Get-Content experiment_orchestrator.log -Tail 50
```

Kill hung Python processes:
```powershell
taskkill /IM python.exe /F
```

### Data Download Issues
Some datasets (AudioSet, COCO) require manual setup. For now:
- Use synthetic IoT data (automatic)
- Use Split-CIFAR-100 (auto-downloads from PyTorch)
- Use Permuted MNIST (auto-downloads from PyTorch)

---

## 📈 Expected Results (Targets from Paper)

### Baseline Comparisons
| Experiment | Target Accuracy | Baseline (Perceiver IO) | Status |
|-----------|-----------------|------------------------|--------|
| Multimodal Classification | ≥95% | 94% | Pending |
| Split-CIFAR-100 (Continual) | ≥85% | 78% | Pending |
| Permuted MNIST (Continual) | ≥82% | 75% | Pending |
| IoT Streaming | ≥88% | 85% | Pending |

*(These will be populated after experiments complete)*

---

## 🎯 Next Steps (After Experiments Complete)

1. **Review Results**
   ```powershell
   # Open monitor dashboard
   monitor_dashboard.html

   # Analyze per-class accuracy
   python scripts/plot_results.py
   ```

2. **Write Results Section**
   - Add actual numbers to paper
   - Include confusion matrices
   - Document any surprises

3. **Run Ablations** (optional but recommended)
   ```powershell
   # Test without replay mechanism
   python train.py --experiment continual ... --no-replay

   # Test different MIME routing strategies
   ```

4. **Prepare for Publication**
   - Finalize results tables
   - Generate supplementary materials
   - Get peer feedback

---

## 💡 Tips & Best Practices

### Keep System Responsive
During full benchmark runs (40+ hours), Windows may slow down:
- Close heavy apps (Chrome, VS Code, Discord, etc.)
- Enable high performance power plan: Settings → System → Power & Battery → Power Mode: Best Performance
- Monitor dashboard uses minimal resources (~5MB memory)

### Data Organization
Results are organized as:
```
results/
├── synthetic_iot_seed0_20260220_120000/
│   ├── best_model.pt          # Best checkpoint
│   ├── latest_model.pt        # Final checkpoint
│   ├── train_log.csv          # Training history
│   ├── eval_results.json      # Evaluation metrics
│   ├── config.yaml            # Hyperparameters
│   └── embeddings.npy         # Learned embeddings (for visualization)
├── split_cifar_seed0_20260220_121500/
└── experiment_log.json        # Overall summary
```

### Reproducibility
All runs are seeded for reproducibility:
- `--seed 42` (default, same results every time)
- Change with `--seed 123`
- Results will vary slightly across GPU types (TitanX vs A100) but should be similar

---

## 📞 Support

If experiments fail, check:
1. `experiment_orchestrator.log` — detailed error messages
2. Individual `results/*/train_log.csv` — training curves
3. GPU memory usage — may need to reduce batch size
4. CUDA/PyTorch compatibility — verify with `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🎓 Citation Format

Once you have results, cite this workflow:
```bibtex
@article{shallbetter2026perception,
  title={Perception Is How We Experience: A Unified Sensory Processing Architecture},
  author={Shallbetter, Zach},
  year={2026},
  note={Experiments run on 3x TitanX GPUs using automated orchestration}
}
```

---

## 📝 Workflow Version
- **Last updated**: Feb 20, 2026
- **Compatible with**: PyTorch 2.1+, Python 3.8+
- **Tested on**: Windows 10/11 with NVIDIA GPU drivers 530+
