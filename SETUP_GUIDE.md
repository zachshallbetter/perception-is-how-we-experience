# Setup Guide: Ready Your System for Automated Experiments

Complete checklist before running experiments on your Windows machine.

## ✅ Pre-Flight Checklist

### 1. Python & Dependencies (15 min)

- [ ] Python 3.8+ installed
  ```powershell
  python --version
  ```
  Should show: `Python 3.x.x`

- [ ] PyTorch with CUDA support
  ```powershell
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- [ ] Project dependencies installed
  ```powershell
  python -m pip install -r requirements.txt
  ```

- [ ] Verify CUDA detection
  ```powershell
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
  ```
  Should show: `CUDA: True` and `GPUs: 3`

### 2. GPU Drivers & Toolkit (30 min if needed)

- [ ] NVIDIA GPU drivers updated
  ```powershell
  nvidia-smi
  ```
  Should show your 3 TitanX GPUs

- [ ] CUDA 11.8 or 12.x installed (if not already)
  - Download from https://developer.nvidia.com/cuda-downloads
  - Choose: Windows, x86_64, Version 12, .exe installer

- [ ] cuDNN 8.x+ installed (optional but recommended)
  - Download from https://developer.nvidia.com/cudnn
  - Extract to `C:\Program Files\NVIDIA\CUDA\v12.x\`

### 3. Git Configuration (5 min, optional)

For auto-push to GitHub:

- [ ] Git installed
  ```powershell
  git --version
  ```

- [ ] Git repo initialized
  ```powershell
  cd <project_directory>
  git init
  git remote add origin https://github.com/<your_username>/<repo_name>.git
  ```

- [ ] GitHub credentials configured
  ```powershell
  # Option 1: SSH keys (recommended)
  git config --global user.name "Zach Shallbetter"
  git config --global user.email "zachshallbetter@icloud.com"

  # Option 2: Personal Access Token (GitHub PAT)
  # Save to: C:\Users\<username>\.git-credentials
  ```

### 4. System Optimization (10 min)

- [ ] High Performance power plan enabled
  - Settings → System → Power & Battery → Power Mode: Best Performance

- [ ] Unnecessary background apps closed
  - Close: Chrome, VS Code, Discord, Slack, OneDrive, etc.
  - Task Manager: Startup tab, disable non-essential apps

- [ ] Disk space available
  ```powershell
  Get-Volume C: | Select-Object SizeRemaining
  ```
  Need at least 50GB free (for results + checkpoints)

- [ ] RAM not maxed out
  ```powershell
  Get-ComputerInfo | Select-Object CsPhyicallyInstalledSystemMemory
  ```
  64GB available as reported; should see ~50GB free initially

### 5. Workspace Setup (5 min)

- [ ] Project directory organized
  ```
  perception-is-how-we-experience/
  ├── train.py
  ├── eval.py
  ├── models.py
  ├── data_loader.py
  ├── experiment_orchestrator.py
  ├── run_experiments.ps1
  ├── requirements.txt
  ├── scripts/
  └── results/  (will be created)
  ```

- [ ] Monitor dashboard location noted
  - Will be created at: `monitor_dashboard.html` (root directory)

- [ ] Results directory created
  ```powershell
  mkdir results
  ```

---

## 🧪 Quick Validation (10 minutes)

**CRITICAL: Run this validation before the full benchmark. It catches setup issues early.**

```powershell
# Comprehensive validation (auto-runs mini training test)
python validate_setup.py --verbose

# If all checks pass, proceed to experiments
# If any checks fail, fix issues and re-run validate_setup.py
```

This script validates:
- ✓ Python 3.8+ installed
- ✓ PyTorch with CUDA support
- ✓ All dependencies installed
- ✓ GPU detection (expects 3 GPUs)
- ✓ Disk space (≥50GB free)
- ✓ Critical files present
- ✓ Mini training test passes (1 epoch on synthetic data)

**Expected output if all pass:**
```
✓ All checks passed! Ready to run experiments.

Next steps:
  1. Run quick test: .\run_experiments.ps1 -Mode quick
  2. Run full benchmark: .\run_experiments.ps1 -Mode full -Parallel
  3. Monitor: Open monitor_dashboard.html in browser
```

**If validation fails:** Follow error messages to fix issues, then re-run `validate_setup.py`

---

## 🚀 Start Experiments

### Quick Test (Safe Start)
```powershell
.\run_experiments.ps1 -Mode quick
```
- Duration: ~4 hours
- Validates full pipeline
- Minimal GPU stress

### Full Benchmark (Production)
```powershell
.\run_experiments.ps1 -Mode full -Parallel
```
- Duration: ~14 hours (parallel on 3 GPUs)
- Duration: ~40 hours (sequential on 1 GPU)
- Generates publication-quality results

### Monitor Progress
While running, open in browser:
- File → `monitor_dashboard.html`
- Auto-refreshes every 30 seconds
- Shows progress, ETA, per-GPU status

---

## 📊 Monitor System Health

Create a PowerShell monitoring script (`monitor_system.ps1`):

```powershell
# Show GPU, CPU, Memory usage every 5 seconds
while($true) {
    Clear-Host
    Write-Host "=== GPU Status ===" -ForegroundColor Cyan
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits | ForEach-Object {
        Write-Host "GPU $_"
    }

    Write-Host "`n=== CPU Status ===" -ForegroundColor Cyan
    $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty LoadPercentage
    Write-Host "CPU: $cpu %"

    Write-Host "`n=== Memory Status ===" -ForegroundColor Cyan
    $mem = Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
    $memFree = Get-CimInstance -ClassName Win32_OperatingSystem | Select-Object -ExpandProperty FreePhysicalMemory
    $memUsed = [math]::Round(($mem - $memFree*1KB) / 1GB, 2)
    $memTotal = [math]::Round($mem / 1GB, 2)
    Write-Host "Memory: $memUsed GB / $memTotal GB"

    Write-Host "`n=== Experiment Status ===" -ForegroundColor Cyan
    $logFile = Get-Content experiment_orchestrator.log -Tail 3
    Write-Host $logFile

    Start-Sleep 5
}
```

Run during experiments:
```powershell
.\monitor_system.ps1
```

---

## 🐛 Troubleshooting Setup

### Python not found
```powershell
# Add Python to PATH
[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Python311", "User")
# Restart PowerShell
```

### CUDA not detected
```powershell
# Check NVIDIA driver
nvidia-smi

# If missing: https://www.nvidia.com/Download/driverDetails.aspx/

# Check PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"
# Should show: 11.8, 12.0, etc.
```

### Out of memory during validation
```powershell
# Use smaller model
python train.py --epochs 1 --batch_size 8 --d_model 512 --depth 4
```

### Git push fails
```powershell
# Test GitHub connection
ssh -T git@github.com

# Or use HTTPS with token
git config credential.helper manager
```

---

## ✨ You're Ready!

Once checklist is complete:

```powershell
# Start quick validation
.\run_experiments.ps1 -Mode quick

# If successful, start full benchmark
.\run_experiments.ps1 -Mode full -Parallel

# Monitor progress
# Open monitor_dashboard.html in browser
```

---

## 📝 Estimated Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Setup & validation | 1 hour | Before experiments |
| Quick test | 4 hours | Recommended first run |
| Full benchmark (parallel) | 14 hours | Main experiment |
| **Total** | **~19 hours** | Spread over 2 days |

---

## 🎯 After Experiments Complete

1. **Review Results**
   - Open `results/summary.csv`
   - Check `monitor_dashboard.html` final stats
   - Review `results/*/eval_results.json`

2. **Generate Visualizations**
   ```powershell
   python scripts/aggregate_results.py --results_dir results
   python scripts/plot_results.py --results_dir results --output_dir results/plots
   ```

3. **Update Paper**
   - Add actual accuracy numbers to Results section
   - Include confusion matrices
   - Add baseline comparisons

4. **Push to GitHub** (if not already done)
   ```powershell
   python scripts/push_results.py --message "Add complete benchmark results"
   ```

---

## 📞 Common Questions

**Q: Can I run multiple experiments in parallel?**
A: Yes! Use `run_experiments.ps1 -Mode full -Parallel` to use all 3 GPUs (14 hours total).

**Q: Will my computer freeze during training?**
A: No, GPU training doesn't freeze the system. You can work normally.

**Q: How much disk space do I need?**
A: ~50GB for full benchmark (model checkpoints + logs).

**Q: Can I stop and resume experiments?**
A: Yes, the orchestrator checks for completed experiments and skips them.

**Q: What if an experiment fails?**
A: Check `experiment_orchestrator.log` for details. The orchestrator will continue with remaining experiments.

---

**You're all set! Run `.\run_experiments.ps1 -Mode quick` to begin.**
