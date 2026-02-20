# Training Guide: Sensory Perceiver

This document outlines the standard training procedures, multi-GPU orchestration, and experimental workflows for the Sensory Perceiver architecture.

## 1. Core Training Scripts

The project utilizes several distinct training tracks depending on the experimental goal:

### A. Continual Learning Benchmark (`scripts/run_local.py`)
This is the **primary continual learning script** (superseding purely synthetic testing in `src/train.py`). It executes Split-CIFAR-100 across 10 sequential tasks to test the replay mechanisms.

```bash
# Run Split-CIFAR-100 benchmark evaluating BWT and FWT
# Defaults to feature replay (--replay_mode feature)
python scripts/run_local.py --batch_size 32 --d_model 768 --num_heads 8

# Run with the heavily tuned contrastive replay mechanism
python scripts/run_local.py --tuned --replay_mode contrastive
```
It outputs Task-Incremental metrics, measuring absolute Catastrophic Forgetting, Forward Transfer (FWT), and Backward Transfer (BWT).

### B. Standard Multimodal Training (`src/train.py`)
Used for generic, single-task multimodal learning across vision, IoT streaming, or medical benchmarks.
```bash
python src/train.py --experiment multimodal --d_model 768 --output_dir results/multimodal
```

### C. Baselines (`train_perceiver_io.py`)
Used to run identical benchmarks using the Perceiver IO architecture to establish baseline accuracy and computing efficiency.
```bash
python train_perceiver_io.py --experiment continual --dataset split_cifar --output_dir results/baseline
```

---

## 2. Multi-GPU Benchmarking Infrastructure (Primary)

For complete results generation (9 model seeds + ablations), execution is parallelized across dedicated CUDA hardware (optimized for 3x 12GB TitanX GPUs or similar compute clusters).

### Orchestration Scripts
The primary benchmarking suite is located in the `scripts/` directory and managed via Python and shell/PowerShell automation.

**Windows/PowerShell Orchestration:**
```powershell
# Quick Validation (4 hours, GPU 0 only)
.\scripts\run_experiments.ps1 -Mode quick

# Full Parallel Benchmark (14-18 hours, 3 GPUs)
.\scripts\run_experiments.ps1 -Mode full -Parallel

# Full Benchmark + Auto-push to GitHub
.\scripts\run_experiments.ps1 -Mode full -Push
```

**Unix/Linux Orchestration:**
```bash
# Run the sequential benchmark bash script
bash scripts/run_full_benchmark.sh
```

### Allocation Strategy (via `experiment_orchestrator.py`)
When running parallel full modes:
- **GPU 0 (`SensoryPerceiver` Primary)**: Runs benchmarks for CIFAR, Permuted MNIST, and IoT data (expected 14 hours).
- **GPU 1 (`Perceiver IO` Baseline)**: Runs baseline comparisons matching GPU 0 data (expected 13 hours).
- **GPU 2 (Ablations)**: Runs all ablations defined in `ablation_study_config.py` (e.g., without replay, without MIME-Type routing) (expected 18 hours).

---

## 3. Local Soft Testing (Apple M1/MPS)

For local development, debugging, and soft testing, the `scripts/run_local.py` script is optimized to run natively on Apple Silicon (M1/M2/M3) using PyTorch's MPS (Metal Performance Shaders) backend, though it works equally well on CUDA/CPU.

### Running Local Tests
```bash
# Run the Split-CIFAR-100 benchmark locally to verify changes
python scripts/run_local.py --batch_size 32 --d_model 768 --num_heads 8
```
*Note: This is strictly for local code validation. Please use the Multi-GPU Orchestration scripts for generating final paper metrics.*

---

## 4. Cloud & Distributed Computing

For production-scale training and benchmarks beyond local multi-GPU setups, the orchestration scripts can be adapted for Cloud infrastructure (AWS/GCP/Lambda Labs).

### Cloud Provisioning Recommendations
For full benchmark replication (~40 hours compute equivalent):
- **AWS Recommendations**: `p3.8xlarge` (4x V100 16GB) or `p4d.24xlarge` (8x A100 40GB).
- **GCP Recommendations**: `a2-highgpu-4g` (4x A100 40GB) or `n1-standard` instances with attached T4/V100 GPUs.
- **Docker/Containerization**: To ensure consistent dependencies across cloud instances, package the repository in a PyTorch CUDA base image (e.g., `nvcr.io/nvidia/pytorch:23.10-py3`).

### Distributed Execution
When managing multiple cloud instances rather than a single multi-GPU machine:
1. Divide the `ablation_study_config.py` runs across worker nodes.
2. Update `--device` mapping if your cloud instance has a different GPU topology than the assumed 3 GPUs.
3. Use the built-in `--push-to-github` flag on the orchestration scripts to continuously sync results back to the central repository from ephemeral cloud instances:

```bash
# Example for a disposable cloud instance
bash scripts/run_full_benchmark.sh && python scripts/push_results.py --message "Cloud benchmark complete"
```

---

## 5. Ablation Studies

Ablation studies isolate the impact of the Sensory Perceiver's core innovations. They are executed using:
```bash
python src/train.py --config configs/ablation_study_config.py --ablation [ABLATION_NAME]
```

**Key Ablations:**
1. **`ablation_1_without_replay`**: Disables the contrastive replay buffer to measure baseline catastrophic forgetting.
2. **`ablation_2_without_mime`**: Replaces MIME-Type modality routing with a generic embedding layer.
3. **`ablation_3_efficiency`**: Scales model parameters down to 100M to measure accuracy vs. efficiency trade-offs.

---

## 6. Monitoring & Results

- **Live Dashboard**: Open `monitor_dashboard.html` in your browser during `scripts/run_experiments.ps1` to view live metric updates and estimated time of arrival (ETA).
- **Logging**: Console output is piped to `experiment_orchestrator.log`.
- **Artifacts**: All trained checkpoints (`best_model.pt`), evaluation metrics (`eval_results.json`), and hyperparameter configs are saved automatically to the `results/` directory under a timestamped folder.

### Aggregating Results
Once runs are complete, compile the CSV summaries and generate Matplotlib figures:
```bash
python scripts/aggregate_results.py --results_dir results
python scripts/plot_results.py --results_dir results --output_dir results/plots
```

---

## 7. Troubleshooting Apple Silicon & CUDA Issues

**CUDA Out of Memory (OOM)** on 12GB TitanX GPUs:
- Reduce memory footprint by halving `--batch_size` (e.g., from 32 to 16).
- If still OOM, reduce `--d_model` (e.g., from 768 to 512) and `--num_heads` (e.g., from 8 to 4).

**Apple Unified Memory OOM**:
- Unlike dedicated VRAM, PyTorch MPS shares memory with the system. Close heavy applications (like browsers) to free up RAM.
- Set the environment variable to limit MPS memory allocation if it crashes the OS:
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  ```

If processes hang or block memory:
```bash
# Verify process using activity monitor or kill python
pkill -f python
# On Windows
taskkill /IM python.exe /F 
```
