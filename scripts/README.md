# Utility Scripts for Sensory Perceiver Experiments

## Dataset Preparation

### `generate_sensor_streaming_data.py`
Generate IoT sensor-stream benchmark data for validation (no external dependencies).

```bash
python scripts/generate_sensor_streaming_data.py \
  --n_sequences 100 \
  --sequence_length 500 \
  --output_dir data/sensor-streaming
```

**Output**: Multimodal sensor data (smartwatch, environmental, robot arm)

### `download_audioset.py` (user implementation)
Download and preprocess AudioSet (22K balanced subset).
- Requires: `yt-dlp`, `librosa`
- Converts to 16kHz mono WAV files
- User should implement using AudioSet's official download instructions

### `preprocess_coco.py` (user implementation)
Preprocess MSCOCO 2017 validation split (5K images).
- Download from: https://cocodataset.org/
- Expected structure: `data/coco/images/val2017/` + annotations JSON

### `generate_av_mnist.py` (user implementation)
Generate audio-visual MNIST pairs (5K samples).
- Combines visual digits with generated speech

### `preprocess_flickr.py` (user implementation)
Preprocess Flickr30K (31K image-text pairs).
- Download from: https://www.flickr30k.org/
- Extract images and captions to parallel files

### `generate_split_cifar.py` (user implementation)
Generate Split-CIFAR-100 (20 sequential tasks, 5 classes each).
- Automatically downloads CIFAR-100
- Splits into task-specific subsets

### `generate_permuted_mnist.py` (user implementation)
Generate Permuted MNIST (20 tasks with random pixel permutations).
- Automatically downloads MNIST
- Applies task-specific permutations

## Training & Evaluation

### `src.train` (main training loop)
Train Sensory Perceiver on any experiment type.

```bash
python -m src.train \
  --experiment continual \
  --dataset split_cifar \
  --batch_size 32 \
  --epochs 50 \
  --output_dir results/my_experiment
```

**Outputs**:
- `best_model.pt`: Best validation checkpoint
- `latest_model.pt`: Final model
- `train_log.csv`: Training history
- `config.yaml`: Experiment configuration

### `src.eval`
Evaluate trained model on validation set.

```bash
python -m src.eval \
  --checkpoint results/my_experiment/best_model.pt \
  --experiment continual \
  --dataset split_cifar \
  --output_file results/my_experiment/eval_results.json
```

**Outputs**:
- `eval_results.json`: Per-class accuracy, confusion matrix, latency
- `embeddings.npy`: Per-modality embeddings (for visualization)

## Benchmarking

### `run_replay_depth_ablation.py`
Run replay-depth ablations across three conditions:
- `feature_only` (`--replay_mode feature`)
- `raw_er` (`--replay_mode er`)
- `raw_contrastive` (`--replay_mode contrastive`)

```bash
python scripts/run_replay_depth_ablation.py \
  --experiment continual \
  --dataset split_cifar \
  --seeds 42,43,44 \
  --epochs 5 \
  --device cpu
```

**Outputs**:
- `ablation_runs.json`: per-run records + aggregate stats
- `ablation_runs.csv`: flat run table
- `ablation_summary.csv`: condition-level mean/std metrics

### `report_replay_depth_ablation.py`
Generate a Markdown report from a completed replay-depth ablation run.

```bash
python scripts/report_replay_depth_ablation.py \
  --ablation_dir results/replay_depth_ablation/run_YYYYMMDD_HHMMSS
```

**Output**:
- `ablation_report.md` in the selected ablation directory

### `run_full_benchmark.sh`
Execute all 3 experiments × 3 seeds with evaluation.

```bash
bash scripts/run_full_benchmark.sh
# Outputs: results/benchmark_YYYYMMDD_HHMMSS/
```

**Runs** (parallel-friendly):
1. Continual Learning: Split-CIFAR-100 (3 seeds)
2. Continual Learning: Permuted MNIST (3 seeds)
3. IoT Streaming: Generated sensor data (3 seeds)
4. Aggregation: Compile statistics

**Duration**: ~6-12 hours on A100 (2h per seed × 3 seeds × 2 datasets)

## Analysis & Visualization

### `aggregate_results.py`
Aggregate results across multiple experiment runs.

```bash
python scripts/aggregate_results.py \
  --results_dir results/benchmark_20260219_120000 \
  --output_file results/benchmark_20260219_120000/summary.csv
```

**Outputs**:
- `summary.csv`: Mean ± std accuracy per experiment
- `aggregate_stats.json`: Full statistics (latency, per-seed breakdowns)

### `plot_results.py`
Generate visualization plots from evaluation results.

```bash
python scripts/plot_results.py \
  --results_dir results/benchmark_20260219_120000 \
  --output_dir results/benchmark_20260219_120000/plots
```

**Plots Generated**:
- `confusion_matrix.png`: Heatmap of prediction errors
- `per_class_accuracy.png`: Bar chart by class
- `training_curves.png`: Loss & accuracy over epochs
- `embeddings_tsne.png`: t-SNE projection of learned representations

## Workflow Example

```bash
# 1. Generate IoT benchmark data
python scripts/generate_sensor_streaming_data.py --output_dir data/sensor-streaming

# 2. Train on IoT experiment
python -m src.train \
  --experiment sensor_streaming \
  --batch_size 32 \
  --epochs 50 \
  --output_dir results/sensor_streaming_seed0

# 3. Evaluate
python -m src.eval \
  --checkpoint results/sensor_streaming_seed0/best_model.pt \
  --experiment sensor_streaming \
  --output_file results/sensor_streaming_seed0/eval_results.json

# 4. Visualize
python scripts/plot_results.py \
  --results_dir results/sensor_streaming_seed0 \
  --output_dir results/sensor_streaming_seed0/plots
```

## Notes

- **External datasets** (AudioSet, COCO, Flickr30K): Users must implement download scripts or use standard dataset repositories
- **Generated data**: Use `generate_sensor_streaming_data.py` for quick validation without external dependencies
- **GPU memory**: Reduce `--batch_size` or `--d_model` if OOM errors occur
- **Reproducibility**: All scripts accept `--seed` parameter for deterministic runs
