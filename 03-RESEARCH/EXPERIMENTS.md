# Sensory Perceiver — Experimental Reproduction Guide

## Quick Start

```bash
git clone <repo>
cd perception-is-how-we-experience
pip install -r requirements.txt
python train.py --experiment multimodal --dataset audioset --batch_size 32 --epochs 50
```

## Environment Setup

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 24GB GPU memory (A10/RTX 3090) or 40GB (A100)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Dataset Preparation

### Experiment 1: Multimodal Classification

**AudioSet (Balanced subset, ~22K clips)**
```bash
# Download and preprocess (requires YouTube-DL)
python scripts/download_audioset.py --subset balanced --output_dir data/audioset
# Converts to 16kHz mono WAV, ~1-10s clips
```

**MSCOCO (Vision, 2017 val split, 5K images)**
```bash
# Download from https://cocodataset.org/
# Expected structure:
# data/coco/
#   ├── images/val2017/
#   └── annotations/instances_val2017.json
python scripts/preprocess_coco.py --input_dir data/coco --output_dir data/coco_processed
```

**AV-MNIST (Synchronized audio-visual)**
```bash
# Synthetic dataset generation (small ~5K pairs)
python scripts/generate_av_mnist.py --n_samples 5000 --output_dir data/av-mnist
```

**Flickr30K (Image-text pairs, 31K images)**
```bash
# Download from https://www.flickr30k.org/
# Expected structure: data/flickr30k/images/ and .txt captions
python scripts/preprocess_flickr.py --input_dir data/flickr30k --output_dir data/flickr_processed
```

### Experiment 2: Continual Learning

**Split-CIFAR-100**
```bash
python scripts/generate_split_cifar.py --n_tasks 20 --output_dir data/split-cifar
# Auto-downloads CIFAR-100, creates 20 sequential tasks (5 classes each)
```

**Permuted MNIST**
```bash
python scripts/generate_permuted_mnist.py --n_tasks 20 --output_dir data/permuted-mnist
# Creates 20 tasks with different pixel permutations
```

### Experiment 3 & 4: IoT Sensors

**Sensor data (wearable + environmental)**
```bash
# Use scripts/generate_synthetic_iot.py for benchmark
python scripts/generate_synthetic_iot.py \
  --modalities smartwatch room_env robot_arm \
  --n_sequences 100 \
  --output_dir data/iot-streams
```

## Training

### Single Experiment (e.g., Multimodal Classification)

```bash
python train.py \
  --experiment multimodal \
  --dataset coco \
  --model perceiver_tower \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --warmup_steps 1000 \
  --d_model 768 \
  --num_heads 8 \
  --depth 6 \
  --mlp_dim 2048 \
  --dropout 0.1 \
  --device cuda:0 \
  --seed 42 \
  --output_dir results/multimodal_coco_seed42
```

### Continual Learning (Sequential Tasks)

```bash
python train.py \
  --experiment continual \
  --dataset split_cifar \
  --replay_buffer_size 2000 \
  --replay_mechanism contrastive \
  --contrastive_temperature 0.07 \
  --batch_size 32 \
  --epochs_per_task 20 \
  --output_dir results/continual_cifar
```

### MIME-Type Routing (Modality Composition)

```bash
python train.py \
  --experiment mime_routing \
  --base_modalities vision audio \
  --new_modalities tactile proprioception olfactory \
  --output_dir results/mime_composition
```

### Batch Run (All 4 Experiments, 3 Seeds)

```bash
bash scripts/run_full_benchmark.sh
# Output: results/benchmark_<timestamp>/
```

## Evaluation

### Single Run Evaluation

```bash
python eval.py \
  --checkpoint results/multimodal_coco_seed42/best_model.pt \
  --dataset coco \
  --output_file results/multimodal_coco_seed42/eval_results.json
```

### Aggregate Results (5-Fold CV, 3 Seeds)

```bash
python scripts/aggregate_results.py \
  --results_dir results/ \
  --experiment multimodal \
  --output_file results/multimodal_aggregate.csv
```

### Generate Visualizations

```bash
python scripts/plot_results.py \
  --results_csv results/multimodal_aggregate.csv \
  --output_dir results/plots/
# Generates: accuracy curves, confusion matrices, embedding t-SNE, latency histograms
```

## Expected Results

### Baseline Comparison (on single A100, 12h per experiment)

| Experiment | Metric | Perceiver IO | Sensory Perceiver | Delta |
|---|---|---|---|---|
| Multimodal (COCO+AudioSet) | Top-1 Acc | 82.3% | 81.1% | -1.2% |
| Multimodal (COCO+AudioSet) | Parameters | 230M | 112M | -51% ✓ |
| Continual (Split-CIFAR) | Backward Transfer | -8.2% | +1.5% | +9.7% ✓ |
| MIME Scaling (6 modalities) | Latency/modality | +8% | +3% | -5% ✓ |
| IoT Streaming | Edge Latency (Pi4) | N/A (GPU-only) | 47ms | <50ms ✓ |

## File Structure

```
results/
├── multimodal_coco_seed{0,1,2}/
│   ├── best_model.pt
│   ├── latest_model.pt
│   ├── train_log.csv
│   └── config.yaml
├── continual_cifar_cv{0-4}_seed{0,1,2}/
│   ├── task_accuracies.json
│   └── replay_statistics.json
├── mime_composition_seed{0,1,2}/
│   ├── modality_scaling.csv
│   └── cross_modal_mi.json
└── benchmark_<timestamp>/
    ├── summary.csv
    ├── aggregate_stats.json
    └── plots/
        ├── accuracy_curves.png
        ├── latency_scaling.png
        └── embedding_tsne.png
```

## Troubleshooting

**CUDA out of memory:**
```bash
python train.py --batch_size 16 --d_model 512  # Reduce batch size / model dim
```

**Dataset download fails:**
```bash
# Manually download and place in data/<dataset>/ then run:
python train.py --skip_download --dataset <name>
```

**Slow data loading:**
```bash
python train.py --num_workers 8 --prefetch_factor 2  # Increase parallelism
```

## Citation

If using this code, cite:
```bibtex
@article{perceiver-sensory,
  title={Perception Is How We Experience: A Unified Sensory Architecture},
  author={Zach},
  year={2026},
  note={Preprint}
}
```

## License

[Specify: MIT, Apache 2.0, etc.]
