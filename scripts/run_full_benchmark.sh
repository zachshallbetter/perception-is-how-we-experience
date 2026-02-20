#!/bin/bash
# Full benchmark suite: 4 experiments × 3 seeds × evaluation

set -e

RESULTS_DIR="results/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Sensory Perceiver Full Benchmark Suite"
echo "Results: $RESULTS_DIR"
echo "=========================================="

# Continual Learning: Split-CIFAR-100
echo -e "\n[1/4] Running Continual Learning (Split-CIFAR-100)..."
for seed in 0 1 2; do
    OUTPUT_DIR="$RESULTS_DIR/continual_cifar_seed_$seed"
    mkdir -p "$OUTPUT_DIR"
    python -m src.train \
        --experiment continual \
        --dataset split_cifar \
        --batch_size 32 \
        --epochs 50 \
        --learning_rate 1e-4 \
        --d_model 768 \
        --num_heads 8 \
        --depth 6 \
        --replay_buffer_size 2000 \
        --device cuda:0 \
        --seed $seed \
        --output_dir "$OUTPUT_DIR"

    python -m src.eval \
        --checkpoint "$OUTPUT_DIR/best_model.pt" \
        --experiment continual \
        --dataset split_cifar \
        --output_file "$OUTPUT_DIR/eval_results.json"
done

# Continual Learning: Permuted MNIST
echo -e "\n[2/4] Running Continual Learning (Permuted MNIST)..."
for seed in 0 1 2; do
    OUTPUT_DIR="$RESULTS_DIR/continual_mnist_seed_$seed"
    mkdir -p "$OUTPUT_DIR"
    python -m src.train \
        --experiment continual \
        --dataset permuted_mnist \
        --batch_size 32 \
        --epochs 50 \
        --learning_rate 1e-4 \
        --d_model 768 \
        --num_heads 8 \
        --depth 6 \
        --replay_buffer_size 2000 \
        --device cuda:0 \
        --seed $seed \
        --output_dir "$OUTPUT_DIR"

    python -m src.eval \
        --checkpoint "$OUTPUT_DIR/best_model.pt" \
        --experiment continual \
        --dataset permuted_mnist \
        --output_file "$OUTPUT_DIR/eval_results.json"
done

# IoT Streaming
echo -e "\n[3/4] Running IoT Streaming Classification..."
for seed in 0 1 2; do
    OUTPUT_DIR="$RESULTS_DIR/iot_streaming_seed_$seed"
    mkdir -p "$OUTPUT_DIR"
    python -m src.train \
        --experiment iot \
        --dataset iot \
        --batch_size 32 \
        --epochs 50 \
        --learning_rate 1e-4 \
        --d_model 768 \
        --num_heads 8 \
        --depth 6 \
        --device cuda:0 \
        --seed $seed \
        --output_dir "$OUTPUT_DIR"

    python -m src.eval \
        --checkpoint "$OUTPUT_DIR/best_model.pt" \
        --experiment iot \
        --dataset iot \
        --output_file "$OUTPUT_DIR/eval_results.json"
done

echo -e "\n[4/4] Aggregating results..."
python scripts/aggregate_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_file "$RESULTS_DIR/summary.csv"

echo -e "\n=========================================="
echo "Benchmark complete!"
echo "Results: $RESULTS_DIR"
echo "=========================================="
