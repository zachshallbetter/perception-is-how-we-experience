#!/bin/bash
# Run all baselines for the Split-CIFAR-100 CL benchmark
# Using batch_size=16 to prevent MPS Out-of-Memory errors on standard Macs.

MODELS=("vilt" "clip" "flava" "perceiver_io" "perceiver")

# Create output directory
mkdir -p results/baselines_cl
echo "Starting Baseline Benchmarking Sequence..."
echo "Results will be saved sequentially to results/baselines_cl/"

for model in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Running benchmark for: $model"
    echo "=================================================="
    python scripts/run_baselines_cl.py \
        --model "$model" \
        --n_tasks 10 \
        --epochs_per_task 5 \
        --batch_size 16 \
        --output_dir results/baselines_cl
    
    if [ $? -ne 0 ]; then
        echo "Error running $model. Continuing to next..."
    fi
done

echo "Baseline benchmark sequence completed."
