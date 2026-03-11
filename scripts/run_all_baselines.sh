#!/bin/bash
# Run all baselines for the Split-CIFAR-100 CL benchmark.
# Override BASELINE_BATCH_SIZE for local hardware, for example 4 on 12 GB GPUs.

MODELS=("vilt" "clip" "flava" "perceiver_io" "perceiver")
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-4}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-5}"
N_TASKS="${N_TASKS:-20}"

# Create output directory
mkdir -p results/baselines_cl
echo "Starting Baseline Benchmarking Sequence..."
echo "Results will be saved sequentially to results/baselines_cl/"
echo "Batch size: ${BASELINE_BATCH_SIZE}"

for model in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Running benchmark for: $model"
    echo "=================================================="
    python scripts/run_baselines_cl.py \
        --model "$model" \
        --n_tasks "$N_TASKS" \
        --epochs_per_task "$EPOCHS_PER_TASK" \
        --batch_size "$BASELINE_BATCH_SIZE" \
        --output_dir results/baselines_cl
    
    if [ $? -ne 0 ]; then
        echo "Error running $model. Continuing to next..."
    fi
done

echo "Baseline benchmark sequence completed."
