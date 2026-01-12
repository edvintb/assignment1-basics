#!/bin/bash
# Learning rate sweep for a given batch size
# Usage: ./scripts/train_bs128_sweep.sh <batch_size> [config_file]
# Example: ./scripts/train_bs128_sweep.sh 128 config/train_slm_ts.yaml

set +e  # Don't exit on error - we want to continue even if some runs fail

# Batch size (required parameter)
if [ -z "$1" ]; then
    echo "Error: Batch size parameter is required"
    echo "Usage: $0 <batch_size> [config_file]"
    echo "Example: $0 128 config/train_slm_ts.yaml"
    exit 1
fi
BATCH_SIZE=$1

# Config file (optional, defaults to train_slm_ts.yaml)
CONFIG=${2:-config/train_slm_ts.yaml}

# Learning rates to test (same as original grid search)
LEARNING_RATES=(1.0e-5 5.0e-5 1.0e-4 3.0e-4 5.0e-4 1.0e-3)

export WANDB_PROJECT="llm_experiments"

# Results directory
RESULTS_DIR="results/grid_search"
mkdir -p "$RESULTS_DIR"

# Results file (we'll append to existing)
RESULTS_FILE="${RESULTS_DIR}/grid_search_results.csv"

echo "Starting batch size $BATCH_SIZE learning rate sweep with config: $CONFIG"
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "Total experiments: ${#LEARNING_RATES[@]}"
echo "Results will be appended to: $RESULTS_FILE"
echo "========================================"

experiment_num=0
total_experiments=${#LEARNING_RATES[@]}

# Learning rate sweep loop
for lr in "${LEARNING_RATES[@]}"; do
    experiment_num=$((experiment_num + 1))

    echo ""
    echo "========================================"
    echo "Experiment $experiment_num / $total_experiments"
    echo "Batch size: $BATCH_SIZE, Learning rate: $lr"
    echo "========================================"

    # Calculate min learning rate
    min_lr=$(awk "BEGIN {print $lr / 10}")

    # Create checkpoint path
    lr_name=$(echo "$lr" | sed 's/\./_/g' | sed 's/-/neg/g')
    checkpoint_path="checkpoints/grid_search/bs_${BATCH_SIZE}_lr_${lr_name}"

    echo "Checkpoint path: $checkpoint_path"
    echo "Max LR: $lr, Min LR: $min_lr"

    # Create a temporary file to capture the last few lines of output
    temp_output=$(mktemp)

    # Run training with overrides
    WANDB_RUN_NAME="bs=${BATCH_SIZE}_lr=${lr}" \
    uv run cs336_basics/train.py \
        --config "$CONFIG" \
        --override \
            data.batch_size="$BATCH_SIZE" \
            training.max_learning_rate="$lr" \
            training.min_learning_rate="$min_lr" \
            training.checkpoint_path="$checkpoint_path" \
        2>&1 | tee "$temp_output"

    exit_code=$?

    # Extract final loss and val loss from output (last logged step)
    last_step_line=$(grep "Step:" "$temp_output" | tail -1)
    final_loss=$(echo "$last_step_line" | grep -oP "Loss: \K[0-9.]+" | head -1)
    final_val_loss=$(echo "$last_step_line" | grep -oP "Val Loss: \K[0-9.]+")

    # If extraction failed, use "N/A"
    if [ -z "$final_loss" ]; then
        final_loss="N/A"
    fi
    if [ -z "$final_val_loss" ]; then
        final_val_loss="N/A"
    fi

    # Append results to CSV
    echo "$BATCH_SIZE,$lr,$min_lr,$final_loss,$final_val_loss,$exit_code,$checkpoint_path" >> "$RESULTS_FILE"

    # Clean up temp file
    rm "$temp_output"

    if [ $exit_code -ne 0 ]; then
        echo "WARNING: Training failed with exit code $exit_code"
        echo "Continuing with next experiment..."
    else
        echo "Completed successfully"
    fi
done

echo ""
echo "========================================"
echo "Batch size $BATCH_SIZE sweep completed!"
echo "Results appended to: $RESULTS_FILE"
echo "========================================"
echo ""

# Display updated results
echo "All results:"
echo ""
cat "$RESULTS_FILE"

echo ""
echo "Checkpoints saved in: checkpoints/grid_search/"
