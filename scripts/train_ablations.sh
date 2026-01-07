#!/bin/bash
# Script to train models for each ablation study
# Each ablation is on a separate git branch and will be trained sequentially

set -e  # Exit on error

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config/train_slm_ts.yaml}"
BASE_BRANCH="main"

# Array of ablations: branch_name:experiment_name
declare -a ABLATIONS=(
    "ablation/no-normalization:ablation_no_normalization"
    "ablation/post-norm:ablation_post_norm"
    "ablation/no-positional-encoding:ablation_no_positional_encoding"
    "ablation/silu-activation:ablation_silu_activation"
)

# Save current branch
ORIGINAL_BRANCH=$(git branch --show-current)

echo "Starting ablation study training..."
echo "Using config: $CONFIG_FILE"
echo "Original branch: $ORIGINAL_BRANCH"
echo ""

# Train baseline on main branch first
echo "========================================="
echo "Training BASELINE model on $BASE_BRANCH branch"
echo "========================================="
git checkout "$BASE_BRANCH"
WANDB_NAME="baseline" uv run python cs336_basics/train.py --config "$CONFIG_FILE"
echo ""

# Train each ablation
for ablation in "${ABLATIONS[@]}"; do
    IFS=':' read -r branch_name experiment_name <<< "$ablation"

    echo "========================================="
    echo "Training $experiment_name"
    echo "Branch: $branch_name"
    echo "========================================="

    # Checkout the ablation branch
    git checkout "$branch_name"

    # Train the model with the specific wandb name
    WANDB_NAME="$experiment_name" uv run python cs336_basics/train.py --config "$CONFIG_FILE"

    echo ""
done

# Return to original branch
echo "========================================="
echo "All ablations complete!"
echo "Returning to original branch: $ORIGINAL_BRANCH"
echo "========================================="
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "Training summary:"
echo "- Baseline model"
for ablation in "${ABLATIONS[@]}"; do
    IFS=':' read -r branch_name experiment_name <<< "$ablation"
    echo "- $experiment_name"
done
