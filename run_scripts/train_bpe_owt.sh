#!/usr/bin/env bash

cd "$(dirname "$0")/../" # Go to project root directory

echo "Running BPE training on TinyStories..."

uv run cs336_basics/train_bpe.py \
    --input_path data/owt_train.txt \
    --vocab_size 32000 \
    --special_tokens "<|endoftext|>"
