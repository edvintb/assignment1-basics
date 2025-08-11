#!/usr/bin/env bash

cd "$(dirname "$0")/../" # Go to project root directory

echo "Running BPE training on TinyStories..."

uv run cs336_basics/train_bpe.py \
    --input_path data/TinyStoriesV2-GPT4-train.txt \
    --vocab_size 10000 \
    --special_tokens "<|endoftext|>"