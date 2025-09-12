#!/usr/bin/env bash

cd "$(dirname "$0")/../" # Go to project root directory

echo "Running BPE training on TinyStories..."

uv run cs336_basics/train_bpe.py \
    --dataset data/TinyStoriesV2-GPT4-train.txt \
    --vocab_size 10000 \
    --special_tokens "<|endoftext|>" \
    --vocab-output data/TinyStoriesV2-GPT4-train.vocab.json \
    --merges-output data/TinyStoriesV2-GPT4-train.merges.json
