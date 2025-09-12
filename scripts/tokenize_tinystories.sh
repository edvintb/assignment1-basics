#!/usr/bin/env bash

cd "$(dirname "$0")/../" # Go to project root directory

# Tokenize TinyStoriesV2 training data
uv run tokenize_dataset/tokenize_dataset.py \
    ts-vocab.json \
    ts-merges.json \
    data/TinyStoriesV2-GPT4-train.txt
    # --special-tokens "<|endoftext|>" \
    # --output data/TinyStoriesV2-GPT4-train_tokenized.npz

# Tokenize TinyStoriesV2 validation data
uv run tokenize_dataset/tokenize_dataset.py \
    ts-vocab.json \
    ts-merges.json \
    data/TinyStoriesV2-GPT4-valid.txt
    # --special-tokens "<|endoftext|>" \
    # --output data/TinyStoriesV2-GPT4-valid_tokenized.npz
