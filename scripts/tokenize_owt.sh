#!/usr/bin/env bash

# python interpreter print
echo "python: $(which python)"

cd "$(dirname "$0")/../" # Go to project root directory

# Tokenize owt training data
uv run tokenize_dataset/tokenize_dataset.py \
    data/owt-train.vocab.json \
    data/owt-train.merges.json \
    data/owt_train.txt
    # --special-tokens "<|endoftext|>" \
    # --output data/TinyStoriesV2-GPT4-train_tokenized.npz

