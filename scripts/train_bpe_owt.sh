#!/usr/bin/env bash

echo "Training BPE training on Open Web Text..."

uv run cs336_basics/train_bpe.py \
    --dataset data/owt_train.txt \
    --vocab_size 32000 \
    --special_tokens "<|endoftext|>" \
    --vocab-output data/owt-train.vocab.json \
    --merges-output data/owt-train.merges.json
