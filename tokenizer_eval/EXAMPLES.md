# Tokenizer Evaluation Examples

## Quick Start

```bash
# Basic analysis with TinyStories tokenizer
uv run tokenizer_eval/sample_and_analyze.py --vocab-file ts-vocab.json --merges-file ts-merges.json

# Detailed token-level analysis
uv run tokenizer_eval/detailed_analysis.py --vocab-file ts-vocab.json --merges-file ts-merges.json
```

## Common Usage Patterns

### 1. Compare Different Tokenizers

```bash
# TinyStories tokenizer (vocab size: 10,000)
uv run tokenizer_eval/sample_and_analyze.py \
    --vocab-file ts-vocab.json \
    --merges-file ts-merges.json \
    --num-samples 10

# OpenWebText tokenizer (vocab size: 32,000)
uv run tokenizer_eval/sample_and_analyze.py \
    --vocab-file owt-vocab.json \
    --merges-file owt-merges.json \
    --num-samples 10
```

### 2. Reproducible Analysis

```bash
# Use specific random seed for reproducible results
uv run tokenizer_eval/sample_and_analyze.py \
    --vocab-file ts-vocab.json \
    --merges-file ts-merges.json \
    --random-seed 42 \
    --num-samples 20
```

### 3. Custom Dataset Paths

```bash
# Specify custom dataset locations
uv run tokenizer_eval/sample_and_analyze.py \
    --vocab-file ts-vocab.json \
    --merges-file ts-merges.json \
    --tinystories-path /path/to/your/tinystories.txt \
    --owt-path /path/to/your/openwebtext.txt
```

### 4. Detailed Analysis with Custom Parameters

```bash
# Analyze fewer samples but read more data
uv run tokenizer_eval/detailed_analysis.py \
    --vocab-file owt-vocab.json \
    --merges-file owt-merges.json \
    --num-samples 2 \
    --read-size 100000 \
    --random-seed 123
```

### 5. Running from Different Directories

```bash
# From project root
uv run tokenizer_eval/sample_and_analyze.py --vocab-file ts-vocab.json --merges-file ts-merges.json

# From tokenizer_eval directory
cd tokenizer_eval
uv run sample_and_analyze.py --vocab-file ../ts-vocab.json --merges-file ../ts-merges.json
```

## Parameter Reference

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--vocab-file` | Path to vocabulary JSON file | - | ✅ |
| `--merges-file` | Path to merges JSON file | - | ✅ |
| `--num-samples` | Number of documents to sample | 10 (sample_and_analyze)<br>3 (detailed_analysis) | ❌ |
| `--random-seed` | Random seed for reproducibility | 42 | ❌ |
| `--tinystories-path` | Path to TinyStories dataset | `data/TinyStoriesV2-GPT4-train.txt` | ❌ |
| `--owt-path` | Path to OpenWebText dataset | `data/owt_train.txt` | ❌ |
| `--read-size` | Bytes to read from dataset (detailed_analysis only) | 50000 | ❌ |

## Available Tokenizer Files

Based on your project structure:

- **TinyStories tokenizer**: `ts-vocab.json`, `ts-merges.json` (vocab size: 10,000)
- **OpenWebText tokenizer**: `owt-vocab.json`, `owt-merges.json` (vocab size: 32,000)

## Expected Output

### sample_and_analyze.py
- Compression ratios for each sampled document
- Summary statistics for both datasets
- Comparison showing which dataset compresses better
- Configuration details

### detailed_analysis.py
- Token-level breakdown for sample documents
- First 10 tokens and their decoded values
- Document previews and statistics
- Analysis of compression differences

## Tips

1. **Use consistent random seeds** for reproducible comparisons
2. **Start with small sample sizes** to test quickly, then increase for more robust results
3. **Compare tokenizers trained on different datasets** to see domain-specific effects
4. **Use detailed_analysis.py** to understand why certain texts compress better
