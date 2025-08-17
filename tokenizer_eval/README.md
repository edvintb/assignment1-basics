# Tokenizer Evaluation

This folder contains scripts and results for evaluating the compression efficiency of the trained BPE tokenizer on TinyStories and OpenWebText datasets.

## Files

- `sample_and_analyze.py` - Main analysis script that samples 10 documents from each dataset and calculates compression ratios
- `detailed_analysis.py` - Detailed analysis showing individual document breakdowns and token examples
- `compression_analysis_results.md` - Comprehensive results summary and analysis
- `README.md` - This file

## Requirements

The scripts require:
- The `cs336_basics` package installed and available on the Python path
- Trained tokenizer files (vocab.json and merges.json)
- Dataset files (TinyStories and OpenWebText)

## Usage

Both scripts now accept command-line arguments for flexibility:

### Main Analysis Script

```bash
# Basic usage with required arguments
uv run sample_and_analyze.py --vocab-file vocab.json --merges-file merges.json

# With custom parameters
uv run sample_and_analyze.py \
    --vocab-file vocab.json \
    --merges-file merges.json \
    --num-samples 20 \
    --random-seed 123 \
    --tinystories-path data/TinyStoriesV2-GPT4-train.txt \
    --owt-path data/owt_train.txt
```

### Detailed Analysis Script

```bash
# Basic usage with required arguments
uv run detailed_analysis.py --vocab-file vocab.json --merges-file merges.json

# With custom parameters
uv run detailed_analysis.py \
    --vocab-file vocab.json \
    --merges-file merges.json \
    --num-samples 5 \
    --random-seed 123 \
    --read-size 100000 \
    --tinystories-path data/TinyStoriesV2-GPT4-train.txt \
    --owt-path data/owt_train.txt
```

### Command-Line Arguments

**Common arguments for both scripts:**
- `--vocab-file`: Path to vocabulary JSON file (required)
- `--merges-file`: Path to merges JSON file (required)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--tinystories-path`: Path to TinyStories dataset (default: data/TinyStoriesV2-GPT4-train.txt)
- `--owt-path`: Path to OpenWebText dataset (default: data/owt_train.txt)

**sample_and_analyze.py specific:**
- `--num-samples`: Number of documents to sample from each dataset (default: 10)

**detailed_analysis.py specific:**
- `--num-samples`: Number of documents to analyze in detail (default: 3)
- `--read-size`: Number of bytes to read from each dataset file (default: 50000)

## Results Summary

The analysis shows that the BPE tokenizer achieves better compression on TinyStories compared to OpenWebText:

- **TinyStories**: 4.062 bytes/token average compression ratio
- **OpenWebText**: 3.348 bytes/token average compression ratio
- **Difference**: 0.714 bytes/token in favor of TinyStories

This indicates the tokenizer is more efficient on simpler, narrative text patterns found in TinyStories.

## Analysis Details

The scripts:
1. Load the trained BPE tokenizer from `vocab.json` and `merges.json`
2. Sample 10 random documents from each dataset
3. Encode each document using the tokenizer
4. Calculate compression ratios (bytes per token)
5. Provide statistical analysis and comparisons

See `compression_analysis_results.md` for detailed findings and interpretation.
