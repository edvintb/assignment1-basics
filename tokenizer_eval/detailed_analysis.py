#!/usr/bin/env python3
"""
Detailed analysis of compression ratios with sample documents.
"""

import argparse
import random
from cs336_basics.tokenizer import Tokenizer


def analyze_sample_document(text: str, tokenizer: Tokenizer, doc_name: str):
    """Analyze a single document in detail."""
    print(f"\n{'-'*50}")
    print(f"Document: {doc_name}")
    print(f"{'-'*50}")
    
    # Show first 200 characters of the document
    preview = text[:200] + "..." if len(text) > 200 else text
    print(f"Preview: {preview}")
    
    # Encode and analyze
    token_ids = tokenizer.encode(text)
    num_bytes = len(text.encode('utf-8'))
    num_tokens = len(token_ids)
    compression_ratio = num_bytes / num_tokens if num_tokens > 0 else 0
    
    print(f"Length: {len(text)} characters")
    print(f"Bytes: {num_bytes}")
    print(f"Tokens: {num_tokens}")
    print(f"Compression ratio: {compression_ratio:.3f} bytes/token")
    
    # Show first few tokens
    print(f"First 10 token IDs: {token_ids[:10]}")
    
    # Decode first few tokens to show what they represent
    first_tokens_decoded = []
    for i, token_id in enumerate(token_ids[:10]):
        try:
            decoded = tokenizer.decode([token_id])
            first_tokens_decoded.append(f"'{decoded}'")
        except:
            first_tokens_decoded.append(f"<error:{token_id}>")
    
    print(f"First 10 tokens decoded: {', '.join(first_tokens_decoded)}")
    
    return num_bytes, num_tokens, compression_ratio


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detailed analysis of compression ratios with sample documents"
    )

    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file"
    )

    parser.add_argument(
        "--merges-file",
        type=str,
        required=True,
        help="Path to the merges JSON file"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of documents to analyze in detail from each dataset (default: 3)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--tinystories-path",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to TinyStories dataset file (default: data/TinyStoriesV2-GPT4-train.txt)"
    )

    parser.add_argument(
        "--owt-path",
        type=str,
        default="data/owt_train.txt",
        help="Path to OpenWebText dataset file (default: data/owt_train.txt)"
    )

    parser.add_argument(
        "--read-size",
        type=int,
        default=50000,
        help="Number of bytes to read from each dataset file (default: 50000)"
    )

    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    random.seed(args.random_seed)

    print("Loading trained tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_file, args.merges_file, special_tokens=["<|endoftext|>"])
    print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    print(f"Using random seed: {args.random_seed}")
    print(f"Vocab file: {args.vocab_file}")
    print(f"Merges file: {args.merges_file}")

    # Sample a few documents from each dataset for detailed analysis
    print("\n" + "="*60)
    print("DETAILED DOCUMENT ANALYSIS")
    print("="*60)

    # TinyStories samples
    print(f"\nTINYSTORIES SAMPLES:")
    with open(args.tinystories_path, 'r', encoding='utf-8') as f:
        content = f.read(args.read_size)  # Read specified amount
        ts_docs = [doc.strip() for doc in content.split("<|endoftext|>") if doc.strip()][:args.num_samples]
    
    ts_total_bytes = 0
    ts_total_tokens = 0
    
    for i, doc in enumerate(ts_docs):
        bytes_count, tokens_count, ratio = analyze_sample_document(doc, tokenizer, f"TinyStories #{i+1}")
        ts_total_bytes += bytes_count
        ts_total_tokens += tokens_count
    
    # OpenWebText samples
    print(f"\n\nOPENWEBTEXT SAMPLES:")
    with open(args.owt_path, 'r', encoding='utf-8') as f:
        content = f.read(args.read_size)  # Read specified amount
        owt_docs = [doc.strip() for doc in content.split("<|endoftext|>") if doc.strip()][:args.num_samples]
    
    owt_total_bytes = 0
    owt_total_tokens = 0
    
    for i, doc in enumerate(owt_docs):
        bytes_count, tokens_count, ratio = analyze_sample_document(doc, tokenizer, f"OpenWebText #{i+1}")
        owt_total_bytes += bytes_count
        owt_total_tokens += tokens_count
    
    # Summary comparison
    print(f"\n\n{'='*60}")
    print("SAMPLE COMPARISON SUMMARY")
    print("="*60)
    
    ts_avg_ratio = ts_total_bytes / ts_total_tokens if ts_total_tokens > 0 else 0
    owt_avg_ratio = owt_total_bytes / owt_total_tokens if owt_total_tokens > 0 else 0
    
    print(f"\nTinyStories (3 samples):")
    print(f"  Total bytes: {ts_total_bytes:,}")
    print(f"  Total tokens: {ts_total_tokens:,}")
    print(f"  Average compression ratio: {ts_avg_ratio:.3f} bytes/token")
    
    print(f"\nOpenWebText (3 samples):")
    print(f"  Total bytes: {owt_total_bytes:,}")
    print(f"  Total tokens: {owt_total_tokens:,}")
    print(f"  Average compression ratio: {owt_avg_ratio:.3f} bytes/token")
    
    print(f"\nDifference: {abs(ts_avg_ratio - owt_avg_ratio):.3f} bytes/token")
    
    if ts_avg_ratio > owt_avg_ratio:
        print("TinyStories shows better compression (higher bytes/token ratio)")
        print("This suggests the tokenizer is more efficient on TinyStories text")
    else:
        print("OpenWebText shows better compression (higher bytes/token ratio)")
        print("This suggests the tokenizer is more efficient on OpenWebText text")
    
    # Analysis of why this might be the case
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print("="*60)
    print("\nPossible reasons for compression differences:")
    print("1. Vocabulary match: The tokenizer was likely trained on similar data")
    print("2. Text complexity: Simpler, more repetitive text compresses better")
    print("3. Domain specificity: Specialized vocabulary may not be well represented")
    print("4. Language patterns: Different writing styles affect tokenization efficiency")

    # Print configuration summary
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print("="*60)
    print(f"Number of samples per dataset: {args.num_samples}")
    print(f"Random seed: {args.random_seed}")
    print(f"Read size per dataset: {args.read_size:,} bytes")
    print(f"TinyStories path: {args.tinystories_path}")
    print(f"OpenWebText path: {args.owt_path}")


if __name__ == "__main__":
    main()
