#!/usr/bin/env python3
"""
Sample documents from TinyStories and OpenWebText datasets and analyze compression ratios
using the trained BPE tokenizer.
"""

import argparse
import random
from cs336_basics.tokenizer import Tokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze compression ratios of BPE tokenizer on TinyStories and OpenWebText datasets"
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
        default=10,
        help="Number of documents to sample from each dataset (default: 10)"
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

    return parser.parse_args()



def sample_documents_from_file(file_path: str, num_samples: int = 10, separator: str = "<|endoftext|>") -> list[str]:
    """
    Sample documents from a text file where documents are separated by a separator.
    Uses memory-efficient streaming approach for large files.

    Args:
        file_path: Path to the text file
        num_samples: Number of documents to sample
        separator: String that separates documents

    Returns:
        List of sampled document strings
    """
    print(f"Reading documents from {file_path}...")

    # For large files, use a streaming approach
    documents = []
    current_doc = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if separator in line:
                # Split the line by separator
                parts = line.split(separator)
                # Add the first part to current document
                current_doc += parts[0]

                # Save the current document if it's not empty
                if current_doc.strip():
                    documents.append(current_doc.strip())

                # Process any complete documents in the middle
                for i in range(1, len(parts) - 1):
                    if parts[i].strip():
                        documents.append(parts[i].strip())

                # Start new document with the last part
                current_doc = parts[-1]

                # If we have enough documents for sampling, break early
                if len(documents) >= num_samples * 100:  # Sample from a larger pool
                    break
            else:
                current_doc += line

    # Add the last document if it exists
    if current_doc.strip():
        documents.append(current_doc.strip())

    print(f"Found {len(documents)} documents in {file_path}")

    if len(documents) < num_samples:
        print(f"Warning: Only {len(documents)} documents available, sampling all of them")
        return documents

    # Randomly sample documents
    sampled_docs = random.sample(documents, num_samples)
    print(f"Sampled {len(sampled_docs)} documents")

    return sampled_docs


def calculate_compression_ratio(text: str, tokenizer: Tokenizer) -> tuple[int, int, float]:
    """
    Calculate compression ratio for a given text.
    
    Args:
        text: Input text string
        tokenizer: Trained tokenizer
    
    Returns:
        Tuple of (num_bytes, num_tokens, compression_ratio)
    """
    # Encode text to get token IDs
    token_ids = tokenizer.encode(text)
    
    # Calculate metrics
    num_bytes = len(text.encode('utf-8'))
    num_tokens = len(token_ids)
    compression_ratio = num_bytes / num_tokens if num_tokens > 0 else 0
    
    return num_bytes, num_tokens, compression_ratio


def analyze_dataset(dataset_name: str, documents: list[str], tokenizer: Tokenizer) -> dict:
    """
    Analyze compression ratios for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        documents: List of document strings
        tokenizer: Trained tokenizer
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\nAnalyzing {dataset_name} dataset...")
    
    total_bytes = 0
    total_tokens = 0
    compression_ratios = []
    
    for i, doc in enumerate(documents):
        num_bytes, num_tokens, compression_ratio = calculate_compression_ratio(doc, tokenizer)
        
        total_bytes += num_bytes
        total_tokens += num_tokens
        compression_ratios.append(compression_ratio)
        
        print(f"  Document {i+1}: {num_bytes} bytes, {num_tokens} tokens, ratio: {compression_ratio:.3f}")
    
    # Calculate overall statistics
    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
    overall_compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    
    results = {
        'dataset_name': dataset_name,
        'num_documents': len(documents),
        'total_bytes': total_bytes,
        'total_tokens': total_tokens,
        'individual_ratios': compression_ratios,
        'average_compression_ratio': avg_compression_ratio,
        'overall_compression_ratio': overall_compression_ratio
    }
    
    return results


def main():
    """Main function to run the analysis."""
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    print(f"Using random seed: {args.random_seed}")

    print("Loading trained tokenizer...")
    # Load the trained tokenizer
    tokenizer = Tokenizer.from_file(args.vocab_file, args.merges_file, special_tokens=["<|endoftext|>"])
    print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    print(f"Vocab file: {args.vocab_file}")
    print(f"Merges file: {args.merges_file}")

    # Sample documents from both datasets
    print("\n" + "="*60)
    print("SAMPLING DOCUMENTS")
    print("="*60)

    # Sample from TinyStories
    tinystories_docs = sample_documents_from_file(args.tinystories_path, num_samples=args.num_samples)

    # Sample from OpenWebText
    owt_docs = sample_documents_from_file(args.owt_path, num_samples=args.num_samples)
    
    # Analyze both datasets
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)
    
    tinystories_results = analyze_dataset("TinyStories", tinystories_docs, tokenizer)
    owt_results = analyze_dataset("OpenWebText", owt_docs, tokenizer)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nTinyStories Results:")
    print(f"  Total documents: {tinystories_results['num_documents']}")
    print(f"  Total bytes: {tinystories_results['total_bytes']:,}")
    print(f"  Total tokens: {tinystories_results['total_tokens']:,}")
    print(f"  Average compression ratio: {tinystories_results['average_compression_ratio']:.3f} bytes/token")
    print(f"  Overall compression ratio: {tinystories_results['overall_compression_ratio']:.3f} bytes/token")

    print(f"\nOpenWebText Results:")
    print(f"  Total documents: {owt_results['num_documents']}")
    print(f"  Total bytes: {owt_results['total_bytes']:,}")
    print(f"  Total tokens: {owt_results['total_tokens']:,}")
    print(f"  Average compression ratio: {owt_results['average_compression_ratio']:.3f} bytes/token")
    print(f"  Overall compression ratio: {owt_results['overall_compression_ratio']:.3f} bytes/token")

    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Number of samples per dataset: {args.num_samples}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  TinyStories path: {args.tinystories_path}")
    print(f"  OpenWebText path: {args.owt_path}")
    
    # Compare the two datasets
    print(f"\nComparison:")
    ts_avg = tinystories_results['average_compression_ratio']
    owt_avg = owt_results['average_compression_ratio']
    
    if ts_avg > owt_avg:
        better_dataset = "TinyStories"
        difference = ts_avg - owt_avg
    else:
        better_dataset = "OpenWebText"
        difference = owt_avg - ts_avg
    
    print(f"  {better_dataset} has better (higher) compression ratio by {difference:.3f} bytes/token")
    print(f"  This means the tokenizer is more efficient on {better_dataset} text")


if __name__ == "__main__":
    main()
