#!/usr/bin/env python3
"""
Script to tokenize a text file using vocab and merges files with parallel processing.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


class TokenizeChunkArgs(NamedTuple):
    """Arguments for tokenizing a chunk of text."""
    path: str
    vocab_file: str
    merges_file: str
    special_tokens: list[str]
    start: int
    end: int
    chunk_id: int  # To maintain order


def tokenize_chunk(args: TokenizeChunkArgs) -> tuple[int, list[int]]:
    """Tokenize a chunk of text from a file."""
    path, vocab_file, merges_file, special_tokens, start, end, chunk_id = args
    
    # Load tokenizer for this process
    tokenizer = Tokenizer.from_file(vocab_file, merges_file, special_tokens)
    
    # Read the chunk
    with open(path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
    
    # Tokenize the chunk
    token_ids = tokenizer.encode(chunk_text)
    
    return chunk_id, token_ids


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize a text file using vocab and merges files"
    )
    parser.add_argument("vocab_file", type=str, help="Path to the vocabulary JSON file")
    parser.add_argument("merges_file", type=str, help="Path to the merges JSON file")
    parser.add_argument("text_file", type=str, help="Path to the text file to tokenize")
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"], 
                       help="Special tokens to use (default: ['<|endoftext|>'])")
    parser.add_argument("--output", type=str, 
                       help="Output NPZ file path (default: input filename with _tokenized.npz suffix)")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes to use (default: CPU count - 1)")
    
    return parser.parse_args()


def main():
    """Main tokenization function with parallel processing."""
    args = parse_args()
    
    # Validate input files exist
    for file_path, name in [(args.vocab_file, "vocab"), (args.merges_file, "merges"), (args.text_file, "text")]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file '{file_path}' does not exist", file=sys.stderr)
            sys.exit(1)
    
    # Determine output file path
    if args.output:
        output_path = args.output
    else:
        text_path = Path(args.text_file)
        if text_path.parent == Path("."):
            output_path = f"{text_path.stem}_tokenized.npz"
        else:
            output_path = text_path.parent / f"{text_path.stem}_tokenized.npz"
    
    # Determine number of processes
    if args.num_processes is None:
        cpu_count = os.cpu_count() or 1
        num_processes = max(cpu_count - 1, 1)
    else:
        num_processes = args.num_processes
    
    print(f"Parallel tokenization setup:")
    print(f"  Vocab file: {args.vocab_file}")
    print(f"  Merges file: {args.merges_file}")
    print(f"  Special tokens: {args.special_tokens}")
    print(f"  Input file: {args.text_file}")
    print(f"  Output file: {output_path}")
    print(f"  Processes: {num_processes}")
    
    # Get file size
    file_size = os.path.getsize(args.text_file)
    print(f"  File size: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
    
    # Find chunk boundaries
    print("Finding chunk boundaries...")
    special_token_bytes = args.special_tokens[0].encode('utf-8') if args.special_tokens else b"<|endoftext|>"
    
    try:
        with open(args.text_file, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes)
        print(f"Split file into {len(boundaries) - 1} chunks")
    except Exception as e:
        print(f"Error finding chunk boundaries: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare arguments for parallel tokenization
    chunk_args = [
        TokenizeChunkArgs(
            path=args.text_file,
            vocab_file=args.vocab_file,
            merges_file=args.merges_file,
            special_tokens=args.special_tokens,
            start=start,
            end=end,
            chunk_id=i
        )
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    
    print(f"Tokenizing {len(chunk_args)} chunks in parallel...")
    
    # Process chunks in parallel
    try:
        with Pool(num_processes) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(tokenize_chunk, chunk_args), 
                             total=len(chunk_args), 
                             desc="Tokenizing chunks"):
                results.append(result)
        
        print("Parallel tokenization completed. Assembling results...")
        
        # Sort results by chunk_id to maintain order
        results.sort(key=lambda x: x[0])
        
        # Concatenate all token lists
        all_token_ids = []
        for chunk_id, token_ids in results:
            all_token_ids.extend(token_ids)
        
        print(f"Tokenized into {len(all_token_ids):,} tokens total")
        
    except Exception as e:
        print(f"Error during parallel tokenization: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check if any token IDs exceed uint16 range
    max_token_id = max(all_token_ids) if all_token_ids else 0
    if max_token_id > 65535:  # 2^16 - 1
        print(f"Warning: Maximum token ID ({max_token_id}) exceeds uint16 range (0-65535)")
        print("Consider using uint32 dtype instead")
        sys.exit(1)
    
    # Convert to numpy array with uint16 dtype
    try:
        token_array = np.array(all_token_ids, dtype=np.uint16)
        print(f"Created numpy array with shape {token_array.shape} and dtype {token_array.dtype}")
    except Exception as e:
        print(f"Error creating numpy array: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Saving tokenized data to: {output_path}")
    
    # Save as NPZ file
    try:
        np.savez_compressed(output_path, tokens=token_array)
        print(f"Successfully saved tokenized data to {output_path}")
        
        # Print some statistics
        print(f"\nTokenization Statistics:")
        print(f"  Input file size: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
        print(f"  Output tokens: {len(all_token_ids):,} tokens")
        print(f"  Compression ratio: {file_size / len(all_token_ids):.2f} bytes/token")
        print(f"  Token ID range: {min(all_token_ids) if all_token_ids else 0} - {max(all_token_ids) if all_token_ids else 0}")
        print(f"  Output file size: {os.path.getsize(output_path):,} bytes")
        print(f"  Chunks processed: {len(chunk_args)}")
        
    except Exception as e:
        print(f"Error saving NPZ file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
