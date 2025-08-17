import json
import os
from collections.abc import Iterator
from typing import NamedTuple
import regex as re  # use regex instead of re
import cProfile
import pstats
from pstats import SortKey
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse
from multiprocessing import Pool

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.common import perform_merge

# setup the type aliases
# important to expclicitly map objects to datatypes
VocabElt = bytes
Pretoken = tuple[VocabElt, ...]
VocabPair = tuple[VocabElt, VocabElt]


class PretokenArgs(NamedTuple):
    path: str | os.PathLike
    special_tokens: list[bytes]
    start: int
    end: int


def pretoken2pairs(pretoken: Pretoken) -> Iterator[VocabPair]:
    """Given a pretoken, return a set of all byte pairs in the pretoken."""
    return zip(pretoken[:-1], pretoken[1:])


def get_byte_corpus_for_chunk(args: PretokenArgs) -> Counter[Pretoken]:
    """Given a chunk of bytes and special tokens as bytes, return a Counter of byte tuples."""
    path, special_tokens, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        pretoken_to_count: Counter[bytes] = pretokenize_chunk(chunk, special_tokens)
        byte_corpus: Counter[Pretoken] = Counter(
            {tuple(bytes([byte]) for byte in key): value for key, value in pretoken_to_count.items()}
        )
        return byte_corpus


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # initalize the vocabulary with all bytes and special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # calculate how many merges we need to do to reach the desired vocab size
    total_merges = vocab_size - len(vocab)

    if total_merges <= 0:
        return vocab, []

    # split large textfile into its independent documents (chunks)
    cpu_count = os.cpu_count() or 0
    num_processes = max(cpu_count - 1, 1)
    with open(input_path, "rb") as f:
        # we will add the code to split this into chunks, even though we might not need it
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    # prepare arguments for pretokenization
    special_tokens_bytes: list[bytes] = [token.encode("utf-8") for token in special_tokens]
    pretoken_args = [
        PretokenArgs(path=input_path, special_tokens=special_tokens_bytes, start=start, end=end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # pretokenize in parallel
    print(f"Pre-tokenizing corpus with {num_processes} processes...")
    with Pool(num_processes) as pool:
        byte_corpus = sum(pool.imap_unordered(get_byte_corpus_for_chunk, pretoken_args), Counter())

    print("Done pre-tokenizing corpus.")

    # a list to track the merges we make
    merges: list[tuple[bytes, bytes]] = []

    # initialize the data structures we need to keep track of pairs
    pair_to_pretokens, pair_to_count = initialize_aux_mappings(byte_corpus)
    for _ in tqdm(range(total_merges), desc="Training BPE", unit=" merges"):
        # get the next pair to merge
        merge_pair = get_next_byte_pair(pair_to_count)

        # if there are no more pairs to merge, break
        if merge_pair is None:
            break

        # add pair to the vocabulary
        vocab[len(vocab)] = b"".join(merge_pair)

        # add pair to the list of merges
        merges.append(merge_pair)

        # update auxiliary data structures
        byte_corpus, pair_to_count, pair_to_pretokens = updates_for_pair(
            merge_pair=merge_pair,
            corpus=byte_corpus,
            pair_to_count=pair_to_count,
            pair_to_pretokens=pair_to_pretokens,
        )

    return vocab, merges


def initialize_aux_mappings(
    corpus: Counter[Pretoken],
) -> tuple[
    dict[VocabPair, set[Pretoken]],
    Counter[VocabPair],
]:
    """
    Args:
        corpus (Counter[tuple[bytes, ...]]): A counter of pretokens and their frequencies.

    Returns:
        tuple containing:
        - A dictionary mapping each byte pair to the set of pretokens containing it
        - A counter mapping each byte pair to its count in the corpus
    """
    pair_to_count: Counter[VocabPair] = Counter()
    pair_to_pretokens: dict[VocabPair, set[Pretoken]] = defaultdict(set)

    for pretoken, count in corpus.items():
        for pair in pretoken2pairs(pretoken):
            pair_to_pretokens[pair].add(pretoken)
            pair_to_count[pair] += count

    return pair_to_pretokens, pair_to_count


def get_next_byte_pair(
    pair_to_count: Counter[VocabPair],
) -> VocabPair | None:
    """
    Get the next byte pair to merge.

    Args:
        pair_to_count (Counter[VocabPair]): A Counter of all byte pairs and their counts.

    Returns:
        The next byte pair to merge.

    """
    max_count = 0
    top_pairs: set[VocabPair] = set()
    for pair, count in pair_to_count.items():
        if count > max_count:
            max_count = count
            top_pairs = {pair}
        elif count == max_count:
            top_pairs.add(pair)

    if len(top_pairs) == 0:
        return None

    next_pair = max(top_pairs)
    return next_pair


def updates_for_pair(
    merge_pair: VocabPair,
    corpus: Counter[Pretoken],
    pair_to_count: Counter[VocabPair],
    pair_to_pretokens: dict[VocabPair, set[Pretoken]],
) -> tuple[Counter[Pretoken], Counter[VocabPair], dict[VocabPair, set[Pretoken]]]:
    """Given a vocabulary and a pair of bytes, update the vocabulary by merging the pair."""

    # we need to copy the set to avoid modifying while iterating
    old_pretokens = pair_to_pretokens[merge_pair].copy()

    for old_pretoken in old_pretokens:
        for pair in pretoken2pairs(old_pretoken):
            # decrement the counts for each pair in the modified pretoken
            pair_to_count[pair] -= corpus[old_pretoken]

            # we need a check for the cases where we have multiple copies of a pair in a pretoken
            if old_pretoken in pair_to_pretokens[pair]:
                # remove the old pretoken from the pair_to_pretokens
                pair_to_pretokens[pair].remove(old_pretoken)

        # merge the vocab elements to create the new pretoken
        new_pretoken = perform_merge(old_pretoken, merge_pair)

        # update the corpus to contain this new pretoken (with the old pretoken's count)
        corpus[new_pretoken] = corpus.pop(old_pretoken)

        # update the counts for each pair in the new pretoken
        for pair in pretoken2pairs(new_pretoken):
            pair_to_count[pair] += corpus[new_pretoken]

            # add the new pretoken to the pair_to_pretokens
            pair_to_pretokens[pair].add(new_pretoken)

    # we should have removed all instances of the merged pair
    # assert (pair_to_count[merge_pair] == 0), f"Pair count for {merge_pair} is not zero: {pair_to_count[merge_pair]}"

    # we are modifying dicts, so I don't think we actually have to return anything
    return corpus, pair_to_count, pair_to_pretokens


def pretokenize_chunk(chunk: bytes, special_tokens: list[bytes]) -> Counter[bytes]:
    """
    Given bytes and special tokens as bytes, return a list of documents,
    where each document is a list of byte tokens.
    """
    # Join special tokens with | directly as bytes
    special_token_bytes = b"|".join(special_tokens)
    special_token_regex = re.compile(special_token_bytes)

    # Split the chunk using the pattern
    documents = special_token_regex.split(chunk)

    # GPT-2 regex pattern as bytes
    pretoken_regex = re.compile(rb"""'(?:[sdmt]|ll|ve|re)|\ ?\p{L}+|\ ?\p{N}+|\ ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # count up all the pretokens
    pretoken_to_count: Counter[bytes] = Counter()
    for doc in documents:
        pretoken_to_count += Counter(match.group() for match in pretoken_regex.finditer(doc))

    return pretoken_to_count


def serialize_vocab(vocab: dict[int, bytes], filepath: str) -> None:
    """Serialize vocab to JSON with bytes as UTF-8 strings where possible."""
    vocab_serializable = {}
    for k, v in vocab.items():
        try:
            vocab_serializable[k] = v.decode("utf-8")
        except UnicodeDecodeError:
            # Fall back to hex for invalid UTF-8 sequences
            vocab_serializable[k] = f"<HEX:{v.hex()}>"

    with open(filepath, "w") as f:
        json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)


def serialize_merges(merges: list[tuple[bytes, bytes]], filepath: str) -> None:
    """Serialize merges to JSON with bytes as UTF-8 strings where possible."""
    merges_serializable = []
    for pair in merges:
        serialized_pair = []
        for token in pair:
            try:
                serialized_pair.append(token.decode("utf-8"))
            except UnicodeDecodeError:
                serialized_pair.append(f"<HEX:{token.hex()}>")
        merges_serializable.append(serialized_pair)

    with open(filepath, "w") as f:
        json.dump(merges_serializable, f, indent=2, ensure_ascii=False)


def deserialize_vocab(filepath: str) -> dict[int, bytes]:
    """Deserialize vocab from JSON with hex strings back to bytes."""
    with open(filepath, "r") as f:
        loaded = json.load(f)

    vocab = {}
    for k, v in loaded.items():
        if v.startswith("<HEX:") and v.endswith(">"):
            # Extract hex string and convert back to bytes
            hex_str = v[5:-1]  # Remove "<HEX:" and ">"
            vocab[int(k)] = bytes.fromhex(hex_str)
        else:
            # Regular UTF-8 string, encode back to bytes
            vocab[int(k)] = v.encode("utf-8")

    return vocab


def deserialize_merges(filepath: str) -> list[tuple[bytes, bytes]]:
    """Deserialize merges from JSON with hex strings back to bytes."""
    with open(filepath, "r") as f:
        loaded = json.load(f)

    merges = []
    for pair in loaded:
        decoded_pair = []
        for token in pair:
            if token.startswith("<HEX:") and token.endswith(">"):
                # Extract hex string and convert back to bytes
                hex_str = token[5:-1]  # Remove "<HEX:" and ">"
                decoded_pair.append(bytes.fromhex(hex_str))
            else:
                # Regular UTF-8 string, encode back to bytes
                decoded_pair.append(token.encode("utf-8"))
        merges.append(tuple(decoded_pair))

    return merges


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--special_tokens", type=str, nargs="*", default=[])
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    # Create a Profile object
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the code to profile
    print("Training BPE tokenizer...")
    print(f"Input path: {args.input_path}")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"Special tokens: {args.special_tokens}")
    vocab, merges = train_bpe(
        input_path=args.input_path, vocab_size=args.vocab_size, special_tokens=args.special_tokens
    )

    print("Done training BPE tokenizer.")

    # Disable profiler and print stats
    profiler.disable()

    # Sort by cumulative time and print top 20 functions
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

    # For more detailed analysis of specific functions
    stats.print_callers("pretokenize_chunk")
    stats.print_callers("get_next_byte_pair")
    stats.print_callers("updates_for_pair")

    # serialize the vocab as a json file
    serialize_vocab(vocab, "vocab.json")

    # serialize the merges as a json file
    serialize_merges(merges, "merges.json")


if __name__ == "__main__":
    main(parse_args())
