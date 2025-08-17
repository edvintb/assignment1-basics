import os
import json
from multiprocessing import Pool
from collections.abc import Iterator

from cs336_basics.pretokenization_example import find_chunk_boundaries

VocabElt = bytes
Pretoken = tuple[VocabElt, ...]
VocabPair = tuple[VocabElt, VocabElt]


def perform_merge(pretoken: Pretoken, pair_to_merge: VocabPair) -> Pretoken:
    """Merge a pair of bytes in a pretoken."""
    i = 0
    new_pretoken = []
    while i < len(pretoken):
        if i < len(pretoken) - 1 and pretoken[i] == pair_to_merge[0] and pretoken[i + 1] == pair_to_merge[1]:
            new_pretoken.append(pair_to_merge[0] + pair_to_merge[1])
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1

    return tuple(new_pretoken)


def pretoken2pairs(pretoken: Pretoken) -> Iterator[VocabPair]:
    """Given a pretoken, return a set of all byte pairs in the pretoken."""
    return zip(pretoken[:-1], pretoken[1:])


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


def pretokenize(input_path: str, special_tokens: list[str]) -> Pretoken:
    """Pretokenize text into bytes."""

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
    return tuple(text.encode("utf-8"))
