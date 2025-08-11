import json

VocabElt = bytes
Pretoken = tuple[VocabElt, ...]
VocabPair = tuple[VocabElt, VocabElt]


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
