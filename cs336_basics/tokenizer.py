import regex
from collections.abc import Iterable, Iterator  # Iterable has an __iter__ method that returns an iterator

from .common import deserialize_merges, deserialize_vocab, pretoken2pairs, perform_merge


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.bytes_to_int: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        # Join special tokens with | directly as bytes
        if special_tokens is not None:
            self.special_token_bytes = b"|".join([special_token.encode("utf-8") for special_token in special_tokens])
            self.special_token_regex = regex.compile(self.special_token_bytes)

        # GPT-2 regex pattern as bytes
        self.pretoken_regex = regex.compile(
            rb"""'(?:[sdmt]|ll|ve|re)|\ ?\p{L}+|\ ?\p{N}+|\ ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @classmethod
    def from_file(cls, vocab_file: str, merges_file: str, special_tokens: list[str] | None = None):
        vocab = deserialize_vocab(vocab_file)
        merges = deserialize_merges(merges_file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token ids.
        """
        if text == "":
            return []
        return list(self.encode_iterable([text]))

    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        """
        Encode a string into a list of token ids.
        """
        for text_string in text:
            # Handle special tokens first
            if self.special_tokens is not None:
                # Split text by special tokens
                parts = self._split_by_special_tokens(text_string)
                for part, is_special in parts:
                    if is_special:
                        # Yield the special token ID
                        special_token_bytes = part.encode("utf-8")
                        yield self.bytes_to_int[special_token_bytes]
                    else:
                        # Process regular text with BPE
                        yield from self._encode_regular_text(part)
            else:
                # No special tokens, process normally
                yield from self._encode_regular_text(text_string)

    def _split_by_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """Split text by special tokens, returning (text, is_special) tuples."""
        if not self.special_tokens:
            return [(text, False)]

        parts = []
        remaining = text

        while remaining:
            # Find the earliest position where any special token starts
            earliest_pos = len(remaining)

            for special_token in self.special_tokens:
                pos = remaining.find(special_token)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos

            if earliest_pos == len(remaining):
                # No more special tokens, add remaining text
                if remaining:
                    parts.append((remaining, False))
                break

            # Add text before special token
            if earliest_pos > 0:
                parts.append((remaining[:earliest_pos], False))

            # At earliest_pos, find the longest special token that matches
            longest_match = None
            for special_token in self.special_tokens:
                if remaining[earliest_pos:].startswith(special_token):
                    if longest_match is None or len(special_token) > len(longest_match):
                        longest_match = special_token

            if longest_match is None:
                # This shouldn't happen, but handle it gracefully
                parts.append((remaining[earliest_pos], False))
                remaining = remaining[earliest_pos + 1:]
            else:
                # Add the longest matching special token
                parts.append((longest_match, True))
                # Continue with remaining text
                remaining = remaining[earliest_pos + len(longest_match):]

        return parts

    def _encode_regular_text(self, text: str) -> Iterator[int]:
        """Encode regular (non-special-token) text with BPE."""
        if not text:
            return

        pretokens: list[bytes] = self.pretoken_regex.findall(text.encode("utf-8"))
        for pretoken_bytes in pretokens:
            pretoken: tuple[bytes, ...] = tuple([bytes([b]) for b in pretoken_bytes])

            # Apply BPE merges greedily until no more merges can be applied
            while True:
                pairs = list(pretoken2pairs(pretoken))
                if not pairs:
                    break

                # Find the highest priority merge (earliest in merge list) that can be applied
                merge_to_apply = None
                for merge in self.merges:
                    if merge in pairs:
                        merge_to_apply = merge
                        break

                # If no merge can be applied, we're done
                if merge_to_apply is None:
                    break

                # Apply the merge
                pretoken = perform_merge(pretoken, merge_to_apply)

            for merged_bytes in pretoken:
                yield self.bytes_to_int[merged_bytes]

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token ids into a string.
        """
        return b"".join([self.vocab[token] for token in tokens]).decode("utf-8", errors="replace")
