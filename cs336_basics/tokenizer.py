from collections.abc import Iterable, Iterator  # Iterable has an __iter__ method that returns an iterator

from .common import deserialize_merges, deserialize_vocab, pretoken2pairs, perform_merge, PRETOKEN_BYTES_REGEX


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # Join special tokens with | directly as bytes
        if special_tokens is not None:
            self.special_token_bytes = b"|".join([special_token.encode("utf-8") for special_token in special_tokens])
            self.special_token_regex = regex.compile(self.special_token_bytes)
        
        self.pretoken_regex = PRETOKEN_BYTES_REGEX


    @classmethod
    def from_file(cls, vocab_file: str, merges_file: str, special_tokens: list[str] | None = None):
        vocab = deserialize_vocab(vocab_file)
        merges = deserialize_merges(merges_file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        pass
