from typing import NamedTuple

class ModelConfig(NamedTuple):
    num_layers: int
    num_heads: int
    d_ff: int
    d_model: int
    d_qk: int
    d_v: int
    vocab_size: int
    tied_weights: bool
    gated_mlp: bool