from collections.abc import Iterable
import math
import torch as th
from einops import einsum, reduce
from jaxtyping import Float, Int

import math

def softmax(x: Float[th.Tensor, "..."], dim: int = -1) -> Float[th.Tensor, "..."]:
    """Compute the softmax of the input tensor."""
    # subtract the max for numerical stability
    x_max = th.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    x_exp = th.exp(x)
    x_sum = th.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum

def scaled_dotproduct_attention(
    Q: Float[th.Tensor, '... seq num_heads d_k'],
    K: Float[th.Tensor, '... seq num_heads d_k'],
    V: Float[th.Tensor, '... seq num_heads d_v'],
    mask: Float[th.Tensor, '... seq seq'] | None = None,
):
    # normlized dot product between keys and values
    logits = einsum(Q, K, '... seq_q num_heads d_k, ... seq_k num_heads d_k -> ... num_heads seq_q seq_k') / math.sqrt(Q.shape[-1])

    if mask is not None:
        # True means "pay attention to this"
        logits = th.where(mask, logits, float('-inf'))

    weights = softmax(logits, dim=-1) # normalize over keys for each query

    # d_v vectors should be weighted by the attention weights, so contract over normalized key dim
    return einsum(weights, V, '... num_heads seq_q seq_k, ... seq_k num_heads d_v -> ... seq_q num_heads d_v')

def silu(x):
    return x * th.sigmoid(x)

def cross_entropy(logits: Float[th.Tensor, '... vocab_size'], targets: Int[th.Tensor, '...']) -> Float:
    """Given logits and targets, compute cross entropoy loss"""
    # subtract for numerical stability -- underflow goes to 0, which is fine
    logits = logits - th.max(logits, dim=-1, keepdim=True)[0]

    # We need sum for normalization
    norms = reduce(th.exp(logits), '... vocab_size -> ...', 'sum')

    # log cancels exponential in the numerator
    logprobs = th.gather(logits, dim=-1, index=targets.unsqueeze(-1).long()).squeeze(-1) - th.log(norms)

    # return average negative log prob
    return -th.mean(logprobs)

def gradient_clipping(
    params: Iterable[th.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    assert max_l2_norm > 0, "Max L2 norm must be positive"
    norm = 0
    for param in params:
        if param.grad is None:
            continue

        norm += th.sum(th.pow(param.grad, 2))

    norm = math.sqrt(norm)

    if norm < max_l2_norm:
        return

    for param in params:
        if param.grad is None:
            continue

        param.grad.data *= (max_l2_norm / (norm + eps))
