import torch as th
from jaxtyping import Float

def softmax(x: Float[th.Tensor, "..."], dim: int = -1) -> Float[th.Tensor, "..."]:
    """Compute the softmax of the input tensor."""
    # subtract the max for numerical stability
    x_max = th.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    x_exp = th.exp(x)
    x_sum = th.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum
    