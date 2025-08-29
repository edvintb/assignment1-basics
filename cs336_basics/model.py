import torch as th
from einops import einsum, reduce, rearrange
from jaxtyping import Int, Float

import math

from cs336_basics.utils import softmax

class Linear(th.nn.Module):
    def __init__(self, out_features: int, in_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = th.nn.Parameter(
            th.empty(
                size=(out_features, in_features),
                device=device,
                dtype=dtype,
            ),
        )
        sigma_init = 2 / (out_features + in_features)
        th.nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0,
            std=sigma_init,
            a=-3 * sigma_init,
            b=3 * sigma_init,
        )
    
    def forward(self, x):
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')


class Embedding(th.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: th.device | None = None,
        dtype: th.dtype | None = None,
    ):
        super().__init__()
        self.weight = th.nn.Parameter(
            th.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype,
            )
        )
        th.nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=1,
            a=3,
            b=3,
        )

    def forward(self, token_ids: Int[th.Tensor, "B T"]) -> Float[th.Tensor, "B T D"]:
        return self.weight[token_ids]
        

class RMSNorm(th.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: th.device | None = None, dtype: th.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = th.nn.Parameter(
            th.empty(d_model, device=device, dtype=dtype)
        )
        th.nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=1,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        in_dtype = x.dtype
        x = x.to(th.float32)

        norm: Float[th.Tensor, 'batch sequence'] = th.sqrt(
            reduce(x**2, '... a -> ...', 'mean') + self.eps
        )
        result = x * self.weight / rearrange(norm, '... -> ... 1')

        return result.to(in_dtype)


class SwiGLU(th.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: th.device | None = None, dtype: th.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[th.Tensor, "... d_model"]):
        a1 = self.w1(x)
        silu = a1 * th.sigmoid(a1) * self.w3(x)
        return self.w2(silu)


class RotaryPositionalEmbedding(th.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: th.device | None = None):
        assert d_k % 2 == 0, "d_k must be even"
        super().__init__()
        # all the even values smaller than d_k
        theta_k: Float[th.Tensor, 'half_d'] = 1 / th.pow(theta, (th.arange(0, d_k, 2) / d_k))

        # all the positions within max sequence len
        i: Float[th.Tensor, 'seq'] = th.arange(start=0, end=max_seq_len, dtype=th.float32)

        # create angle for each seq_pos, vector_pos combination
        theta_ik = einsum(theta_k, i, 'half_d, seq -> half_d seq')

        # take sin and cos of all the angles...
        self.sin_theta_ik: Float[th.Tensor, 'half_d seq'] = th.sin(theta_ik)
        self.cos_theta_ik: Float[th.Tensor, 'half_d seq'] = th.cos(theta_ik)

        # make these cos and sin values part of state dict and model movement
        self.register_buffer('cos', self.cos_theta_ik, persistent=False)
        self.register_buffer('sin', self.sin_theta_ik, persistent=False)
    
    def forward(self, x: Float[th.Tensor, '... seq d_k'], token_positions: Int[th.Tensor, '... seq']) -> Float[th.Tensor, '... d_k']:
        # pick cos and sin values for all token positions in batch
        cos_vals = self.cos_theta_ik[:, token_positions]
        sin_vals = self.sin_theta_ik[:, token_positions]

        # arrange cos and sin vals into a 2x2 matrix
        rotation_matrices = rearrange(
            [cos_vals, -sin_vals, sin_vals, cos_vals],
            '(rows cols) half_d ... -> ... half_d rows cols',
            rows=2, cols=2
        )

        # split the x-vector into pairs
        x_pairs = rearrange(x, '... (half_d two) -> ... half_d two', two=2)

        # contract over the col dim to peform the rotation for each vector
        # we need the explictly named half_d dimension to align the tensors
        result = einsum(rotation_matrices, x_pairs, '... half_d i j, ... half_d j -> ... half_d i')

        # flatten the pairs into a single vector
        result = rearrange(result, '... half_d i -> ... (half_d i)')

        return result


class ScaledDotProductAttention(th.nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(
        self,
        Q: Float[th.Tensor, '... seq_q d_k'],
        K: Float[th.Tensor, '... seq_k d_k'],
        V: Float[th.Tensor, '... seq_v d_v'],
        mask: Float[th.Tensor, '... seq_q seq_k'] | None = None,
    ):
        # normlized dot product between keys and values
        logits = einsum(Q, K, '... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k') / math.sqrt(Q.shape[-1])

        if mask is not None:
            # True means "pay attention to this"
            logits = th.where(mask, logits, float('-inf'))

        weights = th.nn.functional.softmax(logits, dim=-1) # normalize over keys for each query

        # d_v vectors should be weighted by the attention weights, so contract over normalized key dim
        return einsum(weights, V, '... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v')

class MultiheadAttention(th.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: Int[th.Tensor, "... seq"] | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj_weight: Float[th.Tensor, "(num_heads d_k) d_model"] = th.nn.Parameter(
            th.empty(self.d_k * num_heads , d_model)
        )
        self.k_proj_weight: Float[th.Tensor, "(num_heads d_k) d_model"] = th.nn.Parameter(
            th.empty(self.d_k * num_heads , d_model)
        )
        self.v_proj_weight: Float[th.Tensor, "(num_heads d_v) d_model"] = th.nn.Parameter(
            th.empty(self.d_v * num_heads , d_model)
        )
        self.o_proj_weight: Float[th.Tensor, "d_model (num_heads d_v)"] = th.nn.Parameter(
            th.empty(d_model, self.d_v * num_heads)
        )
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            self.token_positions = token_positions  # Can be None, will be created dynamically in forward
        else:
            self.rope = None
    
    def forward(self, in_features: Float[th.Tensor, "... seq d_model"]):
        # project queries and keys
        q_heads = einsum(self.q_proj_weight, in_features,
            'd_hq d_model, ... seq d_model -> ... seq d_hq'
        )
        q_heads = rearrange(q_heads, '... seq (num_heads d_k) -> ... num_heads seq d_k', d_k=self.d_k)
        k_heads = einsum(self.k_proj_weight, in_features,
            'd_hk d_model, ... seq d_model -> ... seq d_hk'
        )
        k_heads = rearrange(k_heads, '... seq (num_heads d_k) -> ... num_heads seq d_k', d_k=self.d_k)

        # apply RoPE if we have it
        if self.rope is not None:
            if self.token_positions is None:
                seq_len = q_heads.shape[-2]  # Get sequence length from input
                # assume token positions are sequential if not provided
                token_positions = th.arange(seq_len, device=q_heads.device)
                # Expand to match batch dimensions using expand
                batch_shape = q_heads.shape[:-3]  # All dims except num_heads, seq, d_k
                token_positions = token_positions.expand(*batch_shape, seq_len)
            else:
                token_positions = self.token_positions
            q_heads = self.rope(q_heads, token_positions)
            k_heads = self.rope(k_heads, token_positions)

        # project values
        v_heads = einsum(self.v_proj_weight, in_features,
            'd_hv d_model, ... seq d_model -> ... seq d_hv'
        )
        v_heads = rearrange(v_heads, '... seq (num_heads d_v) -> ... num_heads seq d_v', d_v=self.d_v)

        # create a causal mask
        causal_mask = th.tril(
            th.ones(q_heads.shape[-2], k_heads.shape[-2], dtype=th.bool, device=q_heads.device), diagonal=0
        )
        
        # compute attention
        attention_heads: Float[th.Tensor, '... num_heads seq d_v'] = ScaledDotProductAttention()(
            Q=q_heads,
            K=k_heads,
            V=v_heads,
            mask=causal_mask
        )

        # project back into d_model
        o_proj_weight = rearrange(self.o_proj_weight, 'd_model (num_heads d_v) -> num_heads d_model d_v', d_v=self.d_v)
        result = einsum(attention_heads, o_proj_weight,
            '... num_heads seq d_v, num_heads d_model d_v -> ... seq d_model'
        )

        return result


class TransformerBlock(th.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: Int[th.Tensor, "... seq"] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.attn = MultiheadAttention(d_model, num_heads, theta, max_seq_len, token_positions)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm0 = RMSNorm(d_model)
        self.norm1 = RMSNorm(d_model)

    def forward(self, x: Float[th.Tensor, "... seq d_model"]):
        y0 = self.norm0(x)    # rms norm x * w / sqrt(x**2 + eps)
        y0 = self.attn(y0)    # attention
        y0 += x               # residual connection 
        y1 = self.norm1(y0)   # rms norm x * w / sqrt(x**2 + eps)
        y1 = self.ffn(y1)     # feedforward
        y1 += y0              # residual connection 
        return y1

if __name__ == "__main__":
    linear = Linear(
        in_features=5,
        out_features=10,
    )
    print(f"linear: {linear}")
    print(f"state dict: {linear.state_dict()}")
    print("keys:")
    for key in linear.state_dict().keys():
        print(key)
