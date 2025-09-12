import torch as th
from einops import einsum, reduce, rearrange
from jaxtyping import Int, Float

from cs336_basics.functions import silu, scaled_dotproduct_attention

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
        a2 = silu(a1) * self.w3(x)
        return self.w2(a2)

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
        self.q_proj = Linear(self.d_k * num_heads, d_model)
        self.k_proj = Linear(self.d_k * num_heads, d_model)
        self.v_proj = Linear(self.d_v * num_heads, d_model)
        self.output_proj = Linear(d_model, num_heads * self.d_v)
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
            self.token_positions = token_positions
        else:
            self.rope = None
    
    def forward(self, in_features: Float[th.Tensor, "... seq d_model"]):
        # project queries, keys, and values
        q_heads = self.q_proj(in_features)
        q_heads = rearrange(q_heads, '... seq (num_heads d_k) -> ... num_heads seq d_k', d_k=self.d_k)
        k_heads = self.k_proj(in_features)
        k_heads = rearrange(k_heads, '... seq (num_heads d_k) -> ... num_heads seq d_k', d_k=self.d_k)
        v_heads = self.v_proj(in_features)
        v_heads = rearrange(v_heads, '... seq (num_heads d_v) -> ... num_heads seq d_v', d_v=self.d_v)

        # apply RoPE to queries and keys
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

        # create a causal mask
        causal_mask = th.tril(
            th.ones(q_heads.shape[-2], k_heads.shape[-2], dtype=th.bool, device=q_heads.device), diagonal=0
        )
        
        # compute attention
        attention_heads: Float[th.Tensor, '... num_heads seq d_v'] = scaled_dotproduct_attention(
            Q=q_heads,
            K=k_heads,
            V=v_heads,
            mask=causal_mask
        )

        # project back into model dimension
        return self.output_proj(
            rearrange(attention_heads,'... num_heads seq d_v -> ... seq (num_heads d_v)')
        )

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
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: Float[th.Tensor, "... seq d_model"]):
        y0 = self.ln1(x)      # rms norm x * w / sqrt(x**2 + eps)
        y0 = self.attn(y0)    # attention
        y0 += x               # residual connection 
        y1 = self.ln2(y0)     # rms norm x * w / sqrt(x**2 + eps)
        y1 = self.ffn(y1)     # feedforward
        y1 += y0              # residual connection 
        return y1


class TransformerLM(th.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        # we need an initial embedding layer
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        # we need transformer blocks
        self.layers = th.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                num_heads=num_heads,
                theta=rope_theta,
                max_seq_len=context_length,
                token_positions=None
            ) for _ in range(num_layers)
        ])

        # we need a final norm
        self.ln_final = RMSNorm(d_model)

        # we need an output layer
        self.lm_head = Linear(vocab_size, d_model)

    def forward(self, token_ids: Int[th.Tensor, 'batch sequence']):
        # embedd token ids
        x = self.token_embeddings(token_ids)

        # run through transformer blocks
        for layer in self.layers:
            x = layer(x)

        # normalize and project
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
