from data_models import ModelConfig


def embedding_flops():
    return 0 # there is no multiplication in the embedding layer, just lookup

def output_flops(d_model, vocab_size, seq):
    return 2 * vocab_size * d_model * seq # multiply by embedding matrix

def rms_norm_flops(d_model, seq):
    norm = 2 * d_model + 3 # elementwise square, sum elements, divide by scalar, add epsilon, take sqrt
    scale = 2 * d_model # divide by norm and multiply by scale
    return (norm + scale) * seq # do this for each elt in sequence

def attention_flops(d_model, num_heads, d_qk, d_v, seq):
    k_proj = (2 * d_qk * d_model * 1) * seq * num_heads 
    q_proj = (2 * d_qk * d_model * 1) * seq * num_heads 
    v_proj = (2 * d_v * d_model * 1) * seq * num_heads 
    # avoid double-counting the output projection
    # o_proj = (2 * d_v * d_model * 1) * seq * num_heads 
    qk_multiply = (2 * d_qk * seq * seq) * num_heads
    softmax = 3 * seq # exponantiate, sum, divide for each elt in the sequence
    v_multiply = (2 * seq * seq * d_v) * num_heads
    o_proj = (2 * seq * d_v * d_model) * num_heads
    return k_proj + q_proj + v_proj + qk_multiply + softmax + v_multiply + o_proj

def mlp_flops(d_model, d_ff, gated, seq):
    up_flops = (2 * d_model * d_ff * 1) * seq
    if gated:
        up_flops *= 2
    down_flops = (2 * d_ff * d_model * 1) * seq
    return up_flops + down_flops

def transformer_block_flops(d_model, num_heads, d_ff, d_qk, d_v, gated_mlp, seq):
    norm1_flops = rms_norm_flops(d_model, seq)
    print(f"First norm flops: {norm1_flops:,}")
    
    attn_flops = attention_flops(d_model, num_heads, d_qk, d_v, seq)
    print(f"Attention flops: {attn_flops:,}")
    
    norm2_flops = rms_norm_flops(d_model, seq)
    print(f"Second norm flops: {norm2_flops:,}")
    
    mlp_layer_flops = mlp_flops(d_model, d_ff, gated_mlp, seq)
    print(f"MLP flops: {mlp_layer_flops:,}")
    
    return norm1_flops + attn_flops + norm2_flops + mlp_layer_flops

def transformer_flops(config: ModelConfig, seq: int):
    embedding = embedding_flops()
    print(f"Embedding flops: {embedding:,}")

    # Calculate single transformer block flops
    single_block_flops = transformer_block_flops(
        config.d_model,
        config.num_heads,
        config.d_ff,
        config.d_qk,
        config.d_v,
        config.gated_mlp,
        seq
    )
    print(f"Single transformer block flops: {single_block_flops:,}")

    transformer_blocks = config.num_layers * single_block_flops
    print(f"Total transformer block flops: {transformer_blocks:,}")

    final_rms_norm = rms_norm_flops(config.d_model, seq)
    print(f"Final RMS norm flops: {final_rms_norm:,}")

    # In many implementations, embedding and output weights are tied
    if config.tied_weights:
        output = 0  # No additional flops if weights are tied
        print(f"Output flops (tied weights): {output:,}")
    else:
        output = output_flops(config.d_model, config.vocab_size, config.max_seq_len)
        print(f"Output flops (separate): {output:,}")

    return embedding + output + final_rms_norm + transformer_blocks
