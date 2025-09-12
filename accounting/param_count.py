from data_models import ModelConfig

def embedding_params(vocab_size, d_model):
    return vocab_size * d_model

def output_params(d_model, vocab_size):
    return vocab_size * d_model

def rms_norm_params(d_model):
    return d_model

def attention_params(d_model, num_heads, d_qk, d_v):
    k_proj = (num_heads * d_qk * d_model) 
    q_proj = (num_heads * d_qk * d_model)
    v_proj = (num_heads * d_v * d_model)
    o_proj = (num_heads * d_v * d_model)
    return k_proj + q_proj + v_proj + o_proj

def mlp_params(d_model, d_ff, gated):
    up_params = 2 * d_model * d_ff if gated else d_model * d_ff
    down_params = d_ff * d_model
    return up_params + down_params

def transformer_block_params(d_model, num_heads, d_ff, d_qk, d_v, gated_mlp):
    norm1_params = rms_norm_params(d_model)
    print(f"First norm params: {norm1_params:,}")
    
    attn_params = attention_params(d_model, num_heads, d_qk, d_v)
    print(f"Attention params: {attn_params:,}")
    
    norm2_params = rms_norm_params(d_model)
    print(f"Second norm params: {norm2_params:,}")
    
    mlp_layer_params = mlp_params(d_model, d_ff, gated_mlp)
    print(f"MLP params: {mlp_layer_params:,}")
    
    return norm1_params + attn_params + norm2_params + mlp_layer_params

def transformer_params(config: ModelConfig):
    embedding = embedding_params(config.vocab_size, config.d_model)
    print(f"Embedding params: {embedding:,}")

    # Calculate single transformer block params
    single_block_params = transformer_block_params(
        config.d_model, 
        config.num_heads, 
        config.d_ff, 
        config.d_qk, 
        config.d_v, 
        config.gated_mlp
    )
    print(f"Single transformer block params: {single_block_params:,}")

    transformer_blocks = config.num_layers * single_block_params
    print(f"Total transformer block params: {transformer_blocks:,}")

    final_rms_norm = rms_norm_params(config.d_model)
    print(f"Final RMS norm params: {final_rms_norm:,}")

    # In many implementations, embedding and output weights are tied
    if config.tied_weights:
        output = 0  # No additional parameters if weights are tied
        print(f"Output params (tied weights): {output:,}")
    else:
        output = output_params(config.d_model, config.vocab_size)
        print(f"Output params (separate): {output:,}")

    return embedding + output + final_rms_norm + transformer_blocks
