#!/usr/bin/env python3
"""
Verification script to validate analytical expressions against actual FLOP counting functions.
"""

from flop_count import attention_flops, mlp_flops, rms_norm_flops, output_flops
from data_models import ModelConfig


def analytical_mlp_flops(d_model: int, d_ff: int, gated: bool, seq: int) -> int:
    """Analytical expression for MLP FLOPs per layer."""
    if gated:
        return 6 * d_model * d_ff * seq
    else:
        return 4 * d_model * d_ff * seq


def analytical_attention_flops_original(d_model: int, num_heads: int, d_qk: int, d_v: int, seq: int) -> int:
    """
    Analytical expression matching the original flop_count.py (with double counting).
    """
    # Projection FLOPs: Q, K, V, O projections (as in original code)
    k_proj = (2 * d_qk * d_model * 1) * seq * num_heads
    q_proj = (2 * d_qk * d_model * 1) * seq * num_heads
    v_proj = (2 * d_v * d_model * 1) * seq * num_heads
    o_proj = (2 * d_v * d_model * 1) * seq * num_heads  # Double-counted!

    # Computation FLOPs: QK^T, softmax, attention*V, output multiply
    qk_multiply = (2 * d_qk * seq * seq) * num_heads
    softmax = 3 * seq  # Note: this is per sequence, not per head
    v_multiply = (2 * seq * seq * d_v) * num_heads
    o_multiply = (2 * seq * d_v * d_model) * num_heads  # Same as o_proj!

    return k_proj + q_proj + v_proj + o_proj + qk_multiply + softmax + v_multiply + o_multiply


def analytical_attention_flops_corrected(d_model: int, num_heads: int, d_qk: int, d_v: int, seq: int) -> int:
    """
    Corrected analytical expression for attention FLOPs per layer (no double counting).
    """
    # Projection FLOPs: Q, K, V projections only
    k_proj = (2 * d_qk * d_model * 1) * seq * num_heads
    q_proj = (2 * d_qk * d_model * 1) * seq * num_heads
    v_proj = (2 * d_v * d_model * 1) * seq * num_heads
    # No o_proj here - it's counted in o_multiply

    # Computation FLOPs: QK^T, softmax, attention*V, output multiply
    qk_multiply = (2 * d_qk * seq * seq) * num_heads
    softmax = 3 * seq  # Note: this is per sequence, not per head
    v_multiply = (2 * seq * seq * d_v) * num_heads
    o_multiply = (2 * seq * d_v * d_model) * num_heads  # Output projection

    return k_proj + q_proj + v_proj + qk_multiply + softmax + v_multiply + o_multiply


def analytical_rms_norm_flops(d_model: int, seq: int) -> int:
    """Analytical expression for RMS norm FLOPs."""
    return (4 * d_model + 3) * seq


def analytical_output_flops(d_model: int, vocab_size: int, seq: int) -> int:
    """Analytical expression for output layer FLOPs."""
    return 2 * vocab_size * d_model * seq


def verify_expressions():
    """Verify analytical expressions match the actual functions."""
    print("=" * 80)
    print("VERIFYING ANALYTICAL EXPRESSIONS")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        {"d_model": 768, "d_ff": 3072, "num_heads": 12, "seq": 1024, "vocab_size": 50257},
        {"d_model": 1024, "d_ff": 4096, "num_heads": 16, "seq": 2048, "vocab_size": 50257},
        {"d_model": 1600, "d_ff": 6400, "num_heads": 25, "seq": 512, "vocab_size": 50257},
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\nTest Configuration {i}:")
        print(f"  d_model={config['d_model']}, d_ff={config['d_ff']}, heads={config['num_heads']}")
        print(f"  seq={config['seq']}, vocab_size={config['vocab_size']}")
        
        d_qk = config['d_model'] // config['num_heads']
        d_v = config['d_model'] // config['num_heads']
        
        # Test MLP FLOPs
        actual_mlp = mlp_flops(config['d_model'], config['d_ff'], False, config['seq'])
        analytical_mlp = analytical_mlp_flops(config['d_model'], config['d_ff'], False, config['seq'])
        mlp_match = actual_mlp == analytical_mlp
        
        print(f"  MLP FLOPs:")
        print(f"    Actual:     {actual_mlp:,}")
        print(f"    Analytical: {analytical_mlp:,}")
        print(f"    Match: {mlp_match} ✓" if mlp_match else f"    Match: {mlp_match} ✗")
        
        # Test Attention FLOPs - Original (with double counting)
        actual_attn = attention_flops(config['d_model'], config['num_heads'], d_qk, d_v, config['seq'])
        analytical_attn_orig = analytical_attention_flops_original(config['d_model'], config['num_heads'], d_qk, d_v, config['seq'])
        analytical_attn_corr = analytical_attention_flops_corrected(config['d_model'], config['num_heads'], d_qk, d_v, config['seq'])

        orig_match = actual_attn == analytical_attn_orig

        print(f"  Attention FLOPs:")
        print(f"    Actual (original):     {actual_attn:,}")
        print(f"    Analytical (original): {analytical_attn_orig:,}")
        print(f"    Analytical (corrected):{analytical_attn_corr:,}")
        print(f"    Original Match: {orig_match} ✓" if orig_match else f"    Original Match: {orig_match} ✗")
        print(f"    Difference: {actual_attn - analytical_attn_corr:,} FLOPs (double-counted output proj)")
        
        # Test RMS Norm FLOPs
        actual_norm = rms_norm_flops(config['d_model'], config['seq'])
        analytical_norm = analytical_rms_norm_flops(config['d_model'], config['seq'])
        norm_match = actual_norm == analytical_norm
        
        print(f"  RMS Norm FLOPs:")
        print(f"    Actual:     {actual_norm:,}")
        print(f"    Analytical: {analytical_norm:,}")
        print(f"    Match: {norm_match} ✓" if norm_match else f"    Match: {norm_match} ✗")
        
        # Test Output FLOPs
        actual_output = output_flops(config['d_model'], config['vocab_size'], config['seq'])
        analytical_output = analytical_output_flops(config['d_model'], config['vocab_size'], config['seq'])
        output_match = actual_output == analytical_output
        
        print(f"  Output FLOPs:")
        print(f"    Actual:     {actual_output:,}")
        print(f"    Analytical: {analytical_output:,}")
        print(f"    Match: {output_match} ✓" if output_match else f"    Match: {output_match} ✗")


def analyze_scaling():
    """Analyze scaling behavior of MLP vs Attention."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS: MLP vs ATTENTION")
    print("=" * 80)
    
    base_config = {"d_model": 768, "d_ff": 3072, "num_heads": 12, "seq": 1024}
    
    print("\nScaling with d_model (keeping d_ff = 4 * d_model):")
    print("d_model | MLP FLOPs    | Attention FLOPs | MLP/Attention Ratio")
    print("-" * 65)
    
    for d_model in [512, 768, 1024, 1280, 1600, 2048]:
        d_ff = 4 * d_model
        num_heads = 12  # Keep constant for simplicity
        d_qk = d_v = d_model // num_heads
        seq = 1024
        
        mlp_flops_val = analytical_mlp_flops(d_model, d_ff, False, seq)
        attn_flops_val = analytical_attention_flops_corrected(d_model, num_heads, d_qk, d_v, seq)
        ratio = mlp_flops_val / attn_flops_val
        
        print(f"{d_model:7d} | {mlp_flops_val:11,} | {attn_flops_val:14,} | {ratio:18.2f}")
    
    print("\nScaling with sequence length:")
    print("seq_len | MLP FLOPs    | Attention FLOPs | MLP/Attention Ratio")
    print("-" * 65)
    
    d_model = 1024
    d_ff = 4096
    num_heads = 16
    d_qk = d_v = d_model // num_heads
    
    for seq in [256, 512, 1024, 2048, 4096, 8192]:
        mlp_flops_val = analytical_mlp_flops(d_model, d_ff, False, seq)
        attn_flops_val = analytical_attention_flops_corrected(d_model, num_heads, d_qk, d_v, seq)
        ratio = mlp_flops_val / attn_flops_val
        
        print(f"{seq:7d} | {mlp_flops_val:11,} | {attn_flops_val:14,} | {ratio:18.2f}")


if __name__ == "__main__":
    verify_expressions()
    analyze_scaling()
