
from data_models import ModelConfig
from flop_count import (
    embedding_flops, output_flops, rms_norm_flops,
    attention_flops, mlp_flops, transformer_flops
)
from typing import NamedTuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be skipped.")


class FlopAnalysisResult(NamedTuple):
    """Results from FLOP distribution analysis for a Transformer model."""
    embedding_flops: int
    norm_flops: int
    attention_flops: int
    mlp_flops: int
    output_flops: int
    total_flops: int
    embedding_fraction: float
    norm_fraction: float
    attention_fraction: float
    mlp_fraction: float
    output_fraction: float


# GPT-2 model configurations
gpt2_small = ModelConfig(
    num_layers=12,
    num_heads=12,
    d_ff=3072,  # 4 * d_model for GPT-2
    d_model=768,
    d_qk=768 // 12,
    d_v=768 // 12,
    vocab_size=50_257,
    tied_weights=True,  # GPT-2 uses tied weights
    gated_mlp=False,
)

gpt2_medium = ModelConfig(
    num_layers=24,
    num_heads=16,
    d_ff=4096,  # 4 * d_model for GPT-2
    d_model=1024,
    d_qk=1024 // 16,
    d_v=1024 // 16,
    vocab_size=50_257,
    tied_weights=True,
    gated_mlp=False,
)

gpt2_large = ModelConfig(
    num_layers=36,
    num_heads=20,
    d_ff=5120,  # 4 * d_model for GPT-2
    d_model=1280,
    d_qk=1280 // 20,
    d_v=1280 // 20,
    vocab_size=50_257,
    tied_weights=True,
    gated_mlp=False,
)

gpt2_xl = ModelConfig(
    num_layers=48,
    num_heads=25,
    d_ff=6400,
    d_model=1600,
    d_qk=1600 // 25,
    d_v=1600 // 25,
    vocab_size=50_257,
    tied_weights=False,
    gated_mlp=False,
)


def analyze_flop_distribution(config: ModelConfig, seq: int = 1024) -> FlopAnalysisResult:
    """
    Analyze the FLOP distribution across different components of a Transformer model.
    Returns a FlopAnalysisResult with FLOP counts and fractions for each component.
    """
    # Calculate FLOPs for each component
    embedding = embedding_flops()

    # Single transformer block components
    norm1_flops = rms_norm_flops(config.d_model, seq)
    attn_flops = attention_flops(config.d_model, config.num_heads, config.d_qk, config.d_v, seq)
    norm2_flops = rms_norm_flops(config.d_model, seq)
    mlp_layer_flops = mlp_flops(config.d_model, config.d_ff, config.gated_mlp, seq)

    # Total for all transformer blocks
    total_norm_flops = (norm1_flops + norm2_flops) * config.num_layers
    total_attn_flops = attn_flops * config.num_layers
    total_mlp_flops = mlp_layer_flops * config.num_layers

    # Final components
    final_rms_norm = rms_norm_flops(config.d_model, seq)

    output = output_flops(config.d_model, config.vocab_size, seq)

    # Total FLOPs
    total_flops = embedding + total_norm_flops + total_attn_flops + total_mlp_flops + final_rms_norm + output

    # Return as named tuple
    return FlopAnalysisResult(
        embedding_flops=embedding,
        norm_flops=total_norm_flops + final_rms_norm,
        attention_flops=total_attn_flops,
        mlp_flops=total_mlp_flops,
        output_flops=output,
        total_flops=total_flops,
        embedding_fraction=embedding / total_flops,
        norm_fraction=(total_norm_flops + final_rms_norm) / total_flops,
        attention_fraction=total_attn_flops / total_flops,
        mlp_fraction=total_mlp_flops / total_flops,
        output_fraction=output / total_flops,
    )


def compare_models(seq_length):
    """
    Compare FLOP distributions across different GPT-2 model sizes.
    """
    models = {
        'GPT-2 Small': gpt2_small,
        'GPT-2 Medium': gpt2_medium,
        'GPT-2 Large': gpt2_large,
        'GPT-2 XL': gpt2_xl,
    }

    results = {}

    print("=" * 80)
    print("FLOP DISTRIBUTION ANALYSIS FOR GPT-2 MODELS")
    print("=" * 80)
    print(f"Sequence length: {seq_length}")
    print()

    for model_name, config in models.items():
        print(f"\n{model_name}:")
        print(f"  Layers: {config.num_layers}, d_model: {config.d_model}, heads: {config.num_heads}")
        print(f"  d_ff: {config.d_ff}, tied_weights: {config.tied_weights}")

        analysis = analyze_flop_distribution(config, seq_length)
        results[model_name] = analysis

        print(f"  Total FLOPs: {analysis.total_flops:,}")
        print(f"  Embedding: {analysis.embedding_fraction:.1%}")
        print(f"  Normalization: {analysis.norm_fraction:.1%}")
        print(f"  Attention: {analysis.attention_fraction:.1%}")
        print(f"  MLP: {analysis.mlp_fraction:.1%}")
        print(f"  Output: {analysis.output_fraction:.1%}")

    return results


def plot_flop_distribution(results, seq_length):
    """
    Create a well-annotated plot showing FLOP distribution across model sizes.
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Skipping visualization.")
        return

    models = list(results.keys())
    components = ['Embedding', 'Normalization', 'Attention', 'MLP', 'Output']

    # Extract fractions for each component
    embedding_fractions = [results[model].embedding_fraction for model in models]
    norm_fractions = [results[model].norm_fraction for model in models]
    attention_fractions = [results[model].attention_fraction for model in models]
    mlp_fractions = [results[model].mlp_fraction for model in models]
    output_fractions = [results[model].output_fraction for model in models]

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Stacked bar chart
    width = 0.6
    x = np.arange(len(models))

    # Colors for each component
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # Create stacked bars
    bottom = np.zeros(len(models))
    bars = []

    fractions_data = [embedding_fractions, norm_fractions, attention_fractions, mlp_fractions, output_fractions]

    for i, (component, fractions, color) in enumerate(zip(components, fractions_data, colors)):
        bars.append(ax1.bar(x, fractions, width, bottom=bottom, label=component, color=color, alpha=0.8))
        bottom += fractions

    ax1.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraction of Total FLOPs', fontsize=12, fontweight='bold')
    ax1.set_title('FLOP Distribution Across GPT-2 Model Sizes\n(Stacked Bar Chart)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)

    # Add percentage labels on bars
    for i, model in enumerate(models):
        y_pos = 0
        for j, (component, fractions) in enumerate(zip(components, fractions_data)):
            if fractions[i] > 0.02:  # Only show labels for components > 2%
                ax1.text(i, y_pos + fractions[i]/2, f'{fractions[i]:.1%}',
                        ha='center', va='center', fontweight='bold', fontsize=9)
            y_pos += fractions[i]

    # Line plot showing trends
    for i, (component, fractions, color) in enumerate(zip(components, fractions_data, colors)):
        ax2.plot(models, fractions, marker='o', linewidth=2.5, markersize=8,
                label=component, color=color, alpha=0.8)

    ax2.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction of Total FLOPs', fontsize=12, fontweight='bold')
    ax2.set_title('FLOP Distribution Trends Across GPT-2 Model Sizes\n(Line Plot)', fontsize=14, fontweight='bold')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(max(fractions) for fractions in fractions_data) * 1.1)

    # Rotate x-axis labels
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'gpt2_flop_distribution_seq={seq_length}.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'gpt2_flop_distribution_seq={seq_length}.png'")
    # plt.show()  # Commented out to avoid hanging in headless environments

    # Print analysis summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nKey Observations:")
    print("1. MLP Component:")
    mlp_trend = [results[model].mlp_fraction for model in models]
    print(f"   - Small: {mlp_trend[0]:.1%}, Medium: {mlp_trend[1]:.1%}, Large: {mlp_trend[2]:.1%}, XL: {mlp_trend[3]:.1%}")
    if mlp_trend[-1] > mlp_trend[0]:
        print("   - MLP fraction INCREASES with model size")
    else:
        print("   - MLP fraction DECREASES with model size")

    print("\n2. Attention Component:")
    attn_trend = [results[model].attention_fraction for model in models]
    print(f"   - Small: {attn_trend[0]:.1%}, Medium: {attn_trend[1]:.1%}, Large: {attn_trend[2]:.1%}, XL: {attn_trend[3]:.1%}")
    if attn_trend[-1] > attn_trend[0]:
        print("   - Attention fraction INCREASES with model size")
    else:
        print("   - Attention fraction DECREASES with model size")

    print("\n3. Output Layer:")
    output_trend = [results[model].output_fraction for model in models]
    print(f"   - Small: {output_trend[0]:.1%}, Medium: {output_trend[1]:.1%}, Large: {output_trend[2]:.1%}, XL: {output_trend[3]:.1%}")
    print("   - Output layer fraction DECREASES significantly with model size")
    print("   - This is because vocab_size stays constant while d_model grows")


if __name__ == "__main__":
    # Run the analysis
    seq_length = 16384
    results = compare_models(seq_length=seq_length)
    plot_flop_distribution(results, seq_length=seq_length)