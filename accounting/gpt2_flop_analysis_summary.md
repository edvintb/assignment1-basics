# GPT-2 FLOP Distribution Analysis

## Model Configurations

Based on the existing GPT-2 XL configuration, I created model configs for the standard GPT-2 variants:

### GPT-2 Small
- **Layers**: 12
- **d_model**: 768  
- **Heads**: 12
- **d_ff**: 3,072 (4 × d_model)
- **d_qk/d_v**: 64 (768 ÷ 12)
- **Tied weights**: True
- **Total FLOPs**: 227,175,152,640

### GPT-2 Medium  
- **Layers**: 24
- **d_model**: 1,024
- **Heads**: 16
- **d_ff**: 4,096 (4 × d_model)
- **d_qk/d_v**: 64 (1024 ÷ 16)
- **Tied weights**: True
- **Total FLOPs**: 773,299,858,432

### GPT-2 Large
- **Layers**: 36
- **d_model**: 1,280
- **Heads**: 20  
- **d_ff**: 5,120 (4 × d_model)
- **d_qk/d_v**: 64 (1280 ÷ 20)
- **Tied weights**: True
- **Total FLOPs**: 1,764,004,011,008

### GPT-2 XL
- **Layers**: 48
- **d_model**: 1,600
- **Heads**: 25
- **d_ff**: 6,400 (4 × d_model)  
- **d_qk/d_v**: 64 (1600 ÷ 25)
- **Tied weights**: False
- **Total FLOPs**: 3,758,997,949,440

## FLOP Distribution Analysis Results

### Key Findings

**1. MLP Component Dominance Increases with Model Size**
- GPT-2 Small: 51.0%
- GPT-2 Medium: 53.3%
- GPT-2 Large: 54.8%
- GPT-2 XL: 53.6%

The MLP component consistently takes up the majority of FLOPs and this fraction generally increases with model size (with a slight decrease for XL due to the untied output layer).

**2. Attention Component Fraction Decreases with Model Size**
- GPT-2 Small: 48.9%
- GPT-2 Medium: 46.7%
- GPT-2 Large: 45.2%
- GPT-2 XL: 42.0%

As models get larger, attention takes up a smaller proportion of the total computation, dropping from nearly 49% in Small to 42% in XL.

**3. Output Layer Impact**
- GPT-2 Small/Medium/Large: 0.0% (tied weights)
- GPT-2 XL: 4.4% (separate output layer)

The output layer only contributes significantly when weights are not tied, as in GPT-2 XL.

**4. Embedding and Normalization**
- All models: ~0.0% for both components
- These components are negligible compared to attention and MLP layers

## Analysis Implications

### Computational Scaling Patterns

1. **MLP Scaling**: The MLP layers scale as O(d_model × d_ff × seq_len × num_layers). Since d_ff = 4 × d_model, this scales as O(d_model² × seq_len × num_layers).

2. **Attention Scaling**: Attention scales as O(d_model × seq_len × num_heads × (d_qk + d_v) × num_layers + seq_len² × num_heads × num_layers). The quadratic seq_len term becomes more significant for longer sequences.

3. **Why MLP Dominates**: For the sequence length used (1024), the O(d_model²) scaling of MLP dominates over the O(seq_len²) scaling of attention, especially as d_model increases faster than seq_len.

### Optimization Implications

- **Memory Optimization**: Focus on MLP layers for memory reduction techniques
- **Compute Optimization**: MLP layers are the primary target for computational optimizations
- **Architecture Design**: The increasing MLP dominance suggests that MLP efficiency improvements have outsized impact on larger models

## Technical Notes

- Analysis performed with sequence length = 1024
- All calculations use the existing FLOP counting functions from the codebase
- GPT-2 Small/Medium/Large use tied weights (standard practice)
- GPT-2 XL uses separate output weights (as in original config)
- Embedding FLOPs are 0 (lookup operation, no multiplication)
- Normalization FLOPs are negligible compared to linear layers
