# Analytical Expressions for Transformer FLOP Counts

This document provides analytical expressions for how different components of a Transformer model's FLOP count depend on model configuration parameters, derived from the functions in `flop_count.py`.

## Model Configuration Parameters

- `d_model`: Model dimension (hidden size)
- `d_ff`: Feed-forward dimension 
- `num_heads`: Number of attention heads
- `d_qk`: Query/key dimension per head
- `d_v`: Value dimension per head  
- `seq`: Sequence length
- `num_layers`: Number of transformer layers
- `vocab_size`: Vocabulary size
- `gated`: Whether MLP uses gating (boolean)

## Component-wise FLOP Expressions

### 1. MLP (Multi-Layer Perceptron) FLOPs

**Per Layer:**
```
MLP_flops = up_flops + down_flops

where:
- up_flops = 2 × d_model × d_ff × seq × (2 if gated else 1)
- down_flops = 2 × d_ff × d_model × seq

For standard (non-gated) MLP:
MLP_flops = 2 × d_model × d_ff × seq + 2 × d_ff × d_model × seq
          = 4 × d_model × d_ff × seq

For gated MLP:
MLP_flops = 2 × (2 × d_model × d_ff × seq) + 2 × d_ff × d_model × seq
          = 6 × d_model × d_ff × seq
```

**Total MLP FLOPs (all layers):**
```
Total_MLP_flops = num_layers × MLP_flops

For standard MLP:
Total_MLP_flops = 4 × num_layers × d_model × d_ff × seq

For gated MLP:
Total_MLP_flops = 6 × num_layers × d_model × d_ff × seq
```

**Scaling Analysis:**
- **Linear in:** `num_layers`, `seq`
- **Quadratic in:** `d_model` (since `d_ff` typically scales with `d_model`)
- **For GPT-2:** `d_ff = 4 × d_model`, so `MLP_flops ∝ d_model²`

### 2. Attention FLOPs

**Per Layer (corrected - removing double counting):**
```
Attention_flops = projection_flops + computation_flops

where:
projection_flops = k_proj + q_proj + v_proj + o_proj
                 = (2 × d_qk × d_model × seq × num_heads) × 2  # Q, K projections
                 + (2 × d_v × d_model × seq × num_heads) × 2   # V, O projections
                 = 4 × d_qk × d_model × seq × num_heads + 4 × d_v × d_model × seq × num_heads

computation_flops = qk_multiply + softmax + v_multiply
                  = (2 × d_qk × seq² × num_heads) +
                    (3 × seq) +
                    (2 × seq² × d_v × num_heads)

Note: The original flop_count.py double-counts the output projection (both o_proj and o_multiply).
The corrected version above removes this double counting.

For typical case where d_qk = d_v = d_model / num_heads:
projection_flops = 8 × (d_model/num_heads) × d_model × seq × num_heads
                 = 8 × d_model² × seq

computation_flops ≈ 2 × (d_model/num_heads) × seq² × num_heads + 
                    3 × seq
                  = 2 × d_model × seq² + 3 × seq
                  = 4 × d_model × seq² + 3 × seq

Total per layer:
Attention_flops = 8 × d_model² × seq + 4 × d_model × seq² + 3 × seq
                = 10 × d_model² × seq + 4 × d_model × seq² + 3 × seq
```

**Total Attention FLOPs (all layers):**
```
Total_Attention_flops = num_layers × Attention_flops
                      = num_layers × (10 × d_model² × seq + 4 × d_model × seq² + 3 × seq)
```

**Scaling Analysis:**
- **Linear in:** `num_layers`
- **Quadratic in:** `d_model` (dominant term: `10 × d_model² × seq`)
- **Quadratic in:** `seq` (term: `4 × d_model × seq²`)
- **Mixed scaling:** `d_model² × seq` and `d_model × seq²`

### 3. Other Components

**RMS Normalization (per application):**
```
RMSNorm_flops = (2 × d_model + 3 + 2 × d_model) × seq
              = (4 × d_model + 3) × seq
              ≈ 4 × d_model × seq  (for large d_model)
```

**Output Layer:**
```
Output_flops = 2 × vocab_size × d_model × seq
```

**Embedding:**
```
Embedding_flops = 0  (lookup operation, no multiplication)
```

## Complete Model FLOP Expression

**For a complete Transformer model:**
```
Total_flops = Embedding + Total_Attention + Total_MLP + Total_Norm + Output

where:
- Embedding = 0
- Total_Attention = num_layers × (10 × d_model² × seq + 4 × d_model × seq² + 3 × seq)
- Total_MLP = 4 × num_layers × d_model × d_ff × seq  (standard MLP)
- Total_Norm = (2 × num_layers + 1) × 4 × d_model × seq  (2 per layer + 1 final)
- Output = 2 × vocab_size × d_model × seq  (if not tied weights)
```

**For GPT-2 models (d_ff = 4 × d_model, tied weights):**
```
Total_flops ≈ num_layers × (10 × d_model² × seq + 4 × d_model × seq² + 16 × d_model² × seq) +
              (2 × num_layers + 1) × 4 × d_model × seq

            = num_layers × (26 × d_model² × seq + 4 × d_model × seq²) +
              (8 × num_layers + 4) × d_model × seq

Dominant terms:
Total_flops ≈ 26 × num_layers × d_model² × seq + 4 × num_layers × d_model × seq²
```

## Scaling Relationships

### MLP vs Attention Scaling

**MLP dominance condition:**
MLP FLOPs > Attention FLOPs when:
```
16 × d_model² × seq > 10 × d_model² × seq + 4 × d_model × seq²

Simplifying:
6 × d_model² × seq > 4 × d_model × seq²
6 × d_model > 4 × seq
d_model > (2/3) × seq

For seq = 1024: d_model > 683
For seq = 2048: d_model > 1365
```

**Key Insights:**
1. **MLP scales as O(d_model²)** when d_ff ∝ d_model
2. **Attention scales as O(d_model² + d_model × seq²)**
3. **For typical sequence lengths and large models, MLP dominates**
4. **As seq increases, attention becomes more significant**
5. **The crossover point depends on the ratio d_model/seq**

## Practical Implications

1. **Memory Optimization:** Focus on MLP layers for large models
2. **Compute Optimization:** MLP efficiency improvements have outsized impact
3. **Sequence Length:** Longer sequences make attention relatively more expensive
4. **Model Scaling:** Doubling d_model roughly quadruples both MLP and attention FLOPs
