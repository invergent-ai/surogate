# High-Level Optimizations
This section describes high-level optimizations implemented in the Surogate framework to improve memory efficiency and training speed.

## Kernel Fusions


## LoRA Fusions


## Fused RoPE

Rotary Position Embedding (RoPE) applies position-dependent rotations to query and key vectors. The standard implementation precomputes frequency tensors (`freq_cis`) and stores them in GPU memory. The fused implementation computes these frequencies on-the-fly within the kernel.

### Enable via config

```yaml
use_fused_rope: true
```

### How it works

**Standard RoPE:**
1. Precompute a `freq_cis` tensor of shape `(T, 2*head_dim)` containing cos/sin values for all positions
2. Store in GPU global memory
3. During forward/backward, load these values from global memory for each head

**Fused RoPE:**
1. No precomputed tensor allocation
2. Each CUDA block computes cos/sin values on-the-fly using `sincosf()`
3. Results cached in shared memory and reused across all heads in the block

### When to use

Fused RoPE is beneficial when:
- Training with long sequence lengths where `freq_cis` allocation is significant
- Memory-constrained scenarios where every MB matters
- The model uses many attention layers (savings compound)


## Multi-threaded multi-GPU


## Chunked Attention


## Chunked LM