# BF16 Scaling Analysis & Improvement Plan

## Executive Summary

Surogate's BF16 performance advantage over Unsloth shrinks from **3× at 0.6B to 1.25× at 8B** on datacenter GPUs. This document analyzes the root causes and proposes improvements.

| Model | H100 Speedup | H200 Speedup | RTX 5090 Speedup |
|-------|--------------|--------------|------------------|
| 0.6B  | 2.53×        | 3.00×        | 1.40×            |
| 1.7B  | 1.63×        | 1.82×        | 1.26×            |
| 4B    | 1.34×        | 1.43×        | 1.24×            |
| 8B    | 1.26×        | 1.29×        | 1.24×            |

**Goal**: Maintain ≥1.5× speedup at 8B+ model sizes.

---

## Root Cause Analysis

### 1. Compute vs Overhead Ratio

As model size increases, matmul compute dominates and both frameworks converge to the same cuBLASLt performance ceiling.

| Model | Hidden | FFN Size | Est. Matmul % | Layers |
|-------|--------|----------|---------------|--------|
| 0.6B  | 1024   | 4096     | ~60%          | 28     |
| 1.7B  | 2048   | 8192     | ~75%          | 28     |
| 4B    | 2560   | 10240    | ~85%          | 36     |
| 8B    | 4096   | 14336    | ~92%          | 32     |

**Implication**: Surogate's advantages (no Python overhead, fused kernels) become proportionally smaller.

### 2. Current BF16 Path Bottlenecks

#### 2.1 cuBLASLt Descriptor Overhead (matmul.cpp:98-191)

Every matmul call creates and destroys 6 descriptors:
```cpp
cublasLtMatmulDescCreate(&operationDesc, ...)
cublasLtMatrixLayoutCreate(&ALayout, ...)
cublasLtMatrixLayoutCreate(&BLayout, ...)
cublasLtMatrixLayoutCreate(&CLayout, ...)
cublasLtMatrixLayoutCreate(&DLayout, ...)
cublasLtMatmulPreferenceCreate(&preference)
// ... execute ...
cublasLtMatmulPreferenceDestroy(preference)
cublasLtMatmulDescDestroy(operationDesc)
// ... destroy layouts ...
```

For 8B model with 32 layers × 7 matmuls/layer × 2 (fwd+bwd) = **448 descriptor create/destroy cycles per iteration**.

#### 2.2 cuBLASLt Heuristic Search (matmul.cpp:166-170)

```cpp
cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, ALayout, BLayout,
                               CLayout, DLayout, preference, 1, &heuristic, &returnedResults);
```

Called every matmul. While NVIDIA caches internally, there's still lookup overhead.

#### 2.3 cuDNN Graph Execution Overhead (cudnn_att.cpp)

cuDNN Frontend graphs have execution overhead:
- Variant pack construction per call
- Graph execution dispatch
- No persistent kernel state between calls

#### 2.4 No Operation Fusion Across Layers

Current architecture executes each operation separately:
```
RMSNorm → Matmul → RoPE → Attention → Matmul → Residual → RMSNorm → Matmul → SwiGLU → Matmul → Residual
```

Each arrow is a kernel launch + global memory round-trip.

#### 2.5 Sequential Backward Pass

From modular_model.h, backward executes strictly sequentially:
```cpp
for (layer_idx = num_layers-1; layer_idx >= 0; --layer_idx) {
    backward_mlp_down(...)
    backward_swiglu(...)
    backward_mlp_up(...)
    backward_rmsnorm(...)
    backward_attention(...)
    backward_qkv(...)
    backward_rmsnorm(...)
}
```

No overlap between layers or between compute and memory operations.

---

## Improvement Plan

### Phase 1: Low-Hanging Fruit (Est. 5-15% improvement)

#### 1.1 Cache cuBLASLt Descriptors

**Problem**: Creating/destroying descriptors for every matmul adds CPU overhead.

**Solution**: Cache descriptors keyed by (M, N, K, dtype, transpose_mode).

```cpp
// Proposed: matmul_cache.h
struct MatmulDescKey {
    int M, N, K;
    cudaDataType_t dtype_a, dtype_b, dtype_c;
    EMMTranspose mode;
    bool has_bias;

    bool operator==(const MatmulDescKey&) const = default;
    size_t hash() const;
};

class MatmulDescCache {
    std::unordered_map<MatmulDescKey, CachedMatmulDesc> cache_;
public:
    CachedMatmulDesc& get_or_create(const MatmulDescKey& key, cublasLtHandle_t handle);
};

struct CachedMatmulDesc {
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout, DLayout;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
};
```

**Files to modify**:
- `csrc/src/kernels/matmul.cpp` - Add caching layer
- `csrc/src/training/model.cpp` - Initialize cache with run_state

**Expected impact**: 3-5% on large models (reduces CPU overhead per matmul)

#### 1.2 Pre-select cuBLASLt Algorithms at Init

**Problem**: Algorithm heuristic search happens per-matmul.

**Solution**: During model init, run heuristic search for all unique (M,N,K) shapes and cache results.

```cpp
void Model::initialize_matmul_algorithms() {
    // Collect unique shapes from model config
    std::set<MatmulShape> shapes = collect_matmul_shapes();

    for (const auto& shape : shapes) {
        auto algo = find_best_algorithm(shape);
        algo_cache_[shape] = algo;
    }
}
```

**Expected impact**: 2-3% (eliminates heuristic search from hot path)

#### 1.3 Persistent cuDNN Attention Graphs

**Current**: Graph lookup + variant pack construction per call.

**Improvement**: Pre-build graphs for expected (B, T) combinations during init.

```cpp
void AttentionModule::warmup(int B, int T, cudnnHandle_t handle) {
    // Pre-build forward and backward graphs
    fwd_graph_ = lookup_cache_or_build_graph_fwd(B, Hq, Hkv, T, HS, false, handle);
    bwd_graph_ = lookup_cache_or_build_graph_bwd(B, Hq, Hkv, T, HS, handle);

    // Pre-allocate variant pack storage
    fwd_variant_pack_.reserve(5);
    bwd_variant_pack_.reserve(10);
}
```

**Expected impact**: 2-4% (reduces per-call overhead)

---

### Phase 2: Kernel Fusion (Est. 10-25% improvement)

#### 2.1 Fused RMSNorm + Matmul Input Preparation

**Current**:
```
RMSNorm output → Global memory → Matmul reads input
```

**Proposed**: Fuse RMSNorm output directly into matmul's register file (requires custom kernel or CUTLASS epilogue).

For cuBLASLt, this can be achieved via epilogue fusion:
```cpp
cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_AUX;  // Example
// Or use CUTLASS for custom epilogues
```

**Files to modify**:
- `csrc/src/kernels/rmsnorm.cu` - Add fused variant
- `csrc/src/recipes/bf16/bf16_recipe.h` - Enable fused path

**Expected impact**: 5-8% (eliminates one global memory round-trip per layer)

#### 2.2 Fused Matmul + Residual Addition

**Current**:
```
Matmul output → Global memory → Residual add kernel → Global memory
```

**Proposed**: Use cuBLASLt's `CUBLASLT_EPILOGUE_BIAS` with beta=1.0 to accumulate directly:
```cpp
// Instead of: C = A @ B, then C += residual
// Do: C = A @ B + residual (single kernel)
matmul(..., /*accumulate=*/true, ...);  // Already supported!
```

**Current code already supports this** via the `accumulate` parameter. Verify it's being used optimally in the forward pass.

**Expected impact**: 3-5% if not already enabled

#### 2.3 Fused Attention Output + Residual + RMSNorm

**Current**:
```
Attention → Matmul(out_proj) → Add(residual) → RMSNorm → Matmul(mlp_up)
```

**Proposed**: CUTLASS grouped GEMM with custom epilogue:
```
Attention → Matmul(out_proj) + residual + RMSNorm (fused) → Matmul(mlp_up)
```

This requires writing a custom CUTLASS kernel with a fused epilogue that:
1. Adds residual
2. Computes RMSNorm statistics
3. Applies normalization

**Files to create**:
- `csrc/src/kernels/fused_proj_residual_norm.cu`

**Expected impact**: 8-12% (eliminates 2 global memory round-trips per layer)

#### 2.4 Fused SwiGLU + Matmul (MLP Down)

**Current**:
```
MLP Up output → SwiGLU kernel → Global memory → MLP Down matmul
```

**Proposed**: Fuse SwiGLU as a custom epilogue on MLP Up matmul OR as prologue on MLP Down.

Using CUTLASS EVT (Epilogue Visitor Tree):
```cpp
// CUTLASS 3.x style
using EpilogueOp = cutlass::epilogue::fusion::LinCombPerRowBiasEltActEltAdd<...>;
// SwiGLU: output = silu(gate) * up
```

**Expected impact**: 5-8%

---

### Phase 3: Pipelining & Overlap (Est. 10-20% improvement)

#### 3.1 Multi-Stream Layer Pipelining

**Current**: Single stream, sequential execution.

**Proposed**: Overlap layer N's backward with layer N+1's gradient all-reduce (for multi-GPU).

```cpp
void backward_pipelined(int num_layers) {
    cudaStream_t compute_stream, comm_stream;

    for (int i = num_layers - 1; i >= 0; --i) {
        // Compute gradients on compute_stream
        backward_layer(i, compute_stream);

        // Overlap: Start all-reduce for layer i+1 while computing layer i
        if (i < num_layers - 1) {
            cudaStreamWaitEvent(comm_stream, layer_done_event[i+1]);
            all_reduce_gradients(i+1, comm_stream);
        }

        cudaEventRecord(layer_done_event[i], compute_stream);
    }
}
```

**Expected impact**: 10-15% on multi-GPU setups

#### 3.2 Prefetch Next Layer Weights

**Current**: Weights read from HBM on-demand.

**Proposed**: While computing layer N, prefetch layer N+1 weights to L2.

```cpp
// Use CUDA's async memcpy with hints
cudaMemPrefetchAsync(next_layer_weights, size, device, prefetch_stream);
```

**Expected impact**: 3-5% (better L2 cache utilization)

#### 3.3 Overlap Optimizer Step with Next Forward

**Current**: Forward → Backward → Optimizer → Next Forward

**Proposed**: Start next forward while optimizer updates weights (for parameters not yet needed).

**Expected impact**: 5-10% (depends on model architecture)

---

### Phase 4: Alternative Attention Implementation (Est. 5-15% improvement)

#### 4.1 FlashAttention-3 Integration

**Current**: cuDNN SDPA (good, but generic).

**Proposed**: Integrate FlashAttention-3 which has:
- Better warp specialization
- Asynchronous softmax
- FP8 support with better accuracy

```cpp
// Option A: Use flash-attn library directly
#include <flash_attn/flash_api.h>

// Option B: Port FA3 kernels to Surogate
```

**Expected impact**: 5-10% on attention-heavy workloads

#### 4.2 Specialized Attention for GQA

**Current**: cuDNN handles GQA generically.

**Proposed**: For common GQA ratios (e.g., Qwen3's 8:1), use specialized kernels.

**Expected impact**: 3-5% for GQA models

---

### Phase 5: Memory Optimization (Enables larger batch sizes)

#### 5.1 Activation Checkpointing Optimization

**Current**: Recompute entire blocks.

**Proposed**: Selective recomputation - only recompute cheap ops (RMSNorm, SwiGLU), keep expensive attention.

**Expected impact**: 20-30% memory reduction → larger batches → better GPU utilization

#### 5.2 Gradient Accumulation Fusion

**Current**: Accumulate gradients in BF16.

**Proposed**: Accumulate in FP32 buffer, cast to BF16 only for communication.

**Expected impact**: Better numerical stability, enables larger effective batch sizes

---

## Implementation Priority

| Phase | Improvement | Effort | Impact | Priority |
|-------|-------------|--------|--------|----------|
| 1.1   | Cache cuBLASLt descriptors | Low | 3-5% | **P0** |
| 1.2   | Pre-select algorithms | Low | 2-3% | **P0** |
| 1.3   | Persistent cuDNN graphs | Low | 2-4% | **P0** |
| 2.2   | Verify residual fusion | Low | 3-5% | **P0** |
| 2.1   | Fused RMSNorm+Matmul | Medium | 5-8% | P1 |
| 2.3   | Fused proj+res+norm | High | 8-12% | P1 |
| 2.4   | Fused SwiGLU+Matmul | Medium | 5-8% | P1 |
| 3.1   | Multi-stream pipeline | Medium | 10-15% | P1 |
| 4.1   | FlashAttention-3 | High | 5-10% | P2 |
| 3.2   | Weight prefetch | Low | 3-5% | P2 |
| 3.3   | Overlap optimizer | Medium | 5-10% | P2 |

**Estimated total improvement**: 30-50% for large models (additive effects vary)

---

## Validation Plan

### Benchmarks to Run

1. **Per-operation profiling** with Nsight Systems:
   ```bash
   nsys profile -o bf16_8b surogate sft --config qwen3-8b-bf16.yaml
   ```

2. **Roofline analysis** to confirm compute vs memory bound:
   ```bash
   ncu --set full -o roofline surogate sft --config qwen3-8b-bf16.yaml
   ```

3. **A/B comparisons** for each optimization:
   - Baseline: Current BF16 recipe
   - Variant: With optimization enabled
   - Metric: tok/sec, GPU utilization, memory bandwidth

### Success Criteria

- 8B model speedup over Unsloth: **≥1.5×** (currently 1.26×)
- No accuracy regression (loss curves must match baseline)
- Memory usage increase: **≤5%**

---

## Appendix: Profiling Commands

```bash
# Full Nsight Systems trace
nsys profile --trace=cuda,nvtx,osrt -o surogate_bf16 \
    surogate sft --config benchmarks/qwen3-8b-bf16.yaml

# Nsight Compute for specific kernels
ncu --kernel-name "volta_.*gemm.*" --launch-count 10 \
    surogate sft --config benchmarks/qwen3-8b-bf16.yaml

# Memory bandwidth analysis
ncu --metrics l2_read_throughput,l2_write_throughput,dram_read_throughput,dram_write_throughput \
    surogate sft --config benchmarks/qwen3-8b-bf16.yaml
```
