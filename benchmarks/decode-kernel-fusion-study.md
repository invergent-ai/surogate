# Feasibility Study: DSL Compilation Backend for Optimized Kernel Generation

## Executive Summary

**The primary throughput gap vs vLLM is CUDA graph segmentation, not kernel fusion.**

Surogate's decode forward pass is split into **3 segments per layer** (96 total for 32-layer Llama)
because `flash_attention` breaks CUDA graph capture. vLLM's FULL mode captures the entire decode
step as **1 CUDA graph** by using a graph-safe attention backend, achieving **~3x speedup** over
non-graphed execution.

### Analysis Results (from partitioner)

| Model | Kernel Launches | CUDA Graph Segments | Segments/Layer | Graph-Breaking Op |
|-------|----------------|--------------------:|---------------:|-------------------|
| Llama-32L | 291 | 96 | 3.0 | flash_attention (32x) |
| Qwen3-32L | 291 | 96 | 3.0 | flash_attention (32x) |
| Qwen3-MoE-24L | 339 | 72 | 3.0 | flash_attention (24x) |

### Priority Order

1. **Graph-safe decode attention** → enables full single-graph capture → **~3x decode speedup**
2. **Kernel fusion via codegen** → reduces kernel count within the graph → **10-20% additional**

Surogate already has:
- A full computation graph with resolved shapes and dtypes
- A JIT kernel pipeline (Triton + CuTe DSL → cubin → manifest → runtime load)
- A fusion-pass registry that rewrites the graph before execution
- Split-attention CUDA graph capture (equivalent to vLLM PIECEWISE mode)

The missing pieces are:
1. A graph-safe decode attention kernel (for full CUDA graph capture)
2. A codegen engine for fusing ops within the graph (like TorchInductor)

---

## 1. What torch.compile Actually Does (and What Matters Here)

torch.compile's pipeline has four stages. Only the last two are relevant to Surogate:

| Stage | torch.compile | Surogate equivalent | Gap |
|-------|--------------|---------------------|-----|
| **Graph capture** | TorchDynamo (bytecode interception) | Python DSL → IR (explicit) | **None** — Surogate's DSL is strictly better: no graph breaks, complete topology |
| **Autodiff split** | AOTAutograd | `autodiff.cpp` + `autodiff_rules.cpp` | **None** — already generates backward graph |
| **Fusion + codegen** | TorchInductor (Triton/C++ codegen) | Hand-fused kernels + fusion-pass registry | **This is the gap** |
| **Runtime dispatch** | Compiled code + guard checks | `graph_executor.cpp` op loop | Minor — see §5 |

### What makes Inductor effective for LLMs

The primary wins come from:

1. **Pointwise fusion**: Chains of elementwise ops (add, mul, scale, activation, cast) → single
   Triton kernel. Eliminates intermediate tensor materialization and kernel launch overhead.
2. **Reduction fusion**: Pointwise ops feeding into reductions (e.g., norm computation) merged.
3. **Epilogue fusion**: Post-matmul operations (bias add, activation, quantization scale tracking)
   fused into the GEMM epilogue via CUTLASS or cuBLAS epilogue callbacks.
4. **Memory planning**: Inductor's scheduler assigns buffers to minimize peak memory.

**The real win is CUDA graph capture, not kernel fusion.** vLLM's 3x speedup from torch.compile
comes primarily from enabling full CUDA graph capture for the entire decode step. TorchInductor's
kernel fusion is a secondary benefit that reduces kernel count within the captured graph.

For dense transformers, Inductor's actual kernel-level wins are modest because the critical path
is dominated by large matmuls (which already dispatch to cuBLAS) and FlashAttention (hand-written).
The wins concentrate in the "glue" between these: norm + residual, activation functions,
view/transpose chains, and quantization bookkeeping.

---

## 2. Current State: Where Surogate's Ops Spend Time

A single transformer layer (e.g., Llama block) executes roughly this sequence:

```
fused_residual_rmsnorm  ← FUSED kernel (residual + norm)
view                    ← metadata only (no kernel)
matmul (QKV proj)       ← cuBLAS
view                    ← metadata only
rope                    ← custom kernel
flash_attention         ← FlashAttention kernel
view                    ← metadata only
matmul (O proj)         ← cuBLAS
view                    ← metadata only
fused_residual_rmsnorm  ← FUSED kernel
view                    ← metadata only
matmul (gate+up)        ← cuBLAS
view                    ← metadata only
swiglu                  ← custom kernel
view                    ← metadata only
matmul (down proj)      ← cuBLAS
view                    ← metadata only
```

### What's already fused
- `fused_residual_rmsnorm`: residual add + RMSNorm (one kernel)
- `matmul_swiglu`: matmul + SwiGLU (available, not used in all models)
- `qkv_qk_norm_rope`: QK norm + RoPE (Qwen3)
- `fused_lm_head_loss`: LM head matmul + cross-entropy
- `moe_grouped_gemm_gate_up/down`: all-expert GEMM
- `mamba_combine_scan`: entire Mamba2 block

### What's NOT fused (and could be)
| Pattern | Current ops | Potential fusion |
|---------|-------------|------------------|
| **Post-matmul reshape + activation** | `matmul → view → swiglu` | Epilogue fusion or single Triton kernel |
| **Quantization bookkeeping** | `abs_max tracking, scale computation` between matmuls | Fused into GEMM epilogue |
| **Bias + activation** | `bias_add → silu_mul` | Single pointwise kernel |
| **RoPE + view** | `view → rope → view` | Eliminate views, fuse into rope kernel |
| **MoE routing** | `moe_softmax → moe_topk → moe_permute` | Single fused routing kernel |
| **Residual bookkeeping** | Between-layer `add` ops in some models | Fuse into next layer's norm |
| **Backward glue** | Gradient accumulation, scaling, casting between backward matmuls | Pointwise fusion |

### Quantification of the opportunity

For **training**, the opportunity is **small-to-moderate** (5-15% throughput improvement):
- Forward + backward is dominated by matmuls (cuBLAS) and FlashAttention
- The "glue" ops are fast but incur kernel launch overhead (~5-10μs each)
- With 16+ ops per layer × 32+ layers = 500+ kernel launches per iteration

For **inference decode** (batch=1, seq=1), the opportunity is **large** (20-50% latency reduction):
- Matmuls are memory-bandwidth-bound at small batch sizes
- Kernel launch overhead becomes a larger fraction of total time
- Fusing the glue eliminates many launches and intermediate memory traffic
- This is where torch.compile shows its biggest wins for LLMs

For **inference prefill** (large seq), the opportunity is **moderate** (10-20%):
- Similar profile to training forward pass
- Fewer backward-related ops

---

## 3. Proposed Architecture

### 3.1 Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │              Existing Pipeline                   │
                    │                                                  │
  Python DSL ──→ IR ──→ C++ GraphCompiler ──→ CompiledGraph           │
                    │         │                     │                  │
                    │         │  ┌──────────────────┘                  │
                    │         │  │                                     │
                    │         ▼  ▼                                     │
                    │   ┌─────────────────┐                           │
                    │   │  Fusion Pass    │  (existing registry)       │
                    │   │  Framework      │                           │
                    │   └────────┬────────┘                           │
                    │            │                                     │
                    └────────────┼─────────────────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────────────────┐
                    │         NEW: Codegen Backend                     │
                    │                                                  │
                    │  ┌──────────────┐   ┌────────────────────┐      │
                    │  │  Subgraph    │   │  Kernel Codegen    │      │
                    │  │  Partitioner │──→│  (Triton / CuTe)   │      │
                    │  └──────────────┘   └─────────┬──────────┘      │
                    │                               │                  │
                    │                    ┌──────────▼──────────┐      │
                    │                    │  JIT Compile + Cache │      │
                    │                    │  (existing pipeline) │      │
                    │                    └──────────┬──────────┘      │
                    │                               │                  │
                    │                    ┌──────────▼──────────┐      │
                    │                    │  Graph Rewrite      │      │
                    │                    │  (replace subgraph  │      │
                    │                    │   with fused op)    │      │
                    │                    └─────────────────────┘      │
                    └──────────────────────────────────────────────────┘
```

### 3.2 Components

#### A. Subgraph Partitioner (`surogate/compiler/partitioner.py`)

Identifies fusible subgraphs in the `CompiledGraph`. Rules:

1. **Pointwise cluster**: Chain of elementwise ops (add, mul, scale, silu, sigmoid, cast, view)
   between two "anchor" ops (matmul, attention, MoE, mamba)
2. **Reduction + pointwise**: RMSNorm/LayerNorm preceded or followed by pointwise ops
3. **Epilogue**: Pointwise ops immediately after a matmul (bias, activation, quantize)
4. **Prologue**: Pointwise ops immediately before a matmul (cast, scale, transpose)

Non-fusible "anchor" ops that break partitions:
- `matmul` / `batched_matmul` (dispatch to cuBLAS/CUTLASS — keep as-is)
- `flash_attention` (highly optimized hand-written kernel)
- `moe_grouped_gemm*` (specialized CUTLASS grouped GEMM)
- `mamba_*` (complex stateful ops)
- Any op with side effects (e.g., `embedding` scatter-add backward)

Output: List of `FusibleSubgraph` objects, each containing ordered ops, input/output tensors,
and intermediate tensors that can be eliminated.

#### B. Kernel Codegen (`surogate/compiler/codegen/`)

Two backends (matching existing JIT infrastructure):

**Triton codegen** (`triton_codegen.py`):
- Generate `@triton.jit` function from subgraph
- Map DSL ops to Triton operations:
  - `add/mul/scale` → `tl.load` + arithmetic + `tl.store`
  - `silu/sigmoid/relu` → inline activation
  - `rmsnorm` → reduction + normalize
  - `cast` → `tl.cast` or `.to()`
  - `view` → offset calculation (no codegen needed)
- Handle reductions (RMSNorm variance) with `tl.reduce`
- Apply standard Triton autotuning for block sizes
- Target: SM80+ (all supported GPUs)

**CuTe DSL codegen** (`cute_codegen.py`):
- Generate quack kernel definitions for Blackwell (SM120+)
- Leverage CuTe's native TMA and shared memory management
- Target: Maximum throughput on Blackwell-class GPUs

**Shared infrastructure**:
- Template library of codegen patterns (activation functions, reductions, casts)
- Shape specialization: Generate kernel variants for known shapes
- Compile to cubin via existing `surogate/kernels/compiler.py`
- Write manifest for existing `JitKernel::load_manifest()` loader

#### C. Graph Rewriter (extends `csrc/src/runtime/dsl/fusions/`)

New fusion pass type that:
1. Takes partitioner output
2. Replaces subgraph ops with single `FusedJitKernel` op
3. Registers the JIT kernel manifest
4. Updates tensor lifetime analysis (intermediates removed)

This integrates with the existing `FusionPassRegistry` — each generated fusion is registered
as a pass with a unique ID and the required JIT kernel capability.

### 3.3 Integration Points

| Component | Integration method | Changes needed |
|-----------|-------------------|----------------|
| **Partitioner** | New Python module | None to existing code |
| **Codegen** | New Python module | None to existing code |
| **JIT compile** | Uses existing `surogate/kernels/compiler.py` | Minor: add batch compilation API |
| **JIT load** | Uses existing `JitKernel::load_manifest()` | None |
| **Graph rewrite** | New fusion pass in `fusions/` directory | Add `CompiledOpType::FusedJitKernel` |
| **Dispatch** | New case in `compiled_ops.cpp` switch | Add `dispatch_fused_jit_kernel()` |
| **Activation** | CLI flag or config option | Add `compile_mode` to training/inference config |

---

## 4. Implementation Strategy (Revised)

### Phase 0: Graph-Safe Decode Attention (HIGHEST PRIORITY)

**Goal**: Enable full CUDA graph capture for the entire decode step, matching vLLM FULL mode.

**The problem**: `flash_attention` breaks CUDA graph capture due to dynamic `cu_seqlens`. For decode
(T=1 per request, fixed KV-cache positions), the attention pattern is actually static and can be
made graph-safe.

**vLLM's approach**: They wrap attention in a custom op (`torch.ops.vllm.unified_attention_with_output`)
with the output tensor pre-allocated as an input. This makes the op fully graph-compatible. They
support multiple attention backends with varying levels of CUDA graph compatibility:
- FlashAttention v3: Always graph-safe (ALWAYS)
- FlashInfer: Graph-safe for uniform decode only (UNIFORM_SINGLE_TOKEN_DECODE)
- FlashAttention v2: Graph-safe for uniform batches (UNIFORM_BATCH)

**vLLM's graph modes** (from their docs):
- `PIECEWISE`: Attention runs eagerly, rest captured — equivalent to Surogate's current split-attention
- `FULL`: Entire decode in 1 CUDA graph — requires graph-safe attention backend
- `FULL_AND_PIECEWISE` (default): Full for decode, piecewise for prefill — best performance

**Implementation options for Surogate**:

A. **Use FlashAttention v3 / FlashInfer for decode** — these backends natively support CUDA graph
   capture for decode batches. Integrate via JIT or build-time linking.

B. **Implement a decode-specific attention kernel** (Triton or CuTe DSL) that:
   - Takes pre-allocated output buffer as input (like vLLM)
   - Uses fixed-shape KV-cache access patterns (no dynamic cu_seqlens)
   - Supports paged KV-cache for memory efficiency
   - Is fully CUDA-graph-compatible

C. **Make existing FlashAttention graph-safe for decode** — for pure decode batches (all queries
   have T=1), pre-compute cu_seqlens as a fixed buffer and avoid any host-side dynamic
   computation. Mark the op as graph-safe when `mode == InferenceDecode`.

**Expected impact**: **~3x decode throughput** (matches vLLM FULL mode).

**Current segment analysis** (from partitioner):
```
Llama-32L:     96 segments → 1 segment  (96x reduction)
Qwen3-32L:     96 segments → 1 segment  (96x reduction)
Qwen3-MoE-24L: 72 segments → 1 segment  (72x reduction)
```

### Phase 1: Epilogue Fusion via Existing Primitives (2-3 weeks)

**Goal**: Use the existing `matmul_swiglu` op and add new fused ops for common patterns.

**Scope** (from partitioner analysis):
- **matmul → swiglu**: 32x per Llama/Qwen3 forward — switch to existing `matmul_swiglu` primitive
- **matmul → rope**: 32x per Llama — consider a `matmul_rope` fused op
- **moe_softmax → moe_topk**: 24x per Qwen3-MoE — fuse into single router kernel
- **moe_grouped_gemm_gate_up → swiglu**: 24x per MoE — fuse activation into GEMM epilogue

**Expected impact**: 10-22% kernel count reduction within the CUDA graph.

### Phase 2: Triton Codegen for Remaining Pointwise Chains (4-6 weeks)

**Goal**: Fuse operations adjacent to matmuls.

**Scope**:
- Post-matmul: bias_add, activation, abs_max tracking → GEMM epilogue
- Pre-matmul: cast, scale, transpose → GEMM prologue
- Requires CUTLASS integration for custom epilogues (cuBLAS epilogue API is limited)
- Alternative: fuse via Triton by generating `tl.dot` + epilogue

**Expected impact**: 5-15% additional, especially for FP8/FP4 quantization overhead.

### Phase 4: CuTe DSL Backend + Blackwell Specialization (4-6 weeks)

**Goal**: Generate CuTe DSL kernels for SM120+ with TMA and async operations.

**Scope**:
- CuTe codegen backend using quack library
- Leverage TMA for efficient data movement
- Warp-specialized kernels for compute-bound fusions
- Target: Maximum throughput on B200/GB200/RTX 50xx

### Phase 5: Training-Specific Optimizations (3-4 weeks)

**Goal**: Apply fusion to backward pass and gradient operations.

**Scope**:
- Generate backward kernels for fused forward ops (or rely on autodiff of fused op)
- Fuse gradient accumulation + scaling + casting chains
- Optimize gradient checkpointing boundaries with fused recompute

---

## 5. Key Design Decisions

### 5.1 Codegen in Python (Triton) vs C++ (CUDA)

**Recommendation: Python-first (Triton), CuTe DSL for Blackwell.**

Rationale:
- Triton codegen is dramatically simpler than raw CUDA codegen
- The JIT pipeline already supports both backends
- Triton autotuning handles block-size selection
- CuTe DSL gives Blackwell-specific optimizations (TMA, warp specialization)
- Avoid building a full CUDA code generator (massive engineering effort, low ROI vs Triton)

### 5.2 When to Compile: Startup vs First-Use

**Recommendation: Compile at model initialization (startup).**

Rationale:
- Shapes are known after IR compilation (symbolic dims resolved from config)
- One-time cost, amortized over entire training run
- Consistent with existing JIT kernel compilation (`jit_compile.py`)
- Cache compiled cubins on disk for fast re-initialization

### 5.3 Granularity of Fusion

**Recommendation: Conservative fusion with escape hatches.**

Rules:
1. Never fuse across layer boundaries
2. Never fuse ops with different shard strategies
3. Never fuse ops that have LoRA hook points (need hook injection)
4. Allow user override via `_inference_opts_` to enable/disable specific fusions
5. Start with inference-only; extend to training after validation

### 5.4 Handling Dynamic Shapes (Inference)

For inference, batch size and sequence length vary:
- **Decode**: batch varies, seq=1 → specialize for seq=1, parameterize batch
- **Prefill**: both vary → generate a few shape variants (bucket sizes) or use dynamic shapes
- Consistent with existing `pre_compile_token_buckets` approach

### 5.5 Interaction with Recipe System

Fused kernels must respect the active recipe's precision requirements:
- BF16: All fused ops in bfloat16
- FP8-Hybrid: Fused ops track abs_max for delayed scaling
- NVFP4: Fused ops handle 2D block scaling

The codegen must emit recipe-aware code. This is the most complex integration point.

---

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Numerical divergence** from fused kernels | Training instability | Bit-exact validation against per-op execution; recipe-aware codegen |
| **Codegen complexity** grows with op vocabulary | Maintenance burden | Template-based codegen; limit fused op types to pointwise + simple reductions |
| **Triton compilation overhead** | Slow startup | Aggressive disk caching; pre-compile common configurations |
| **Debugging fused kernels** | Harder to isolate issues | Fallback to per-op mode via config flag; `SUROGATE_FUSION_DEBUG` |
| **Backward correctness** for fused ops | Gradient errors | Generate backward from fused forward via autodiff; cross-validate |
| **Recipe interaction** | Wrong precision in fused kernels | Recipe-parameterized codegen templates; test all recipe × fusion combinations |

---

## 7. Comparison to Alternatives

### Alternative A: Integrate torch.compile directly

Use `torch.compile` on a PyTorch wrapper that calls Surogate ops.

**Pros**: Get Inductor's fusion for free; large community.
**Cons**: Requires PyTorch tensors (Surogate uses its own `Tensor` class); graph breaks at
every custom C++ op; AOTAutograd conflicts with Surogate's autodiff; recipe system not
expressible in PyTorch's dtype model; loses all advantages of the DSL.

**Verdict**: Not viable — the impedance mismatch is too large.

### Alternative B: Use Triton directly for all ops (replace cuBLAS)

Write the entire model in Triton, including matmuls and attention.

**Pros**: Maximum fusion flexibility; single codegen backend.
**Cons**: Triton matmuls underperform cuBLAS for large shapes; FlashAttention in Triton is
possible but slower than the hand-tuned CUDA version; massive rewrite.

**Verdict**: Not recommended — keep cuBLAS and FlashAttention for anchor ops.

### Alternative C: MLIR-based compilation (e.g., via StableHLO/Linalg)

Lower the DSL IR to MLIR and use IREE or XLA's compiler.

**Pros**: Mature fusion and tiling infrastructure; hardware-agnostic.
**Cons**: Enormous integration effort; MLIR ecosystem targets different abstractions;
loses recipe-level precision control; poor FP8/FP4 support in MLIR today.

**Verdict**: Interesting long-term, but too expensive for near-term impact.

### Alternative D: The proposed approach (this document)

Generate Triton/CuTe DSL kernels for fusible subgraphs; keep hand-optimized anchor ops.

**Pros**: Incremental; leverages existing JIT pipeline; recipe-aware; high ROI.
**Cons**: Requires building a codegen engine (moderate effort).

**Verdict**: Recommended.

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Decode latency** (Llama-8B, bs=1) | 20-30% reduction | `surogate serve` benchmark |
| **Prefill throughput** (Llama-8B, seq=4096) | 10-15% improvement | `surogate serve` benchmark |
| **Training throughput** (Llama-8B, SFT) | 5-10% improvement | `surogate sft` benchmark |
| **Kernel count per layer** | 50%+ reduction | `SUROGATE_OP_TRACE` profiling |
| **Compilation overhead** | < 60s cold, < 5s warm | Startup timing |
| **Numerical parity** | < 1e-5 relative error vs per-op | Validation suite |

---

## 9. Recommended Next Steps

1. **Prototype the partitioner** on a Llama block IR — identify all fusible subgraphs and
   estimate the theoretical kernel-count reduction.

2. **Prototype Triton codegen** for the simplest case: a chain of 2-3 pointwise ops
   (e.g., `bias_add → silu → scale`). Validate correctness and measure speedup.

3. **Benchmark the gap** — profile a Llama decode iteration with `SUROGATE_OP_PROFILE` to
   quantify time spent in fusible vs anchor ops. This gives a concrete ceiling for improvement.

4. **Design the recipe-aware codegen API** — define how `BF16Recipe` vs `FP8HybridRecipe`
   parameterize the generated Triton code (dtype selection, scaling factor tracking).

5. **Extend the fusion pass framework** with a `FusedJitKernel` op type and dispatch function
   in `compiled_ops.cpp`.
