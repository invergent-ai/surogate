# Autotuned GEMMs + Fused MLP + FlashAttention-3 (FA3) Plan

This document proposes **compute-path** optimizations aimed at improving Surogate throughput as model size grows (e.g. Qwen3 8B+), where end-to-end performance converges toward the underlying GEMM and attention kernel ceilings.

It complements (and corrects) the older “BF16 Scaling Analysis” focus on CPU/launch overhead by targeting: **(1) better GEMM algorithm selection**, **(2) fewer / cheaper FFN kernels and memory round-trips**, and **(3) a faster attention backend when available**.

Relevant implementation touchpoints today:
- cuBLASLt GEMM wrapper: `csrc/src/kernels/matmul.cpp`
- SwiGLU kernel: `csrc/src/kernels/swiglu.cu`
- cuDNN SDPA graph path: `csrc/src/kernels/cudnn_att.cpp` (already caches graphs)
- Transformer block orchestration: `csrc/src/modules/modular_model.h`
- CUTLASS already in-tree (FP4 path): `csrc/src/recipes/nvfp4/kernels/matmul_cutlass_fp4.cu`

---

## Goals

1. **Raise 8B+ BF16 throughput** by improving the “dominant kernels” rather than only shaving host overhead.
2. Keep changes **option-gated** (feature flags) with safe fallbacks to existing cuBLASLt / cuDNN paths.
3. Preserve numerical behavior within acceptable tolerances (and determinism where required).

---

## Proposal A — Autotuned cuBLASLt GEMMs (“Matmul Plans”)

### Motivation

Today, `matmul_cublaslt`:
- creates/destructs descriptors every call, and
- requests only **1** heuristic (`cublasLtMatmulAlgoGetHeuristic(..., request=1, ...)`)

Heuristic #1 is often good, but it is not guaranteed optimal across:
- GPU generations (Hopper vs Blackwell),
- workspace sizes,
- alignments / strides / transpositions,
- BF16 vs FP8 paths and epilogues.

The core idea: **build and cache a “matmul plan” per unique (shape + layout + dtype + epilogue)** and optionally **micro-benchmark top-K candidate algos** once, then reuse the winner for the rest of the run.

### Design: `MatmulPlanCache`

**Key** (must include all algo-relevant attributes):
- `M, N, K`
- `ldc` / leading dimensions for A/B/C/D (or enough to reconstruct layouts)
- transpose flags (EMMTranspose)
- `dtype_a, dtype_b, dtype_c, dtype_bias`
- epilogue kind (default, bias, maybe GELU/SILU if ever added)
- workspace budget (bytes)
- alignment class (e.g. 16B/128B) if needed for correctness/perf
- (optional) GPU arch / device id to avoid cross-device reuse

**Value**:
- `cublasLtMatmulDesc_t` + `cublasLtMatrixLayout_t` (persisted)
- selected `cublasLtMatmulAlgo_t` (or `cublasLtMatmulHeuristicResult_t`)
- metadata: measured time, algo id, required workspace, etc.

**Where to store**:
- Prefer per-device storage owned by the run state (e.g. `IRunState` / `ModularRunState`) so it:
  - shares across forward/backward calls,
  - is safe with CUDA graph capture (no “first call” work during replay),
  - can be freed cleanly at teardown.

### Autotune Procedure (Top-K + Microbench)

When a key is first seen:
1. Build descriptors/layouts once.
2. Call `cublasLtMatmulAlgoGetHeuristic` with `requestAlgoCount = K` (e.g. 8–32).
3. Filter candidates:
   - workspace requirement ≤ available workspace
   - supported for the given dtype/epilogue
4. Time each candidate with CUDA events:
   - warmup launches (e.g. 3)
   - timed launches (e.g. 10–20), take median
5. Cache the winner and use it thereafter.

**Important**: time using *realistic pointers* (actual tensor addresses for common paths), not synthetic pointers that may differ in alignment/stride.

### Controls / Rollout

Add runtime toggles (env vars or config flags), for example:
- `SUROGATE_MATMUL_AUTOTUNE=0|1` (default off initially)
- `SUROGATE_MATMUL_AUTOTUNE_TOPK=16`
- `SUROGATE_MATMUL_AUTOTUNE_ITERS=15`
- `SUROGATE_MATMUL_AUTOTUNE_WARMUP=3`
- `SUROGATE_MATMUL_AUTOTUNE_VERBOSE=0|1` (prints chosen algo ids / timings)

Fallback behavior:
- If autotune is off, or no candidates succeed: use the existing heuristic single-result behavior.

### Expected Impact

Highly shape/hardware dependent; typical outcomes seen in similar systems:
- **0–5%** when heuristic already picks best algo
- **5–15%** when shapes/workspace/arch cause heuristic suboptimality

This is a “must-have” enabler for the next proposals (fused MLP, consistent performance across model sizes/hardware) because it reduces variance and removes accidental slow paths.

### Risks / Caveats

- **Startup cost**: the first step may take extra seconds to benchmark many shapes; mitigate by:
  - lazy-tuning only shapes encountered,
  - or pre-warming only the “top shapes” (QKV, out proj, MLP up/down).
- **Determinism**: some GEMM algos may be non-deterministic; if strict determinism is required, autotune must filter to deterministic candidates (this needs explicit policy + testing).
- **CUDA graphs**: autotune must happen before graph capture, or be guarded so capture doesn’t include one-time tuning logic.

---

## Proposal B — Fused MLP (FFN) Kernels

### Motivation

The FFN path is typically the largest share of compute at 8B+:
1. `mlp_up`: GEMM to **2D** (gate+up)
2. `swiglu`: elementwise activation (reads 2D, writes D)
3. `mlp_down`: GEMM from **D → C**

Current implementation uses separate kernels and global-memory intermediates (`csrc/src/kernels/swiglu.cu` plus GEMMs via `matmul.cpp`).

The optimization target is not only “fewer launches”, but primarily:
- **remove large global-memory round trips** for `mlp_up` and/or `swiglu`,
- keep data in registers/shared memory longer,
- and/or fuse elementwise into GEMM prologues/epilogues.

### Strategy (Incremental → Full)

#### B1. “Easy win”: Fuse SwiGLU into the MLP-down GEMM *prologue*

Keep `mlp_up` GEMM as-is (still produces the 2D buffer), but eliminate the standalone SwiGLU kernel and the D-sized intermediate:
- `mlp_down` GEMM reads `mlp_up` (2D) and computes `swiglu` on-the-fly as it loads tiles for A.

Benefits:
- removes one kernel launch,
- removes write+read of the D-sized `swiglu` tensor.

Cost:
- `mlp_down` becomes a custom GEMM (CUTLASS), not cuBLASLt.

This is a good “first fusion step” because it avoids trying to collapse a 2D GEMM output into D in the epilogue.

#### B2. “Bigger win”: GEMM-up with embedded gate+up split + SwiGLU output

Goal: write only the **D**-sized SwiGLU output (or even keep it transient), eliminating:
- the explicit SwiGLU kernel, and
- the 2D `mlp_up` global-memory output.

This typically requires a specialized kernel (CUTLASS or bespoke) that computes:
- gate projection and up projection, then combines them per element.

Implementation options:
- **Grouped GEMM** (two GEMMs: gate and up) + fused epilogue that writes SwiGLU output.
- A bespoke fused “dual-projection” kernel when weights are stored as a fused (2D, C) matrix.

#### B3. “Full fused MLP”: Two GEMMs fused into one kernel

Compute:
`out = (SiLU(xW_gate) * xW_up) W_down`

This is the highest payoff and highest complexity:
- minimizes intermediate storage,
- can reduce memory traffic substantially,
- but is non-trivial to implement for **training** (backward is complex).

Recommended scope control:
- Start with **forward-only** fused MLP (useful for inference / eval and potentially forward-heavy benchmarks).
- Then extend to **LoRA training** constraints (often skips base weight grads), which can simplify backward.
- Only then consider full weight-gradient fused backward if needed.

### Where It Fits in Surogate

FFN entry points in the forward path are clearly delineated in `csrc/src/modules/modular_model.h`:
- MLP up projection: `MatmulOp::MLPUp`
- SwiGLU: `swiglu_forward(...)` or recipe-driven equivalent
- MLP down projection: `MatmulOp::MLPDown`

This makes it practical to:
- introduce a recipe capability like `recipe->handles_fused_mlp()` or a dedicated module call,
- feature-gate per recipe / per GPU arch.

### Expected Impact

Order-of-magnitude expectations (depends heavily on batch/seq and GPU):
- B1 (fuse SwiGLU into GEMM2 prologue): **~2–6%**
- B2 (avoid writing mlp_up, avoid standalone swiglu): **~5–12%**
- B3 (full fused MLP): **~8–20%**

These gains scale *with model size*, because FFN dominates more at 8B+.

### Risks / Caveats

- **Backward complexity**: the largest wins require corresponding backward fusion to realize full training speedups.
- **Build surface area**: CUTLASS-based kernels add compilation time and architecture specialization needs (similar to FP4 notes in `benchmarks/speed.md`).
- **Numerics**: ensure BF16 accumulation policy matches baseline expectations (likely FP32 accumulate).

---

## Proposal C — FlashAttention-3 (FA3) Backend

### Motivation

Surogate currently uses cuDNN frontend SDPA graphs (`csrc/src/kernels/cudnn_att.cpp`), and graph construction is already cached. Remaining overhead is per-call variant-pack setup and whatever performance ceiling cuDNN’s chosen kernel hits for the given head size / GQA / causal mode.

FA3 can outperform generic SDPA on supported GPUs by:
- warp-specialized kernels,
- better memory movement scheduling,
- aggressive use of SM90/SM100 features.

### Integration Approach

Add a second attention backend:
- `AttentionBackend::CUDNN` (existing)
- `AttentionBackend::FA3` (new, optional)

Selection heuristics:
- GPU arch: enable FA3 on Hopper/Blackwell where FA3 is known to shine
- head dim constraints (e.g. multiples of 8/16)
- causal only (if that’s the dominant path)
- GQA support required

Implementation options:
1. **Vendor FA3 kernels** into `csrc/src/kernels/flash_attn3/` and compile with CMake.
2. Add FA3 as an optional third-party dependency with a CMake toggle.

Fallback:
- If FA3 isn’t available for the current arch/shape, fall back to cuDNN SDPA.

### Expected Impact

Attention is often a smaller fraction than FFN at 8B+, but still material:
- **~3–10%** end-to-end on attention-heavy configs
- potentially more for smaller models / shorter FFN / larger head counts

### Risks / Caveats

- **Build complexity**: FA3 tends to be template-heavy and architecture-tuned; keep it optional.
- **Backward correctness**: must validate gradients carefully; cuDNN currently forces deterministic algorithms in some versions.
- **API stability**: keep FA3 integration behind a stable internal wrapper so upstream changes don’t ripple through the codebase.

---

## Measurement & Validation

### Benchmarks

Use existing benchmark harness:
- `./benchmarks/benchmark.sh "Qwen/Qwen3-8B" bf16`

For A/B toggles, add (or log) feature flags so runs are reproducible:
- `SUROGATE_MATMUL_AUTOTUNE=1`
- `SUROGATE_FUSED_MLP=1` (once implemented)
- `SUROGATE_ATTENTION_BACKEND=fa3|cudnn`

### Profiling Checklist

1. Verify GEMM selection and runtime stability:
   - kernel names (Nsight Systems / Compute)
   - achieved FLOPs vs theoretical
2. Verify FFN memory traffic reductions:
   - fewer launches (SwiGLU fused)
   - smaller or eliminated intermediate tensors
3. Validate attention backend choice:
   - correctness (loss curves)
   - performance (tokens/sec)

### Correctness

Minimum acceptable checks:
- forward output close (BF16 tolerance)
- backward gradients close (relative/absolute thresholds)
- no divergence in loss curves for short runs

---

## Suggested Implementation Order

1. **Matmul plan caching + top-K heuristic + optional microbench** (enables stable GEMM perf and reduces variance).
2. **Fused SwiGLU into MLP-down (B1)** via CUTLASS (moderate complexity, quick FFN win).
3. Expand to **B2/B3 fused MLP** if FFN remains dominant after (1–2).
4. Add **FA3 backend** behind an internal wrapper + feature flag; gate by arch/shape; keep cuDNN fallback.

