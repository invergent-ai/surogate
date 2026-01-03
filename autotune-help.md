# Matmul Autotuning (“Matmul Plans”) — How To Use

Surogate’s cuBLASLt GEMM wrapper supports a **plan cache** that can optionally **autotune** cuBLASLt algorithms the first time a new GEMM “shape” is seen, then reuse the fastest choice for the rest of the run.

This feature is implemented in `csrc/src/kernels/matmul.cpp` (`MatmulPlanCache`).

## What Gets Cached (and When)

For each GPU, Surogate caches a “matmul plan” keyed by the properties that affect cuBLASLt algorithm choice:

- `M, N, K` and `ldc` (output leading dimension / stride)
- transpose flags (`EMMTranspose`)
- dtypes for A/B/D and bias dtype (if present)
- whether a bias epilogue is used
- whether FP8 scale pointers are used (FP8 paths)
- **workspace size** (in bytes)

When autotuning is enabled, the first time a key is encountered the code:

1. queries up to `TOPK` heuristic candidates (`cublasLtMatmulAlgoGetHeuristic`)
2. filters out candidates requiring more than the available workspace
3. benchmarks each candidate using CUDA events and picks the **median** time winner
4. stores the winner in the cache and reuses it for subsequent calls

When autotuning is disabled, the cache still avoids per-call descriptor/layout creation and simply uses the **top-1 heuristic** once per key.

## Environment Variables

These variables are read **once**, when the per-device `MatmulPlanCache` is constructed (so set them **before** launching `train` / tests).

### `SUROGATE_MATMUL_AUTOTUNE`

Enable/disable benchmarking-based autotuning.

- `0|false|no|off` → disabled (default)
- `1|true|yes|on` → enabled

Example:

```bash
export SUROGATE_MATMUL_AUTOTUNE=1
```

### `SUROGATE_MATMUL_AUTOTUNE_TOPK`

How many heuristic candidates to request from cuBLASLt per plan.

- default: `16`
- clamped to: `[1, 64]`

Example:

```bash
export SUROGATE_MATMUL_AUTOTUNE_TOPK=16
```

### `SUROGATE_MATMUL_AUTOTUNE_WARMUP`

Warmup launches per candidate before timing.

- default: `3`
- clamped to: `[0, 50]`

Example:

```bash
export SUROGATE_MATMUL_AUTOTUNE_WARMUP=3
```

### `SUROGATE_MATMUL_AUTOTUNE_ITERS`

Timed launches per candidate (median is used).

- default: `15`
- clamped to: `[1, 100]`

Example:

```bash
export SUROGATE_MATMUL_AUTOTUNE_ITERS=15
```

### `SUROGATE_MATMUL_AUTOTUNE_VERBOSE`

Prints which algorithm is initially selected and (if autotune is enabled) which one wins benchmarking.

- `0|false|no|off` → quiet (default)
- `1|true|yes|on` → verbose

Example:

```bash
export SUROGATE_MATMUL_AUTOTUNE_VERBOSE=1
```

You’ll see logs like:

```text
[matmul-plan][dev0] init ... algo_id=...
[matmul-plan][dev0] tuned ... algo_id=... med_ms=... ws=...B
```

## Important Behavior / Caveats

- **CUDA graphs**: autotuning is skipped while a stream is under CUDA graph capture. If you capture graphs, run a warmup step *before* capture to populate/tune plans.
- **Accumulate mode**: autotuning is skipped for `accumulate=true` matmuls (to avoid perturbing accumulation buffers). These calls still use the cached plan + heuristic algo.
- **Determinism**: autotuning does not currently filter for deterministic-only algorithms. If strict bitwise determinism is required, keep autotuning disabled.
- **Workspace matters**: the chosen algorithm depends on the workspace budget. Surogate’s run state typically provides a ~32MB cuBLASLt workspace; changing workspace size effectively creates different plan keys.
- **Safety fallback**: if a cached algo fails at runtime, the code refreshes the top-1 heuristic and retries once.

## Recommended Settings

### 1) “Balanced” (recommended starting point)

Good tradeoff between startup cost and algo quality:

```bash
export SUROGATE_MATMUL_AUTOTUNE=1
export SUROGATE_MATMUL_AUTOTUNE_TOPK=16
export SUROGATE_MATMUL_AUTOTUNE_WARMUP=2
export SUROGATE_MATMUL_AUTOTUNE_ITERS=10
export SUROGATE_MATMUL_AUTOTUNE_VERBOSE=0
```

If you want maximum stability (at higher startup cost), keep the defaults (`WARMUP=3`, `ITERS=15`).

### 2) “Fast startup” (dev / short runs)

Minimize time spent tuning:

```bash
export SUROGATE_MATMUL_AUTOTUNE=1
export SUROGATE_MATMUL_AUTOTUNE_TOPK=8
export SUROGATE_MATMUL_AUTOTUNE_WARMUP=1
export SUROGATE_MATMUL_AUTOTUNE_ITERS=5
```

### 3) “Thorough” (benchmarking / squeezing performance)

Higher confidence you found the best algo for each shape:

```bash
export SUROGATE_MATMUL_AUTOTUNE=1
export SUROGATE_MATMUL_AUTOTUNE_TOPK=32
export SUROGATE_MATMUL_AUTOTUNE_WARMUP=3
export SUROGATE_MATMUL_AUTOTUNE_ITERS=20
```

### 4) Disable autotuning (safe default)

Use cached descriptors/layouts and a single top-1 heuristic per shape (no benchmarking):

```bash
export SUROGATE_MATMUL_AUTOTUNE=0
```

## Practical Workflow

1. Start with `SUROGATE_MATMUL_AUTOTUNE=1` and `SUROGATE_MATMUL_AUTOTUNE_VERBOSE=1` for one short run to confirm tuning is happening.
2. Turn verbose off for real training runs.
3. If startup time is too high, reduce `TOPK` and/or `ITERS`.
4. If performance is inconsistent across machines/driver versions, increase `TOPK` and `ITERS` and compare the chosen `algo_id` logs.

