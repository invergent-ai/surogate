# Vendored FLA KDA kernels

Source: [flash-linear-attention v0.5.2](https://github.com/fla-org/flash-linear-attention/tree/v0.5.2),
commit `9c8e42e762fce087c27b673af4922795d9edb85e`.
The selected device kernels retain their upstream MIT license in [LICENSE](LICENSE).
Copyright (c) 2023–2026 Songlin Yang, Yu Zhang, Zhiyuan Li;
[upstream contributors](https://github.com/fla-org/flash-linear-attention/graphs/contributors).

| Local file | Upstream file | Selected kernels |
|---|---|---|
| `norm.py` | `fla/modules/l2norm.py` | `l2norm_fwd_kernel`, `l2norm_bwd_kernel` |
| `cumsum.py` | `fla/ops/utils/cumsum.py` | `chunk_local_cumsum_vector_kernel` |
| `intra.py` | `fla/ops/kda/chunk_intra.py` | `chunk_kda_fwd_kernel_intra_sub_chunk`, `chunk_kda_fwd_kernel_inter_solve_fused`, `chunk_kda_bwd_kernel_intra` |
| `wy.py` | `fla/ops/kda/wy_fast.py` | `recompute_w_u_fwd_kda_kernel` |
| `state.py` | `fla/ops/common/chunk_delta_h.py` | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` |
| `output.py` | `fla/ops/gla/chunk.py` | `chunk_gla_fwd_kernel_o` |
| `backward.py` | `fla/ops/kda/chunk_bwd.py` | `chunk_kda_bwd_kernel_dAv`, `chunk_kda_bwd_kernel_wy_dqkg_fused` |
| `ops.py` | `fla/ops/utils/op.py` | Default `exp2` implementation and `tl.gather` alias |

Local changes:

- Removed Python/autograd wrappers, external FLA imports, autotune decorators and
  argument heuristics. Surogate's compiler supplies explicit constexpr values,
  tiles, signatures and manifests; no installed FLA package is required.
- Kernels receiving `chunk_indices` return early for sequence index `-1`.
  Surogate fills spare entries with that sentinel to keep CUDA graph launch
  dimensions stable when document lengths change. The numerical bodies are
  otherwise unchanged.
- Use `tf32x3` for default FP32 dot products to reduce rounding in gate
  gradients; the triangular solver retains upstream's explicit TF32 setting.
  Use two warps for the forward state kernel, retaining upstream's Blackwell
  correctness restriction. Tile sizes are fixed and recorded in manifests;
  there is no runtime autotuning.
- `../kimi_delta_rule.py` adds Surogate metadata and beta-gradient reduction
  kernels plus AOT compilation. `csrc/src/runtime/jit/kimi_delta_rule_kernels.*`
  owns the native forward/backward launch order and scratch layout.

The integrated path uses BF16 activations, FP32 decay and FP32 output gradients,
Q/K L2 normalization, bounded-gate intra-chunk computation, 64-token chunks,
equal Q/K/V head counts and dimensions, and zero initial state per document.
It recomputes forward intermediates in backward. Context parallelism, persistent
decode state and FLA's Python model/autograd APIs are outside this vendored subset.

When updating, diff these functions against the pinned source, retain licensing,
and run `tests/train/test_kda_triton.py`, `tests/train/test_glm5_training.py`,
the GLM native-colocate GPU test and `benchmarks/bench_glm_kda.py`.
The cache hashes all vendored Python sources, compiler source, geometry, Triton
version and target SM. Any source or compiler-version change recompiles the cubins.
