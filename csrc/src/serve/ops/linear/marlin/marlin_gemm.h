#pragma once

// Marlin weight-quantized GEMM (vendored from vLLM, Apache-2.0). Computes
// C[M,N] = A[M,K] x dequant(B) with A row-major BF16 -- which is exactly the
// serve engine's [K,T] column-major activation block viewed as [T,K] -- and
// B in the Marlin tile format produced by the load-time repack.

#include <cstddef>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

int marlin_workspace_locks_count(int sm_count);
std::size_t marlin_c_tmp_floats(int sm_count, int max_tokens);

// locks must be zero-initialized once and persist (kernels self-reset them).
void marlin_gemm_bf16(const void* a, const void* b_packed, const void* b_scales, void* c,
                      void* c_tmp, int* locks, int prob_m, int prob_n, int prob_k,
                      int group_size, bool b_is_fp8, int sm_count, cudaStream_t stream);

} // namespace sinfer::ops::detail
