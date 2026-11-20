#pragma once

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void w8_linear_swiglu_generic_launch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
// The same kernels with a BF16 pre-activation delta [T][2 * I] ([gate; up]) added to the FP32 gate / up sums before
// the SiLU (adapter rounds' shared expert, sparse_moe_nvfp4.cu). A zero delta reproduces the plain launches' bits.
void w8_linear_swiglu_generic_delta_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           const __nv_bfloat16* delta, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c128_delta_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                const __nv_bfloat16* delta, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c128_delta_launch(const Tensor& x, const Weight& w, Tensor& out,
                                                const __nv_bfloat16* delta, cudaStream_t stream);

void w8_linear_swiglu_decode_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_decode_pair_r4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream);
void w8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);
void w8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);

} // namespace sinfer::ops::detail
