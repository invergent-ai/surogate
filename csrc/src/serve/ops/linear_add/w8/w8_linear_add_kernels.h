#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

// The exact-T split-K launcher bakes (rows, k) tables over T = 2..last. These
// read the tables the launcher owns, so the plan's SplitKMmaExactT bands, the
// bench's candidate row and the plan/launcher agreement test all consult the
// same list instead of restating it. last_cols is 0 when (rows, k) has no
// table; covers(rows, k, t) is t in [2, last_cols].
std::int32_t w8_linear_add_exact_t_last_cols(std::int32_t rows, std::int32_t k) noexcept;
bool w8_linear_add_exact_t_covers(std::int32_t rows, std::int32_t k, std::int32_t t) noexcept;

void w8_linear_add_decode_r4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream);
void w8_linear_add_decode_r8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream);
void w8_linear_add_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void w8_linear_add_simt_r8_c4_launch(bool full, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_simt_r8_c8_launch(bool full, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_splitk_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void w8_linear_add_medium_splitk_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void w8_linear_add_mma_r32_c128_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c32_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c48_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c64_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c80_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c96_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c112_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c64_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c80_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c96_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c112_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c128_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c32_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c48_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c64_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c80_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c96_launch(bool full, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c112_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c128_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r128_c64_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r128_c80_launch(bool full, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);

} // namespace sinfer::ops::detail
