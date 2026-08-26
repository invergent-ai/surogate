#pragma once

// Repack the serve W8G32 residency (codes [N,K] int8 row-major, scales
// [N,K/32] FP16) into the Marlin B tile format plus permuted BF16 scales.

#include <cstddef>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// gptq_tmp must hold (k/4)*n uint32; b_out marlin_b_out_words(n,k) uint32;
// scales_out (k/32)*n bf16.
void marlin_repack_w8g32(const void* codes, const void* scales_f16, int n, int k,
                         void* gptq_tmp, void* b_out, void* scales_out, cudaStream_t stream);

std::size_t marlin_b_out_words(int n, int k);

} // namespace ninfer::ops::detail
