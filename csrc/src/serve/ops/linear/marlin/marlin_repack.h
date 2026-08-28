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

// The 4-bit Marlin tile repack, used by the MoE path (#89). gptq_tmp holds (k/8)*n uint32
// in GPTQ order; b_out receives marlin_b_out_words(n, k) uint32.
void marlin_repack_tiles_q4(const void* gptq_tmp, void* b_out, int n, int k,
                            cudaStream_t stream);

// Repack the serve FP8 residency (codes [N,K] e4m3 row-major, scales [N]
// BF16 per output channel) into Marlin B tiles plus channelwise scales.
// gptq_tmp holds (k/4)*n uint32; b_out marlin_b_out_words(n,k) uint32;
// scales_out n bf16.
void marlin_repack_fp8_row(const void* codes, const void* row_scales_bf16, int n, int k,
                           void* gptq_tmp, void* b_out, void* scales_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
