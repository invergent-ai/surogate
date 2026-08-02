// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Byte layout of a streamed NVFP4 block weight.

#ifndef SUROGATE_SRC_DSL_NVFP4_STREAM_LAYOUT_H
#define SUROGATE_SRC_DSL_NVFP4_STREAM_LAYOUT_H

#include <cstddef>

#include "kernels/kernels.h"

namespace dsl {

/**
 * @brief Offsets of the six sections of a streamed NVFP4 weight blob.
 *
 * Under cpu_training / offload_master the frozen matmul weights are quantized
 * to NVFP4 ONCE (at import) and streamed as FP4 bytes, instead of streaming
 * BF16 and re-quantizing at every matmul. Forward and dgrad need DIFFERENT
 * layouts of the same weight -- forward consumes W as (N, K), dgrad consumes
 * W^T as (K, N), and packed FP4 with 16-element block scales cannot be
 * transposed cheaply -- so the blob carries both, each quantized from the
 * original BF16 master by the same kernel the on-the-fly path would have used.
 *
 * Layout (each section 16-byte aligned):
 *
 *     [ fwd data  N*(K/2) ][ fwd scales ][ fwd amax 4B ]
 *     [ bwd data  K*(N/2) ][ bwd scales ][ bwd amax 4B ]
 *
 * Total is ~1.125 bytes/param against BF16's 2.0, and it fits inside the
 * BF16 master buffer, so the quantization happens in place with no realloc.
 */
struct Nvfp4StreamLayout {
    std::size_t fwd_data = 0;
    std::size_t fwd_scales = 0;
    std::size_t fwd_scales_bytes = 0;
    std::size_t fwd_amax = 0;
    std::size_t bwd_data = 0;
    std::size_t bwd_scales = 0;
    std::size_t bwd_scales_bytes = 0;
    std::size_t bwd_amax = 0;
    std::size_t total = 0;

    /// @param N Weight rows (C_out), @param K weight cols (C_in), both from the
    /// ORIGINAL row-major BF16 shape.
    /// Section alignment. CUTLASS block-scaled GEMMs read the scale tensor through a
    /// swizzled (F8_128x4) atom layout and the FP4 operand through wide vector loads;
    /// both assume the strong alignment a device allocator hands out, which is what the
    /// per-weight cache buffers used to provide. 256 B matches cudaMalloc's guarantee and
    /// costs a few hundred padding bytes per weight. (Under-aligning here is the FP4
    /// analogue of the FP8 scale-pointer bug: the GEMM does not fail, it returns NaN.)
    static constexpr std::size_t kAlign = 256;

    static Nvfp4StreamLayout make(long N, long K) {
        auto align_up = [](std::size_t v) { return (v + (kAlign - 1)) & ~(kAlign - 1); };
        const auto n = static_cast<std::size_t>(N);
        const auto k = static_cast<std::size_t>(K);

        Nvfp4StreamLayout l;
        std::size_t off = 0;

        l.fwd_data = off;
        off = align_up(off + n * (k / 2));
        l.fwd_scales = off;
        l.fwd_scales_bytes = compute_nvfp4_cutlass_scale_size(static_cast<int>(N), static_cast<int>(K));
        off = align_up(off + l.fwd_scales_bytes);
        l.fwd_amax = off;
        off = align_up(off + sizeof(float));

        // dgrad consumes W^T: (K, N) row-major, so rows=K, cols=N.
        l.bwd_data = off;
        off = align_up(off + k * (n / 2));
        l.bwd_scales = off;
        l.bwd_scales_bytes = compute_nvfp4_cutlass_scale_size(static_cast<int>(K), static_cast<int>(N));
        off = align_up(off + l.bwd_scales_bytes);
        l.bwd_amax = off;
        off = align_up(off + sizeof(float));

        l.total = off;
        return l;
    }

    /// True when the blob fits inside the weight's BF16 master allocation
    /// (2*N*K bytes), i.e. the in-place quantization is safe.
    [[nodiscard]] bool fits_in_bf16_master(long N, long K) const {
        return total <= static_cast<std::size_t>(N) * static_cast<std::size_t>(K) * 2u;
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_NVFP4_STREAM_LAYOUT_H
