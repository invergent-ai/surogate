#pragma once
// sinfer::ops - the batch-consistent W8G32 route's Hopper kernel (w8_rowsplit_wgmma_sm90.cu): the
// medium-T kernel's arithmetic with each K step on wgmma instead of mma.sync, so its outputs are
// that kernel's bits. sm_90a-only: builds without 90a link a stub that declines every call.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

struct W8WgmmaProblem {
    const __nv_bfloat16* x      = nullptr; ///< [columns][k] BF16
    const std::uint8_t* codes   = nullptr; ///< [n][k] int8 codes
    const std::uint8_t* scales  = nullptr; ///< [n][k / 32] FP16 group scales
    __nv_bfloat16* out          = nullptr; ///< [columns][n] BF16
    std::int32_t n = 0, k = 0, columns = 0;
};

/// Whether this build and the current device run the kernel: sm_90a code is built in, the device
/// is an sm_90 part, and SUROGATE_SERVE_W8_WGMMA is not "0".
[[nodiscard]] bool w8_wgmma_available() noexcept;

/// The narrowest round the consistent route sends here (SUROGATE_SERVE_W8_WGMMA_MIN_COLUMNS,
/// default 33; up to 32 columns the pipelined kernel stays faster).
[[nodiscard]] std::int32_t w8_wgmma_min_columns() noexcept;

/// Runs `p` (1..64 columns, 64 rows a CTA) and returns true, or returns false without launching
/// when the kernel is not available or the shape does not tile: K a multiple of 256, N of 64 rows.
bool w8_wgmma_consistent(const W8WgmmaProblem& p, cudaStream_t stream);

} // namespace sinfer::ops::detail
