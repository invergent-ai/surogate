#pragma once

// Derived Marlin residency for W8G32 weights: the artifact stays W8G32 and
// at first decode-band use the engine repacks a weight into the Marlin B
// tile format plus permuted BF16 scales (see marlin_repack.h). Registry
// discipline mirrors w8fp8_plane.h: process-global, keyed by the device
// codes pointer, derive-on-first-use with a VRAM guard; a capturing stream
// may look up a finished plane (the pre-capture batch-band warmup derives
// and synchronizes it) but never derives. Shared per-call scratch (gemm
// output, fp32 reduce buffer, lock array) is allocated at first derive and
// its addresses stay fixed so captured graphs may bake them.

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

struct MarlinPlane {
    const void* b_packed = nullptr;  // marlin B tiles (u32)
    const void* scales   = nullptr;  // bf16, [k/32, n] permuted
};

void marlin_plane_set_enabled(bool enabled) noexcept;
bool marlin_plane_enabled() noexcept;
std::size_t marlin_plane_bytes() noexcept;

MarlinPlane marlin_plane_for(const Weight& weight, cudaStream_t stream);

struct MarlinScratch {
    void* gemm_out = nullptr;  // bf16, >= max_n * kMarlinMaxBandTokens
    void* c_tmp    = nullptr;  // fp32 reduce buffer
    int* locks     = nullptr;  // zero-initialized lock array
    int sm_count   = 0;
};

// Band floor. Marlin wins by ~3x at the 4B's wide shapes even at T=16,
// while the exact-T split-K kernels stay competitive at the small ones;
// SUROGATE_SERVE_MARLIN_MIN_T overrides for sweeps.
int marlin_min_band_tokens() noexcept;
// Band ceiling. Marlin picks its kernel from thread_m_blocks =
// min(ceil(M/16), 4), so a band spanning a 16-token boundary would put
// different kernels in decode graphs that share a topology class and the
// exec update rejects them. 17..32 is exactly one class (mb = 2).
inline constexpr int kMarlinMaxBandTokens = 32;

// Valid once any plane derived; null members otherwise.
MarlinScratch marlin_scratch() noexcept;

// Ensures a plane exists for `weight` (deriving outside capture) and returns
// the shared scratch; gemm_out is null when the weight has no usable plane.
MarlinScratch marlin_scratch_for(const Weight& weight, cudaStream_t stream);

// Called once after graph capture: the shared scratch may no longer move,
// because captured graphs have baked its addresses. Derives that would need
// a larger scratch afterwards decline instead (callers fall back).
void marlin_plane_freeze_scratch() noexcept;

// Runs the Marlin GEMM for `weight` through its derived plane into
// `out` ([n, t] BF16 column-major). Returns false (and does nothing) when
// no plane is available (caller falls back to its own kernels).
bool marlin_w8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
