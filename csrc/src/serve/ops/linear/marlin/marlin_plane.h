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

// Same registry for the FP8 residency (codes [n,k] e4m3, per-channel BF16
// scales), which Marlin serves through its kFE4M3fn kernels at
// group_blocks = -1. The 27B's decode band lives here: its FP8 GEMMs are
// 47% of that model's device time at 412-503 GB/s.
MarlinPlane marlin_fp8_plane_for(const Weight& weight, cudaStream_t stream);

// Repacks a weight into Marlin tiles in place and stamps QuantLayout::MarlinTiles.
// Only for weights whose every consumer can be served from Marlin; see the note
// at the definition.
bool marlin_fp8_adopt_residency(Weight& weight, cudaStream_t stream);

// Adopts on first use from a Marlin-capable route; no-op during capture or when
// adoption is not enabled. Returns true when the weight holds Marlin tiles.
bool marlin_fp8_maybe_adopt(const Weight& weight, cudaStream_t stream);

// Staging bytes a fused projection needs when its weight holds Marlin tiles:
// Marlin emits the whole [parent_rows, T] parent and the fusion splits it
// afterwards. Zero when adoption is not enabled, so the plan does not carry
// the reservation on builds that will never use it.
std::size_t marlin_fused_parent_bytes(std::int32_t parent_rows, std::int32_t columns) noexcept;

struct MarlinScratch {
    void* gemm_out = nullptr;  // bf16 [kMarlinFixedM, n] row-major
    void* a_pad    = nullptr;  // bf16 [kMarlinFixedM, k], pad rows zeroed
    void* c_tmp    = nullptr;  // fp32 reduce buffer
    int* locks     = nullptr;  // zero-initialized lock array
    int sm_count   = 0;
};

// Every band call runs the GEMM at this fixed M, copying the round's real
// rows into a zero-padded A. Marlin picks its kernel from
// thread_m_blocks = min(ceil(M/16), 4), so a width-dependent M puts
// different kernels into decode graphs that share a topology class and
// cudaGraphExecUpdate rejects them (PATCHES.md #35). Pinning M removes that
// coupling, and it is nearly free: Marlin's cost is flat from M=24 to M=32
// (gate_up 42.1 vs 42.4 us measured). The pad rows compute garbage that no
// reader ever sees, since C's first t rows are exactly the [n, t] result.
// Set once at engine construction from the runtime lane count: the 4B gains
// 7.5% from a 64-wide band at 64 lanes while the 0.8B loses 11% paying that
// width on narrow shapes, so the width has to follow the deployment rather
// than a compile-time constant.
void marlin_set_fixed_m(int lanes) noexcept;
int marlin_fixed_m() noexcept;

// Band floor. Marlin wins by ~3x at the 4B's wide shapes even at T=16,
// while the exact-T split-K kernels stay competitive at the small ones;
// SUROGATE_SERVE_MARLIN_MIN_T overrides for sweeps.
int marlin_min_band_tokens() noexcept;
// Band ceiling: the fixed M every call pads to. Rounds wider than this fall
// back to the engine's own kernels; widening means raising kMarlinFixedM
// (one more zero-padded row block for every call), not splitting the band.


// Valid once any plane derived; null members otherwise.
MarlinScratch marlin_scratch() noexcept;

// Ensures a plane exists for `weight` (deriving outside capture) and returns
// the shared scratch; gemm_out is null when the weight has no usable plane.
MarlinScratch marlin_scratch_for(const Weight& weight, cudaStream_t stream);

// FP8 counterparts (PATCHES.md #38). marlin_fp8_scratch_for derives the plane
// first and returns the scratch after, since reading the scratch before the
// first derive sees it unallocated.
MarlinScratch marlin_fp8_scratch_for(const Weight& weight, cudaStream_t stream);
bool marlin_fp8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

// Called once after graph capture: the shared scratch may no longer move,
// because captured graphs have baked its addresses. Derives that would need
// a larger scratch afterwards decline instead (callers fall back).
void marlin_plane_freeze_scratch() noexcept;

// Runs the Marlin GEMM for `weight` through its derived plane into
// `out` ([n, t] BF16 column-major). Returns false (and does nothing) when
// no plane is available (caller falls back to its own kernels).
bool marlin_w8_run(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
