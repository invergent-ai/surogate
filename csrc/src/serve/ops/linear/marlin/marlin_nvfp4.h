#pragma once

// Weight-only NVFP4 (QType::NVFP4_A16) on the vendored Marlin kernels.
//
// A ModelOpt W4A16_NVFP4 export stores E2M1 codes, one E4M3 block scale per 16 values and an
// FP32 second level that multiplies, and leaves the activations in BF16. vLLM serves exactly
// these weights through Marlin on every GPU (MarlinNvFp4LinearKernel for a dense layer, the
// MARLIN NVFP4 MoE backend for the experts), and so does this: at load the stored words are
// rewritten in place into Marlin's B tiles and its scale encoding -- size-exact against the
// stored planes, so the residency costs nothing extra -- and the weight is stamped
// QuantLayout::MarlinTiles, after which only the Marlin routes read it.
//
// The rewrite is vLLM's (marlin_utils_fp4.py) step for step, so a token sees the numbers the
// reference serves:
//   codes        [n, k/2] -> GPTQ [k/8, n] u32 -> gptq_marlin_repack (4-bit tiles);
//   block scales e4m3 [n, k/16] -> transposed [k/16, n] -> the 64-wide group permutation ->
//                the [0, 2, 1, 3] reorder of each four -> times a power of two `sf` that lifts
//                the bank's largest scale to the top of E4M3 -> times 2^7 -> values below 2
//                flushed -> the high byte of (fp16 bits << 1), Marlin's "S0E5M3" encoding;
//   second level multiplier * 2^(2^7 - 2^1 - 7) / sf, the exponent bias the FP4 -> BF16
//                dequant leaves out, as one FP32 per row block.
// `sf` is shared by every block of one weight (vLLM computes it over all experts of a bank).
// The one lossy step is the reference's own: an E4M3 block scale below 2^-6 (after `sf`) has no
// S0E5M3 encoding and becomes zero, exactly as vLLM serves it.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail {

/// Whether `weight` is NVFP4_A16 already rewritten into Marlin tiles.
[[nodiscard]] bool nvfp4_marlin_ready(const Weight& weight) noexcept;

/// Rewrites a stored NVFP4_A16 weight -- `blocks` stacked matrices of `weight.n / blocks` rows
/// each (a routed bank's experts; one for a dense matrix) -- into Marlin tiles in place, sets
/// `weight.marlin_global_scale` to its per-block global scales (owned by the engine and device)
/// and stamps it MarlinTiles. Block b's second level is `multipliers[b * stride]` (device FP32);
/// null reads the payload's trailing word, a dense matrix's own. A weight already in Marlin
/// tiles is left as it is. Load time only: it allocates staging and synchronises `stream`.
void nvfp4_marlin_adopt(Weight& weight, std::int32_t blocks, const float* multipliers,
                        std::int32_t stride, cudaStream_t stream);

/// out[N, T] = weight[N, K] . x[K, T] for a dense NVFP4_A16 weight in Marlin tiles, at any T.
/// x is [K, T] contiguous BF16, which is Marlin's row-major [T, K] A; out [N, T] is its [T, N] C.
void nvfp4_marlin_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

/// The per-device scratch every NVFP4_A16 Marlin call reduces through: the FP32 reduce buffer
/// (sized for the widest MoE block) and the zero-initialised lock words the kernels self-reset.
/// Allocated by the first adoption on a device, so its addresses are fixed before any capture;
/// calls on one device are ordered on its one stream, as the W8 Marlin scratch assumes.
struct Nvfp4MarlinScratch {
    void* c_tmp       = nullptr;
    std::size_t c_tmp_bytes = 0;
    int* locks        = nullptr;
    int sm_count      = 0;
};

[[nodiscard]] Nvfp4MarlinScratch nvfp4_marlin_scratch();

} // namespace sinfer::ops::detail
