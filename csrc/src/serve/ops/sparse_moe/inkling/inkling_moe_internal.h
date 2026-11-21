#pragma once

// Internal launchers of the Inkling MoE (api/ops/inkling_moe.h). Rows are addressed by SLOT
// index i = token * 8 + slot: the gate_up activations y, their quantised form and the down rows
// all live at their slot index, so the combine needs no map. `sorted_rows` lists the routed
// slots grouped by expert (offsets[e] .. offsets[e + 1]); the order inside an expert is free.

#include "api/ops/inkling_moe.h"
#include "api/ops/lora.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::inkling {

/// Live token count of a call: `*live` clamped to `tokens`, or `tokens`.
struct LiveTokens {
    const std::int32_t* device = nullptr;
    std::int32_t tokens        = 0;
};

struct RouteOutputs {
    std::int32_t* ids      = nullptr; ///< [T * 8] global expert id per slot, -1 = padding
    float* gammas          = nullptr; ///< [T * 8]
    std::int32_t* ranks    = nullptr; ///< [T * 8] row rank inside its expert (placement only)
    std::int32_t* counts   = nullptr; ///< [128] zeroed by the logits kernel (placement only)
    std::int32_t* ticket   = nullptr; ///< [1]   zeroed by the logits kernel (placement only)
    std::int32_t* offsets  = nullptr; ///< [129] (placement only)
    std::int32_t* sorted   = nullptr; ///< [T * 8] slots grouped by expert (placement only)
    std::int32_t* capture  = nullptr; ///< capture rows or null
    std::int32_t capture_capacity = 0;
    std::int32_t capture_first    = 0;
    std::int32_t capture_stride   = 8;
};

/// Router logits fp32 [T][128] (`(p0 + p1) + p2` of bf16 m16n8k16 chains per 1024-wide K group).
/// With `zero` non-null, block (0, 0) also zeroes `zero_words` int32 words there.
void launch_router_logits(const __nv_bfloat16* x, const __nv_bfloat16* router, float* logits,
                          std::int32_t tokens, std::int32_t* zero, std::int32_t zero_words,
                          cudaStream_t stream);

/// Selection, routing weights, capture; with `out.counts` set, also each expert's row ranks and
/// (last CTA) the offsets and the expert-grouped slot list.
void launch_router_select(const float* logits, const float* bias, LiveTokens live,
                          const RouteOutputs& out, cudaStream_t stream);

/// Per-row NVFP4 four-over-six of bf16 rows [rows][k]: codes [rows][k/2], scales [rows][k/16]
/// (row-major), amax [rows]. `row_ids` (nullable): rows whose id is negative are skipped;
/// `live` bounds the rows when `row_ids` is null.
void launch_quant_nvfp4(const __nv_bfloat16* input, std::int32_t rows, std::int32_t k,
                        const std::int32_t* row_ids, LiveTokens live, std::uint8_t* codes,
                        std::uint8_t* scales, float* amax, cudaStream_t stream);

/// Per-32-block MXFP8 of bf16 rows [rows][k]: any of codes [rows][k] (e4m3), exponents
/// [rows][k/32] (e8m0), dequantised bf16 [rows][k] (exact) may be null.
void launch_quant_mxfp8(const __nv_bfloat16* input, std::int32_t rows, std::int32_t k,
                        const std::int32_t* row_ids, LiveTokens live, std::uint8_t* codes,
                        std::uint8_t* exponents, __nv_bfloat16* dequantised, cudaStream_t stream);

/// Standalone SiTU over interleaved gate_up rows [rows][2304] -> y [rows][1152].
void launch_softcap_swiglu(const __nv_bfloat16* gate_up, const float* gamma, std::int32_t rows,
                           __nv_bfloat16* y, cudaStream_t stream);

/// out[t] = bf16(sum over slots in order of down[t * 8 + s]) for live tokens, 0 for padding.
void launch_combine(const __nv_bfloat16* down, const std::int32_t* ids, LiveTokens live,
                    __nv_bfloat16* out, cudaStream_t stream);

/// Adapter state of a round as the kernels read it.
struct LoraRound {
    const LoraBank* table          = nullptr; ///< device bank table (bind_lora_moe order)
    const std::int32_t* token_slot = nullptr;
    const std::int32_t* uniform_slot = nullptr;
    std::int32_t rank              = 0;       ///< padded rank (<= 64)
    float* low                     = nullptr; ///< [T * 8][low_stride] fp32
    std::int32_t low_stride        = 0;
};

/// low[i][b * rank + r] = A_b[e(i)][slot] . src(i) for the banks b of `projection` (gate_up:
/// gate then up, source = x row of the token; down: down, source = y row of the slot).
void launch_lora_shrink(const __nv_bfloat16* source, std::int32_t source_k, bool gate_up,
                        const std::int32_t* ids, LiveTokens live, const LoraRound& lora,
                        cudaStream_t stream);

/// Grouped expert GEMM operands of one projection.
struct GemmParams {
    // Activations (source row = slot >> a_row_shift).
    const std::uint8_t* a_codes  = nullptr; ///< NVFP4 [rows][k/2]
    const std::uint8_t* a_scales = nullptr; ///< NVFP4 [rows][k/16] row-major
    const float* a_amax          = nullptr; ///< NVFP4 [rows]
    const __nv_bfloat16* a_bf16  = nullptr; ///< MXFP8: exactly dequantised activations [rows][k]
    std::int32_t a_row_shift     = 0;
    // Routing.
    const std::int32_t* offsets  = nullptr; ///< [129]
    const std::int32_t* sorted   = nullptr; ///< expert-grouped slots
    const float* gammas          = nullptr; ///< [T * 8] (gate_up only)
    // Weights.
    InklingExpertBank weights;
    // Output (gate_up: y bf16 [T * 8][1152]; down: bf16 [T * 8][2304]).
    __nv_bfloat16* out = nullptr;
    // Adapter (table null = none).
    LoraRound lora;
};

/// Tile schedules: Narrow for decode / verify widths, Wide for prefill. Same arithmetic.
enum class GemmSchedule : std::uint8_t { Narrow, Wide };

/// The token rows one tile of `schedule` carries (the routed rows of an expert are cut into
/// tiles of this many).
[[nodiscard]] std::int32_t gemm_tile_rows(InklingExpertFormat format, GemmSchedule schedule);
[[nodiscard]] std::int32_t nvfp4_tile_rows(GemmSchedule schedule);
[[nodiscard]] std::int32_t mxfp8_tile_rows(GemmSchedule schedule);

void launch_nvfp4_gate_up(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                          cudaStream_t stream);
void launch_nvfp4_down(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                       cudaStream_t stream);
void launch_mxfp8_gate_up(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                          cudaStream_t stream);
void launch_mxfp8_down(const GemmParams& params, std::int32_t slots, GemmSchedule schedule,
                       cudaStream_t stream);

} // namespace sinfer::ops::detail::inkling
