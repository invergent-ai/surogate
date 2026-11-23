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
#include <cstdlib>
#include <stdexcept>

namespace sinfer::ops::detail::inkling {

inline constexpr int kHidden       = InklingMoeGeometry::kHidden;
inline constexpr int kExperts      = InklingMoeGeometry::kExperts;
inline constexpr int kTopK         = InklingMoeGeometry::kTopK;
inline constexpr int kIntermediate = InklingMoeGeometry::kIntermediate;
inline constexpr int kGateUpRows   = InklingMoeGeometry::kGateUpRows;
inline constexpr int kRouterGroup  = InklingMoeGeometry::kRouterKGroup;
/// Arrival counters per token of the fused combine: one per 64-column tile of the down output
/// (the narrowest column tile of any down kernel).
inline constexpr int kArrivalTiles = kHidden / 64;

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
    /// 1: the last select CTA also writes `sorted` (narrow rounds, <= 512 slots); 0: only the
    /// offsets, and `launch_router_place` scatters the slots in parallel (wide rounds).
    std::int32_t last_cta_places  = 1;
    /// Fused combine: [T][kArrivalTiles] counters zeroed here per token; padding tokens' output
    /// rows (bf16 [T][2304]) written as zeros here (no down row ever arrives for them).
    std::int32_t* arrivals        = nullptr;
    __nv_bfloat16* combined       = nullptr;
    /// Device-side only (one-tile fused route): the selector's shared-memory copy of the ids.
    std::int32_t* local_ids       = nullptr;
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

/// Narrow rounds: router logits (as launch_router_logits) and the x quantisation (NVFP4: codes /
/// scales / amax; MXFP8: the dequantised bf16 rows) in one grid.
void launch_route_quant_narrow(const __nv_bfloat16* x, const __nv_bfloat16* router, float* logits,
                               std::int32_t tokens, std::int32_t* zero, std::int32_t zero_words, LiveTokens live,
                               bool nvfp4, std::uint8_t* x_codes, std::uint8_t* x_scales, float* x_amax,
                               __nv_bfloat16* x_dequantised, cudaStream_t stream);

/// Persistent state words (InklingMoeOptions::state): [0, 128) expert counts, [128] ticket, [129, 137) the
/// arrival counter of each 8-token router tile. Zero between calls (each call returns them to zero).
inline constexpr int kStateCountsWord   = 0;
inline constexpr int kStateTicketWord   = kExperts;
inline constexpr int kStateTileWord     = kExperts + 1;
inline constexpr int kNarrowStateMaxTokens = 64;
inline constexpr int kStateWords        = kStateTileWord + kNarrowStateMaxTokens / 8;

/// Narrow rounds with the persistent state: router logits, x quantisation and the selection (+ offsets and
/// placement by the last selector) in one grid; `out.counts` / `out.ticket` and `tile_arrivals` point into
/// the state and are zero on entry and on exit.
void launch_route_select_narrow(const __nv_bfloat16* x, const __nv_bfloat16* router, const float* bias,
                                float* logits, LiveTokens live, bool nvfp4, std::uint8_t* x_codes,
                                std::uint8_t* x_scales, float* x_amax, __nv_bfloat16* x_dequantised,
                                const RouteOutputs& out, std::int32_t* tile_arrivals, cudaStream_t stream);

/// sorted[offsets[ids[i]] + ranks[i]] = i for every routed slot i (after the select kernel).
void launch_router_place(const RouteOutputs& out, std::int32_t tokens, cudaStream_t stream);

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

/// Unit weights of experts [first, first + count) of `bank` (K = k) -> bf16 [count * 2304][k], and the
/// per-expert factor [count] (null to skip). See `inkling_decode_expert_units`.
void launch_decode_expert_units(const InklingExpertBank& bank, bool nvfp4, std::int32_t k, std::int32_t first,
                                std::int32_t count, __nv_bfloat16* unit, float* factor, cudaStream_t stream);

/// Test-only row-load mode of the PDL audit (inkling_test_read_only_loads): 0 plain, 1 ld.nc, 2 ld.cg; one
/// host-side mode (row_load_mode_host) that picks the kernel instantiations; serving always launches mode 0.
void set_read_only_control(int mode);
void set_row_load_mode_nvfp4(int mode);
/// Test-only planter kernel (inkling_test_plant_reads).
void launch_test_plant_reads(const void* buffer, std::int64_t bytes, int* sink, cudaStream_t stream);
/// Test-only PDL writer kernel (inkling_test_copy_rows).
void launch_test_copy_rows(const void* source, void* destination, std::int64_t bytes, int hostile, int* sink,
                           cudaStream_t stream);

/// inkling_bench_predecessor's kernel (benchmark only).
void launch_bench_predecessor(std::int32_t nanoseconds, cudaStream_t stream);

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
    /// Test tap (gate_up only): the bf16 gate_up rows [T * 8][2304], interleaved. Null in serving.
    __nv_bfloat16* gate_up_tap = nullptr;
    /// Down only: fused combine into `combined` (bf16 [T][2304]) through the arrival counters
    /// [T][kArrivalTiles] (zeroed by the select kernel). Null = rows only.
    std::int32_t* arrivals     = nullptr;
    __nv_bfloat16* combined    = nullptr;
    /// Down output buffer (prefill column passes): `out` holds output columns [out_column0, out_column0 +
    /// out_stride) of every slot row, row stride out_stride elements. Default = whole rows. The launchers derive
    /// the N-tile range (n_tile0, n_tiles) of the launch from it; the combine reads the rows back the same way.
    std::int32_t out_stride  = kHidden;
    std::int32_t out_column0 = 0;
    std::int32_t n_tile0     = 0;
    std::int32_t n_tiles     = 0; ///< 0 = every N tile
    // Adapter (table null = none).
    LoraRound lora;
    /// L2 prefetch-size hint for the weight-code loads (0 none, 128, 256): same bytes (with_l2_prefetch).
    std::int32_t l2_prefetch = 0;
    /// Narrow NVFP4 down WITHOUT the fused y quantisation (rounds of T >= kPlainDownMinTokens): A = the y
    /// rows quantised once by the standalone kernel (a_codes / a_scales / a_amax = y_codes / y_scales / y_amax); the
    /// routing is read and the w2 stages are requested after the dependency wait. Same arithmetic.
    std::int32_t plain_narrow_down = 0;
};

/// The weight loads' L2 prefetch hint (`GemmParams::l2_prefetch`): 128 bytes. Measured 12:54 (card 7, cold,
/// layer 0 NVFP4): T = 8 151.6 -> 131.0 us, T = 64 336.9 -> 291.1, T = 1 36.9 -> 35.7; 256 B the same; T = 4,096
/// and MXFP8 unchanged. SUROGATE_INKLING_L2_PREFETCH = 0 | 256 overrides it (A/B only). Same bytes, same bits.
inline GemmParams with_l2_prefetch(GemmParams params) {
    static const int value = [] {
        const char* text = std::getenv("SUROGATE_INKLING_L2_PREFETCH");
        if (text == nullptr) { return 128; }
        const int v = std::atoi(text);
        return v == 128 || v == 256 ? v : 0;
    }();
    params.l2_prefetch = value;
    return params;
}

/// The N-tile range of a launch with `block_n`-row output tiles (from out_column0 / out_stride; whole rows = all).
inline void set_n_tiles(GemmParams& params, int block_n) {
    if (params.out_column0 % block_n != 0 || params.out_stride % block_n != 0) {
        throw std::invalid_argument("inkling_moe: a down column pass must cover whole N tiles");
    }
    params.n_tile0 = params.out_column0 / block_n;
    params.n_tiles = params.out_stride / block_n;
}

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
