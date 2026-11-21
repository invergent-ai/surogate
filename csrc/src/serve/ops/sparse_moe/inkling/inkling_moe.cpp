// Inkling MoE op wrapper: workspace carving, routing, expert GEMMs, combine (api/ops/inkling_moe.h).
//
// Launch order of one call (each kernel waits for its predecessor, PDL):
//   router logits (also zeroes the expert counters) -> x quantisation -> selection + placement
//   [-> placement (wide rounds)] [-> LoRA shrink (w13)] -> gate_up + SiTU -> y quantisation
//   [-> LoRA shrink (w2)] -> down with the combine fused (the CTA completing a token's 8 rows of a
//   column tile sums them in slot order).
// The fusions of DESIGN-moe.md 2.1 (route + quant in one grid, quantise y in down's prologue,
// combine in down) change no arithmetic and come with the M3 timing.

#include "api/ops/inkling_moe.h"
#include "api/ops/sparse_moe.h"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace detail::inkling {

std::int32_t gemm_tile_rows(InklingExpertFormat format, GemmSchedule schedule) {
    return format == InklingExpertFormat::Nvfp4 ? nvfp4_tile_rows(schedule) : mxfp8_tile_rows(schedule);
}

} // namespace detail::inkling

namespace {

using namespace detail::inkling;

constexpr std::size_t kAlignment = 256;
/// Rounds below this width take the narrow tile schedules (same arithmetic; a speed choice).
constexpr std::int32_t kNarrowMaxTokens = 64;

constexpr std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

struct Buffers {
    float* logits             = nullptr;
    std::int32_t* ids         = nullptr;
    float* gammas             = nullptr;
    std::int32_t* ranks       = nullptr;
    std::int32_t* counters    = nullptr; ///< [128 counts | ticket]
    std::int32_t* arrivals    = nullptr; ///< [T][kArrivalTiles] fused-combine counters
    std::int32_t* offsets     = nullptr;
    std::int32_t* sorted      = nullptr;
    std::uint8_t* x_codes     = nullptr;
    std::uint8_t* x_scales    = nullptr;
    float* x_amax             = nullptr;
    __nv_bfloat16* x_bf16     = nullptr;
    __nv_bfloat16* y          = nullptr;
    std::uint8_t* y_codes     = nullptr;
    std::uint8_t* y_scales    = nullptr;
    float* y_amax             = nullptr;
    __nv_bfloat16* y_bf16     = nullptr;
    __nv_bfloat16* down       = nullptr;
    float* low_gate_up        = nullptr;
    float* low_down           = nullptr;
};

constexpr std::int32_t kCounterWords = kExperts + 1;

/// Prefill (T > kNarrowMaxTokens): the down GEMM and its fused combine may run in column passes, so the down rows
/// buffer holds 2304 / passes columns per slot. Allowed pass counts cover whole N tiles of every down schedule
/// (NVFP4 768 = 3 x 256 / 6 x 128; MXFP8 1152 / 768 / 384 = 9 / 6 / 3 x 128). The arrival counters are per (token,
/// 64-column tile), so the combine of a column tile completes inside its pass; every weight byte is still read once.
/// The most passes = the compact layout `inkling_moe_workspace_bytes` books; a call uses the fewest passes whose
/// layout fits what the caller booked (InklingMoeOptions::workspace_tokens), measured 15:5x: 6 passes cost MXFP8
/// +10 % at T = 256, ~0 at 8,192.
constexpr std::int32_t kNvfp4DownPasses[] = {1, 3};
constexpr std::int32_t kMxfp8DownPasses[] = {1, 2, 3, 6};

constexpr std::int32_t max_down_passes(InklingExpertFormat format) {
    return format == InklingExpertFormat::Nvfp4 ? 3 : 6;
}

template <class Allocate>
Buffers carve(InklingExpertFormat format, std::int32_t tokens, std::int32_t rank, std::int32_t passes,
              Allocate&& allocate) {
    const std::size_t t     = static_cast<std::size_t>(tokens);
    const std::size_t slots = t * kTopK;
    const bool nvfp4        = format == InklingExpertFormat::Nvfp4;
    if (tokens > kNarrowMaxTokens) {
        // Prefill layout with aliasing (lifetimes in launch order: logits -> x quant -> select / place ->
        // gate_up -> y quant -> down passes):
        //   region A: logits (dead after the selection) | y bf16 (gate_up -> y quant; MXFP8 quantises y IN
        //             PLACE, so A also holds y_bf16 for the down) | NVFP4: the down pass rows (y is dead after the
        //             y quant);
        //   region B (MXFP8): x_bf16 (dead after gate_up) | the down pass rows.
        const std::size_t pass_columns = static_cast<std::size_t>(kHidden / passes);
        const std::size_t logits_bytes = t * kExperts * sizeof(float);
        const std::size_t y_bytes      = slots * kIntermediate * sizeof(__nv_bfloat16);
        const std::size_t pass_bytes   = slots * pass_columns * sizeof(__nv_bfloat16);
        Buffers b;
        b.ids      = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
        b.gammas   = static_cast<float*>(allocate(slots * sizeof(float)));
        b.ranks    = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
        b.counters = static_cast<std::int32_t*>(allocate(kCounterWords * sizeof(std::int32_t)));
        b.arrivals = static_cast<std::int32_t*>(allocate(t * kArrivalTiles * sizeof(std::int32_t)));
        b.offsets  = static_cast<std::int32_t*>(allocate((kExperts + 1) * sizeof(std::int32_t)));
        b.sorted   = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
        if (nvfp4) {
            auto* region = allocate(std::max({logits_bytes, y_bytes, pass_bytes}));
            b.logits     = static_cast<float*>(region);
            b.y          = static_cast<__nv_bfloat16*>(region);
            b.down       = static_cast<__nv_bfloat16*>(region);
            b.x_codes    = static_cast<std::uint8_t*>(allocate(t * (kHidden / 2)));
            b.x_scales   = static_cast<std::uint8_t*>(allocate(t * (kHidden / 16)));
            b.x_amax     = static_cast<float*>(allocate(t * sizeof(float)));
            b.y_codes    = static_cast<std::uint8_t*>(allocate(slots * (kIntermediate / 2)));
            b.y_scales   = static_cast<std::uint8_t*>(allocate(slots * (kIntermediate / 16)));
            b.y_amax     = static_cast<float*>(allocate(slots * sizeof(float)));
        } else {
            auto* region_a = allocate(std::max(logits_bytes, y_bytes));
            b.logits       = static_cast<float*>(region_a);
            b.y            = static_cast<__nv_bfloat16*>(region_a);
            b.y_bf16       = static_cast<__nv_bfloat16*>(region_a);
            auto* region_b = allocate(std::max(t * kHidden * sizeof(__nv_bfloat16), pass_bytes));
            b.x_bf16       = static_cast<__nv_bfloat16*>(region_b);
            b.down         = static_cast<__nv_bfloat16*>(region_b);
        }
        if (rank > 0) {
            b.low_gate_up = static_cast<float*>(allocate(slots * 2 * static_cast<std::size_t>(rank) * sizeof(float)));
            b.low_down    = static_cast<float*>(allocate(slots * static_cast<std::size_t>(rank) * sizeof(float)));
        }
        return b;
    }
    Buffers b;
    b.logits   = static_cast<float*>(allocate(t * kExperts * sizeof(float)));
    b.ids      = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
    b.gammas   = static_cast<float*>(allocate(slots * sizeof(float)));
    b.ranks    = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
    b.counters = static_cast<std::int32_t*>(allocate(kCounterWords * sizeof(std::int32_t)));
    b.arrivals = static_cast<std::int32_t*>(allocate(t * kArrivalTiles * sizeof(std::int32_t)));
    b.offsets  = static_cast<std::int32_t*>(allocate((kExperts + 1) * sizeof(std::int32_t)));
    b.sorted   = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
    if (nvfp4) {
        b.x_codes  = static_cast<std::uint8_t*>(allocate(t * (kHidden / 2)));
        b.x_scales = static_cast<std::uint8_t*>(allocate(t * (kHidden / 16)));
        b.x_amax   = static_cast<float*>(allocate(t * sizeof(float)));
    } else {
        b.x_bf16 = static_cast<__nv_bfloat16*>(allocate(t * kHidden * sizeof(__nv_bfloat16)));
    }
    b.y = static_cast<__nv_bfloat16*>(allocate(slots * kIntermediate * sizeof(__nv_bfloat16)));
    if (nvfp4) {
        b.y_codes  = static_cast<std::uint8_t*>(allocate(slots * (kIntermediate / 2)));
        b.y_scales = static_cast<std::uint8_t*>(allocate(slots * (kIntermediate / 16)));
        b.y_amax   = static_cast<float*>(allocate(slots * sizeof(float)));
    } else {
        b.y_bf16 = static_cast<__nv_bfloat16*>(allocate(slots * kIntermediate * sizeof(__nv_bfloat16)));
    }
    b.down = static_cast<__nv_bfloat16*>(allocate(slots * kHidden * sizeof(__nv_bfloat16)));
    if (rank > 0) {
        b.low_gate_up = static_cast<float*>(allocate(slots * 2 * static_cast<std::size_t>(rank) * sizeof(float)));
        b.low_down    = static_cast<float*>(allocate(slots * static_cast<std::size_t>(rank) * sizeof(float)));
    }
    return b;
}

const __nv_bfloat16* as_bf16(const void* pointer) { return static_cast<const __nv_bfloat16*>(pointer); }

void require_tensor(const Tensor& t, DType dtype, std::int32_t inner, std::int32_t outer, const char* what) {
    if (t.data == nullptr || t.dtype != dtype || t.ne[0] != inner || t.ne[1] != outer || !t.is_contiguous()) {
        throw std::invalid_argument(std::string("inkling_moe: ") + what + " has the wrong dtype, shape or layout");
    }
}

void require_weights(const InklingMoeWeights& w) {
    if (w.router == nullptr || w.router_bias == nullptr) {
        throw std::invalid_argument("inkling_moe: the router weight and bias are required");
    }
    if (w.w13.codes == nullptr || w.w13.scales == nullptr || w.w2.codes == nullptr || w.w2.scales == nullptr) {
        throw std::invalid_argument("inkling_moe: both expert banks need codes and scales");
    }
    if (w.format == InklingExpertFormat::Nvfp4 &&
        (w.w13.amax == nullptr || w.w13.outer_den == nullptr || w.w2.amax == nullptr ||
         w.w2.outer_den == nullptr)) {
        throw std::invalid_argument("inkling_moe: NVFP4 banks need the raw amax and outer_scale_denominator");
    }
}

/// Bytes of one call's layout (carve) with `passes` down column passes, incl. the alignment slack.
std::size_t layout_bytes(InklingExpertFormat format, std::int32_t tokens, std::int32_t rank, std::int32_t passes) {
    std::size_t total = kAlignment; // the arena's offset may be unaligned when the call starts
    carve(format, tokens, rank, passes, [&](std::size_t bytes) -> void* {
        total += aligned(bytes);
        return nullptr;
    });
    return total;
}

/// The fewest down column passes whose layout fits what the caller booked for `workspace_tokens` (0 = unknown:
/// the compact layout of this call).
std::int32_t choose_down_passes(InklingExpertFormat format, std::int32_t tokens, std::int32_t rank,
                                std::int32_t workspace_tokens) {
    if (tokens <= kNarrowMaxTokens) { return 1; }
    const std::int32_t most = max_down_passes(format);
    if (workspace_tokens <= 0) { return most; }
    const std::size_t booked =
        layout_bytes(format, std::max(workspace_tokens, tokens), rank, most);
    const bool nvfp4 = format == InklingExpertFormat::Nvfp4;
    const std::int32_t* first = nvfp4 ? std::begin(kNvfp4DownPasses) : std::begin(kMxfp8DownPasses);
    const std::int32_t* last  = nvfp4 ? std::end(kNvfp4DownPasses) : std::end(kMxfp8DownPasses);
    for (const std::int32_t* passes = first; passes != last; ++passes) {
        if (layout_bytes(format, tokens, rank, *passes) <= booked) { return *passes; }
    }
    return most;
}

} // namespace

std::size_t inkling_moe_state_bytes() {
    return aligned(static_cast<std::size_t>(kStateWords) * sizeof(std::int32_t));
}

std::size_t inkling_moe_workspace_bytes(InklingExpertFormat format, std::int32_t max_tokens, bool adapters) {
    if (max_tokens <= 0) { throw std::invalid_argument("inkling_moe_workspace_bytes: max_tokens must be positive"); }
    // The largest layout of any width <= max_tokens: the narrow one at min(max_tokens, 64) and, above 64,
    // the prefill one at max_tokens (both grow with T).
    const std::int32_t rank = adapters ? InklingMoeGeometry::kMaxAdapterRank : 0;
    std::size_t total       = layout_bytes(format, std::min(max_tokens, kNarrowMaxTokens), rank, 1);
    if (max_tokens > kNarrowMaxTokens) {
        total = std::max(total, layout_bytes(format, max_tokens, rank, max_down_passes(format)));
    }
    return total;
}

void inkling_moe(const Tensor& x, const InklingMoeWeights& weights, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream, const InklingMoeOptions& options) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("inkling_moe: T must be positive"); }
    require_tensor(x, DType::BF16, kHidden, tokens, "x");
    require_tensor(out, DType::BF16, kHidden, tokens, "out");
    require_weights(weights);
    const bool nvfp4 = weights.format == InklingExpertFormat::Nvfp4;

    const InklingMoeAdapters* adapters = options.adapters;
    const bool adapted = adapters != nullptr && adapters->table != nullptr;
    const std::int32_t rank = adapted ? adapters->max_rank : 0;
    if (adapted && (rank <= 0 || rank > InklingMoeGeometry::kMaxAdapterRank)) {
        throw std::invalid_argument("inkling_moe: expert LoRA rank must be in [1, 64]");
    }
    if (adapted && adapters->token_slot == nullptr && adapters->uniform_slot == nullptr) {
        throw std::invalid_argument("inkling_moe: an adapter round needs token slots or a uniform slot");
    }

    const std::int32_t passes = choose_down_passes(weights.format, tokens, rank, options.workspace_tokens);
    const Buffers b = carve(weights.format, tokens, rank, passes, [&](std::size_t bytes) -> void* {
        return workspace.alloc_bytes(bytes, kAlignment).data;
    });
    const LiveTokens live{options.live_tokens, tokens};
    const auto* input = as_bf16(x.data);
    const std::int32_t slots = tokens * kTopK;

    // Router logits (the same launch zeroes the expert counters the selection counts into) and x
    // quantised once per token (the reference writes it once per routed row; the bits are equal).
    // Narrow rounds run both in one grid.
    const bool narrow = tokens <= kNarrowMaxTokens;
    static_assert(kNarrowMaxTokens <= kNarrowStateMaxTokens);
    const bool fused_select = narrow && options.state != nullptr;
    if (narrow && !fused_select) {
        launch_route_quant_narrow(input, as_bf16(weights.router), b.logits, tokens, b.counters, kCounterWords, live,
                                  nvfp4, b.x_codes, b.x_scales, b.x_amax, b.x_bf16, stream);
    } else if (!narrow) {
        launch_router_logits(input, as_bf16(weights.router), b.logits, tokens, b.counters, kCounterWords, stream);
        if (nvfp4) {
            launch_quant_nvfp4(input, tokens, kHidden, nullptr, live, b.x_codes, b.x_scales, b.x_amax, stream);
        } else {
            launch_quant_mxfp8(input, tokens, kHidden, nullptr, live, nullptr, nullptr, b.x_bf16, stream);
        }
    }

    RouteOutputs route;
    route.ids     = b.ids;
    route.gammas  = b.gammas;
    route.ranks   = b.ranks;
    route.counts  = b.counters;
    route.ticket  = b.counters + kExperts;
    route.offsets = b.offsets;
    route.sorted  = b.sorted;
    route.arrivals = b.arrivals;
    route.combined = static_cast<__nv_bfloat16*>(out.data);
    if (options.capture != nullptr && options.capture->ids != nullptr) {
        if (options.capture->token_stride < kTopK) {
            throw std::invalid_argument("inkling_moe: capture token_stride must be at least 8");
        }
        route.capture          = options.capture->ids;
        route.capture_capacity = options.capture->capacity_tokens;
        route.capture_first    = options.capture->first_token;
        route.capture_stride   = options.capture->token_stride;
    } else if (const SparseMoeCaptureTarget* target = sparse_moe_capture_target();
               target != nullptr && target->ids != nullptr) {
        if (target->top_k != kTopK) { throw std::invalid_argument("inkling_moe: capture target top_k must be 8"); }
        route.capture          = target->ids;
        route.capture_capacity = target->capacity_tokens;
        route.capture_first    = 0;
        route.capture_stride   = kTopK;
    }
    route.last_cta_places = narrow ? 1 : 0;
    if (fused_select) {
        route.counts = options.state + kStateCountsWord;
        route.ticket = options.state + kStateTicketWord;
        launch_route_select_narrow(input, as_bf16(weights.router), weights.router_bias, b.logits, live, nvfp4,
                                   b.x_codes, b.x_scales, b.x_amax, b.x_bf16, route, options.state + kStateTileWord,
                                   stream);
    } else {
        launch_router_select(b.logits, weights.router_bias, live, route, stream);
    }
    if (!narrow) { launch_router_place(route, tokens, stream); }

    // Test taps are copied right after the kernel that produces them: the prefill layout reuses the logits /
    // y / x regions later in the call (carve).
    const InklingMoeTaps* taps = options.taps;
    const std::size_t s        = static_cast<std::size_t>(slots);
    const auto copy = [&](void* destination, const void* source, std::size_t bytes) {
        if (destination == nullptr || source == nullptr) { return; }
        const cudaError_t status = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("inkling_moe taps: ") + cudaGetErrorString(status));
        }
    };
    if (taps != nullptr) {
        copy(taps->logits, b.logits, static_cast<std::size_t>(tokens) * kExperts * sizeof(float));
        copy(taps->ids, b.ids, s * sizeof(std::int32_t));
        copy(taps->gammas, b.gammas, s * sizeof(float));
    }

    LoraRound lora;
    if (adapted) {
        lora.table        = adapters->table;
        lora.token_slot   = adapters->token_slot;
        lora.uniform_slot = adapters->uniform_slot;
        lora.rank         = rank;
        lora.low          = b.low_gate_up;
        lora.low_stride   = 2 * rank;
        launch_lora_shrink(input, kHidden, true, b.ids, live, lora, stream);
    }

    const GemmSchedule schedule = narrow ? GemmSchedule::Narrow : GemmSchedule::Wide;
    GemmParams gate_up;
    gate_up.a_codes     = b.x_codes;
    gate_up.a_scales    = b.x_scales;
    gate_up.a_amax      = b.x_amax;
    gate_up.a_bf16      = b.x_bf16;
    gate_up.a_row_shift = 3; // slot -> token
    gate_up.offsets     = b.offsets;
    gate_up.sorted      = b.sorted;
    gate_up.gammas      = b.gammas;
    gate_up.weights     = weights.w13;
    gate_up.out         = b.y;
    gate_up.lora        = lora;
    gate_up.gate_up_tap = taps != nullptr ? static_cast<__nv_bfloat16*>(taps->gate_up) : nullptr;
    if (nvfp4) {
        launch_nvfp4_gate_up(gate_up, slots, schedule, stream);
    } else {
        launch_mxfp8_gate_up(gate_up, slots, schedule, stream);
    }
    if (taps != nullptr) { copy(taps->y, b.y, s * kIntermediate * sizeof(__nv_bfloat16)); }

    // The down adapter's shrink reads the bf16 y rows: before the y quantisation (MXFP8 prefill quantises y in
    // place). Narrow rounds keep the order quant -> shrink (no aliasing there).
    const auto down_shrink = [&] {
        if (!adapted) { return; }
        lora.low        = b.low_down;
        lora.low_stride = rank;
        launch_lora_shrink(b.y, kIntermediate, false, b.ids, live, lora, stream);
    };
    if (!narrow) { down_shrink(); }
    if (nvfp4) {
        // Narrow rounds quantise y inside the down kernel; the standalone pass then runs only for test taps.
        if (!narrow || taps != nullptr) {
            launch_quant_nvfp4(b.y, slots, kIntermediate, b.ids, live, b.y_codes, b.y_scales, b.y_amax, stream);
        }
    } else {
        launch_quant_mxfp8(b.y, slots, kIntermediate, b.ids, live, taps != nullptr ? taps->y_codes : nullptr,
                           taps != nullptr ? taps->y_scales : nullptr, b.y_bf16, stream);
    }
    if (narrow) { down_shrink(); }
    if (taps != nullptr && nvfp4) {
        copy(taps->y_codes, b.y_codes, s * (kIntermediate / 2));
        copy(taps->y_scales, b.y_scales, s * (kIntermediate / 16));
        copy(taps->y_amax, b.y_amax, s * sizeof(float));
    }

    GemmParams down;
    down.a_codes     = b.y_codes;
    down.a_scales    = b.y_scales;
    down.a_amax      = b.y_amax;
    down.a_bf16      = nvfp4 ? b.y : b.y_bf16; // NVFP4: the bf16 y rows the narrow down kernel quantises
    down.a_row_shift = 0; // slot rows
    down.offsets     = b.offsets;
    down.sorted      = b.sorted;
    down.weights     = weights.w2;
    down.out         = b.down;
    down.lora        = lora;
    down.arrivals    = b.arrivals; // combine fused into down (summed in slot order by the last arriver)
    down.combined    = static_cast<__nv_bfloat16*>(out.data);
    // Prefill: column passes (choose_down_passes); the down rows buffer holds one pass's columns.
    const std::int32_t pass_columns = kHidden / passes;
    for (std::int32_t pass = 0; pass < passes; ++pass) {
        GemmParams columns  = down;
        columns.out_column0 = pass * pass_columns;
        columns.out_stride  = pass_columns;
        if (nvfp4) {
            launch_nvfp4_down(columns, slots, schedule, stream);
        } else {
            launch_mxfp8_down(columns, slots, schedule, stream);
        }
        if (taps != nullptr && taps->down != nullptr) {
            const std::size_t row_bytes = static_cast<std::size_t>(pass_columns) * sizeof(__nv_bfloat16);
            const cudaError_t status    = cudaMemcpy2DAsync(
                static_cast<std::uint8_t*>(taps->down) + static_cast<std::size_t>(pass) * row_bytes,
                kHidden * sizeof(__nv_bfloat16), b.down, row_bytes, row_bytes, s, cudaMemcpyDeviceToDevice, stream);
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("inkling_moe taps: ") + cudaGetErrorString(status));
            }
        }
    }
}

void inkling_route(const Tensor& x, const InklingMoeWeights& weights, Tensor& logits, Tensor& ids,
                   Tensor& route_weights, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("inkling_route: T must be positive"); }
    require_tensor(x, DType::BF16, kHidden, tokens, "x");
    require_tensor(logits, DType::FP32, kExperts, tokens, "logits");
    require_tensor(ids, DType::I32, kTopK, tokens, "ids");
    require_tensor(route_weights, DType::FP32, kTopK, tokens, "route_weights");
    if (weights.router == nullptr || weights.router_bias == nullptr) {
        throw std::invalid_argument("inkling_route: the router weight and bias are required");
    }
    launch_router_logits(as_bf16(x.data), as_bf16(weights.router), static_cast<float*>(logits.data), tokens,
                         nullptr, 0, stream);
    RouteOutputs route;
    route.ids    = static_cast<std::int32_t*>(ids.data);
    route.gammas = static_cast<float*>(route_weights.data);
    launch_router_select(static_cast<const float*>(logits.data), weights.router_bias, LiveTokens{nullptr, tokens},
                         route, stream);
}

void inkling_quant_nvfp4_4o6(const Tensor& x, Tensor& codes, Tensor& scales, Tensor& amax, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t rows = x.ne[1];
    if (k <= 0 || rows <= 0 || k % 16 != 0) {
        throw std::invalid_argument("inkling_quant_nvfp4_4o6: x must be [K, R] with K a multiple of 16");
    }
    require_tensor(x, DType::BF16, k, rows, "x");
    require_tensor(codes, DType::U8, k / 2, rows, "codes");
    require_tensor(scales, DType::U8, k / 16, rows, "scales");
    if (amax.data == nullptr || amax.dtype != DType::FP32 || amax.numel() != rows) {
        throw std::invalid_argument("inkling_quant_nvfp4_4o6: amax must be FP32 [R]");
    }
    launch_quant_nvfp4(as_bf16(x.data), rows, k, nullptr, LiveTokens{nullptr, rows},
                       static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data),
                       static_cast<float*>(amax.data), stream);
}

void inkling_quant_mxfp8(const Tensor& x, Tensor& codes, Tensor& exponents, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t rows = x.ne[1];
    if (k <= 0 || rows <= 0 || k % 32 != 0) {
        throw std::invalid_argument("inkling_quant_mxfp8: x must be [K, R] with K a multiple of 32");
    }
    require_tensor(x, DType::BF16, k, rows, "x");
    require_tensor(codes, DType::U8, k, rows, "codes");
    require_tensor(exponents, DType::U8, k / 32, rows, "exponents");
    launch_quant_mxfp8(as_bf16(x.data), rows, k, nullptr, LiveTokens{nullptr, rows},
                       static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(exponents.data), nullptr,
                       stream);
}

void inkling_softcap_swiglu(const Tensor& gate_up, const Tensor& gamma, Tensor& y, cudaStream_t stream) {
    const std::int32_t rows = gate_up.ne[1];
    if (rows <= 0) { throw std::invalid_argument("inkling_softcap_swiglu: R must be positive"); }
    require_tensor(gate_up, DType::BF16, kGateUpRows, rows, "gate_up");
    require_tensor(y, DType::BF16, kIntermediate, rows, "y");
    if (gamma.data == nullptr || gamma.dtype != DType::FP32 || gamma.numel() != rows) {
        throw std::invalid_argument("inkling_softcap_swiglu: gamma must be FP32 [R]");
    }
    launch_softcap_swiglu(as_bf16(gate_up.data), static_cast<const float*>(gamma.data), rows,
                          static_cast<__nv_bfloat16*>(y.data), stream);
}

void inkling_bench_predecessor(std::int32_t nanoseconds, cudaStream_t stream) {
    launch_bench_predecessor(nanoseconds, stream);
}

void inkling_decode_expert_units(const InklingExpertBank& bank, InklingExpertFormat format, std::int32_t k,
                                 std::int32_t first_expert, std::int32_t count, Tensor& unit, Tensor& factor,
                                 cudaStream_t stream) {
    static_assert(kGateUpRows == kHidden, "both banks stack 2304 rows per expert");
    const bool nvfp4 = format == InklingExpertFormat::Nvfp4;
    if (k != kHidden && k != kIntermediate) {
        throw std::invalid_argument("inkling_decode_expert_units: k must be 2304 (w13) or 1152 (w2)");
    }
    if (first_expert < 0 || count <= 0 || first_expert + count > kExperts) {
        throw std::invalid_argument("inkling_decode_expert_units: experts outside [0, 128)");
    }
    if (bank.codes == nullptr || bank.scales == nullptr ||
        (nvfp4 && (bank.amax == nullptr || bank.outer_den == nullptr))) {
        throw std::invalid_argument("inkling_decode_expert_units: the bank needs codes, scales (and amax / den)");
    }
    require_tensor(unit, DType::BF16, k, count * kGateUpRows, "unit");
    const bool with_factor = factor.data != nullptr;
    if (with_factor && (factor.dtype != DType::FP32 || factor.numel() != count)) {
        throw std::invalid_argument("inkling_decode_expert_units: factor must be FP32 [count]");
    }
    launch_decode_expert_units(bank, nvfp4, k, first_expert, count, static_cast<__nv_bfloat16*>(unit.data),
                               with_factor ? static_cast<float*>(factor.data) : nullptr, stream);
}

} // namespace sinfer::ops
