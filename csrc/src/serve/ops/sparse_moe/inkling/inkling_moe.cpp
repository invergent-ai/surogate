// Inkling MoE op wrapper: workspace carving, routing, expert GEMMs, combine (api/ops/inkling_moe.h).
//
// Launch order of one call (each kernel waits for its predecessor, PDL):
//   router logits (also zeroes the expert counters) -> x quantisation -> selection + placement
//   [-> LoRA shrink (w13)] -> gate_up + SiTU -> y quantisation [-> LoRA shrink (w2)] -> down
//   -> combine.
// The fusions of DESIGN-moe.md 2.1 (route + quant in one grid, quantise y in down's prologue,
// combine in down) change no arithmetic and come with the M3 timing.

#include "api/ops/inkling_moe.h"
#include "api/ops/sparse_moe.h"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
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

template <class Allocate>
Buffers carve(InklingExpertFormat format, std::int32_t tokens, std::int32_t rank, Allocate&& allocate) {
    const std::size_t t     = static_cast<std::size_t>(tokens);
    const std::size_t slots = t * kTopK;
    const bool nvfp4        = format == InklingExpertFormat::Nvfp4;
    Buffers b;
    b.logits   = static_cast<float*>(allocate(t * kExperts * sizeof(float)));
    b.ids      = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
    b.gammas   = static_cast<float*>(allocate(slots * sizeof(float)));
    b.ranks    = static_cast<std::int32_t*>(allocate(slots * sizeof(std::int32_t)));
    b.counters = static_cast<std::int32_t*>(allocate(kCounterWords * sizeof(std::int32_t)));
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

} // namespace

std::size_t inkling_moe_workspace_bytes(InklingExpertFormat format, std::int32_t max_tokens, bool adapters) {
    if (max_tokens <= 0) { throw std::invalid_argument("inkling_moe_workspace_bytes: max_tokens must be positive"); }
    std::size_t total = kAlignment; // the arena's offset may be unaligned when the call starts
    carve(format, max_tokens, adapters ? InklingMoeGeometry::kMaxAdapterRank : 0, [&](std::size_t bytes) -> void* {
        total += aligned(bytes);
        return nullptr;
    });
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

    const Buffers b = carve(weights.format, tokens, rank, [&](std::size_t bytes) -> void* {
        return workspace.alloc_bytes(bytes, kAlignment).data;
    });
    const LiveTokens live{options.live_tokens, tokens};
    const auto* input = as_bf16(x.data);
    const std::int32_t slots = tokens * kTopK;

    // Router logits; the same launch zeroes the expert counters the selection counts into.
    launch_router_logits(input, as_bf16(weights.router), b.logits, tokens, b.counters, kCounterWords, stream);

    // x quantised once per token (the reference writes it once per routed row; the bits are equal).
    if (nvfp4) {
        launch_quant_nvfp4(input, tokens, kHidden, nullptr, live, b.x_codes, b.x_scales, b.x_amax, stream);
    } else {
        launch_quant_mxfp8(input, tokens, kHidden, nullptr, live, nullptr, nullptr, b.x_bf16, stream);
    }

    RouteOutputs route;
    route.ids     = b.ids;
    route.gammas  = b.gammas;
    route.ranks   = b.ranks;
    route.counts  = b.counters;
    route.ticket  = b.counters + kExperts;
    route.offsets = b.offsets;
    route.sorted  = b.sorted;
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
    launch_router_select(b.logits, weights.router_bias, live, route, stream);

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

    const GemmSchedule schedule = tokens <= kNarrowMaxTokens ? GemmSchedule::Narrow : GemmSchedule::Wide;
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
    if (nvfp4) {
        launch_nvfp4_gate_up(gate_up, slots, schedule, stream);
        launch_quant_nvfp4(b.y, slots, kIntermediate, b.ids, live, b.y_codes, b.y_scales, b.y_amax, stream);
    } else {
        launch_mxfp8_gate_up(gate_up, slots, schedule, stream);
        launch_quant_mxfp8(b.y, slots, kIntermediate, b.ids, live, nullptr, nullptr, b.y_bf16, stream);
    }

    if (adapted) {
        lora.low        = b.low_down;
        lora.low_stride = rank;
        launch_lora_shrink(b.y, kIntermediate, false, b.ids, live, lora, stream);
    }

    GemmParams down;
    down.a_codes     = b.y_codes;
    down.a_scales    = b.y_scales;
    down.a_amax      = b.y_amax;
    down.a_bf16      = b.y_bf16;
    down.a_row_shift = 0; // slot rows
    down.offsets     = b.offsets;
    down.sorted      = b.sorted;
    down.weights     = weights.w2;
    down.out         = b.down;
    down.lora        = lora;
    if (nvfp4) {
        launch_nvfp4_down(down, slots, schedule, stream);
    } else {
        launch_mxfp8_down(down, slots, schedule, stream);
    }

    launch_combine(b.down, b.ids, live, static_cast<__nv_bfloat16*>(out.data), stream);
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

} // namespace sinfer::ops
