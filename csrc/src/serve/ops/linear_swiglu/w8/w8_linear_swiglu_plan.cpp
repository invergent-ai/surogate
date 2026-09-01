#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include <array>
#include <string>
#include <limits>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8LinearSwiGluScheduleId schedule;
};

constexpr std::array<RouteSpec, 18> kRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 48, W8LinearSwiGluScheduleId::SplitKMmaExactT},
    {49, 64, W8LinearSwiGluScheduleId::MmaR32C64},
    {65, 80, W8LinearSwiGluScheduleId::MmaR32C80},
    {81, 96, W8LinearSwiGluScheduleId::MmaR32C96},
    {97, 128, W8LinearSwiGluScheduleId::MmaR64C64},
    {129, 192, W8LinearSwiGluScheduleId::MmaR32C64},
    {193, 240, W8LinearSwiGluScheduleId::MmaR128C80},
    {241, 255, W8LinearSwiGluScheduleId::MmaR32C128},
    {256, 256, W8LinearSwiGluScheduleId::MmaR64C128},
    {257, 264, W8LinearSwiGluScheduleId::MmaR64C64},
    {265, 288, W8LinearSwiGluScheduleId::MmaR64C96},
    {289, 320, W8LinearSwiGluScheduleId::MmaR64C64},
    {321, 384, W8LinearSwiGluScheduleId::MmaR64C128},
    {385, 448, W8LinearSwiGluScheduleId::MmaR128C64},
    {449, 512, W8LinearSwiGluScheduleId::MmaR64C128},
    {513, 560, W8LinearSwiGluScheduleId::MmaR128C80},
    {561, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "W8 LinearSwiGLU routes must be exact and closed");

// surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b (7168 x 1024) keeps
// the 35B catalog except the 449-512 region, where the measured winner on
// an idle RTX 5090 is MmaR128C80 (92.9us vs 107.1us on MmaR64C128 at
// T=472; bench/ops/q08_route_sweep_bench).
constexpr std::array<RouteSpec, 18> kQ08Routes = [] {
    std::array<RouteSpec, 18> routes = kRoutes;
    for (RouteSpec& route : routes) {
        if (route.first == 449 && route.last == 512) {
            route.schedule = W8LinearSwiGluScheduleId::MmaR128C80;
        }
    }
    return routes;
}();

// surogate vendor patch (PATCHES.md #18/#28): qwen3.5-4b. The k=2560 exact-T
// table is instantiated now; the 2..48 band rides it (batch decode rounds
// live at T=2..16; the engine's A8 path takes T >= 224 regardless).
constexpr std::array<RouteSpec, 4> kQ4BRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 48, W8LinearSwiGluScheduleId::SplitKMmaExactT},
    {49, 256, W8LinearSwiGluScheduleId::MmaR32C64},
    {257, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};

// qwen3-0.6b mlp {6144->3072, k=1024}. The exact-T split-K table is baked per
// (intermediate, hidden) and only 3584/1024 exists at this hidden size, so this
// shape takes the runtime-shaped MMA tiles for every T above decode: they read
// both extents from the weight and 3072 divides every registered BM/2. Decode
// has its own 3072 instantiation. A tuned exact-T table can follow later.
constexpr std::array<RouteSpec, 3> kQ3_06bRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 256, W8LinearSwiGluScheduleId::MmaR32C64},
    {257, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kQ4BRoutes),
              "W8 LinearSwiGLU 4b routes must be exact and closed");
static_assert(routes_are_closed(kQ3_06bRoutes),
              "W8 LinearSwiGLU 0.6b routes must be exact and closed");

// tinyllama-1.1b mlp {11264->5632, k=2048}. It shares k with the base shape,
// whose exact-T instantiations are baked 12288 wide, so -- exactly as for the
// 0.6b at k=1024 -- it takes only the kernels that read their extents from the
// weight: the SIMT decode and the runtime-shaped MMA tiles.
constexpr std::array<RouteSpec, 3> kTinyLlamaRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 1024, W8LinearSwiGluScheduleId::MmaR32C128},
    {1025, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};
static_assert(routes_are_closed(kTinyLlamaRoutes),
              "W8 LinearSwiGLU tinyllama routes must be exact and closed");

bool supported_shape(const W8LinearSwiGluProblem& problem) noexcept {
    // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b mlp {7168->3584, k=1024}.
    const bool base = problem.gate_up_rows == 12288 && problem.output_rows == 6144 &&
                      problem.k == 2048 && problem.padded_k == 2048;
    const bool q08 = problem.gate_up_rows == 7168 && problem.output_rows == 3584 &&
                     problem.k == 1024 && problem.padded_k == 1024;
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b mlp.
    const bool q4b = problem.gate_up_rows == 18432 && problem.output_rows == 9216 &&
                     problem.k == 2560 && problem.padded_k == 2560;
    // qwen3-0.6b mlp {6144->3072, k=1024}. Same k as the 0.8b above, so it
    // reuses that shape's exact-T instantiations; only the row counts differ.
    const bool q3_06b = problem.gate_up_rows == 6144 && problem.output_rows == 3072 &&
                        problem.k == 1024 && problem.padded_k == 1024;
    // tinyllama-1.1b mlp {11264->5632, k=2048}.
    const bool tinyllama = problem.gate_up_rows == 11264 && problem.output_rows == 5632 &&
                           problem.k == 2048 && problem.padded_k == 2048;
    return base || q08 || q4b || q3_06b || tinyllama;
}

} // namespace

const char* w8_linear_swiglu_schedule_name(W8LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
        return "linear_swiglu.w8.decode.pair.r16";
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        return "linear_swiglu.w8.splitk.mma.pair.exact_t";
    case W8LinearSwiGluScheduleId::MmaR32C64:
        return "linear_swiglu.w8.mma.pair.r16.c64";
    case W8LinearSwiGluScheduleId::MmaR32C80:
        return "linear_swiglu.w8.mma.pair.r16.c80";
    case W8LinearSwiGluScheduleId::MmaR32C96:
        return "linear_swiglu.w8.mma.pair.r16.c96";
    case W8LinearSwiGluScheduleId::MmaR32C128:
        return "linear_swiglu.w8.mma.pair.r16.c128";
    case W8LinearSwiGluScheduleId::MmaR64C64:
        return "linear_swiglu.w8.mma.pair.r32.c64";
    case W8LinearSwiGluScheduleId::MmaR64C96:
        return "linear_swiglu.w8.mma.pair.r32.c96";
    case W8LinearSwiGluScheduleId::MmaR64C128:
        return "linear_swiglu.w8.mma.pair.r32.c128";
    case W8LinearSwiGluScheduleId::MmaR128C64:
        return "linear_swiglu.w8.mma.pair.r64.c64";
    case W8LinearSwiGluScheduleId::MmaR128C80:
        return "linear_swiglu.w8.mma.pair.r64.c80";
    }
    return "linear_swiglu.w8.unknown";
}

bool w8_linear_swiglu_schedule_uses_mma(W8LinearSwiGluScheduleId schedule) noexcept {
    return schedule != W8LinearSwiGluScheduleId::DecodePairR16;
}

bool w8_linear_swiglu_admits(const W8LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8LinearSwiGluPlan w8_linear_swiglu_resolve_plan(const W8LinearSwiGluProblem& problem) {
    if (!w8_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "W8 LinearSwiGLU: exact problem or column count is not admitted (gate_up_rows " +
            std::to_string(problem.gate_up_rows) + ", output_rows " +
            std::to_string(problem.output_rows) + ", k " + std::to_string(problem.k) +
            ", padded_k " + std::to_string(problem.padded_k) + ", cols " +
            std::to_string(problem.cols) + ")");
    }
    // 4b (k=2560) has no exact-T instantiations: decode + the runtime-shaped
    // mma bands only (A8 takes T >= 224 in the engine anyway).
    if (problem.k == 2560) {
        for (const RouteSpec& route : kQ4BRoutes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("W8 LinearSwiGLU: 4b problem has no route");
    }
    const auto resolve_from = [&](const auto& routes) -> W8LinearSwiGluPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("W8 LinearSwiGLU: admitted problem has no route");
    };
    // Two models share k=1024 and differ in intermediate, so the table has to be
    // chosen on the pair: the 0.8b's exact-T bakes are 3584-wide and would run
    // the 0.6b's weight at the wrong row count.
    if (problem.k == 1024) {
        return problem.gate_up_rows == 6144 ? resolve_from(kQ3_06bRoutes)
                                            : resolve_from(kQ08Routes);
    }
    // k=2048 is shared the same way: the base table's exact-T bakes are 12288
    // wide and would run tinyllama's weight at the wrong row count.
    if (problem.k == 2048 && problem.gate_up_rows == 11264) {
        return resolve_from(kTinyLlamaRoutes);
    }
    return resolve_from(kRoutes);
}

void w8_linear_swiglu_execute_plan(const W8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, cudaStream_t stream) {
    const W8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const W8LinearSwiGluPlan resolved = w8_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("W8 LinearSwiGLU: plan does not match exact problem");
    }
    switch (plan.schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
        w8_linear_swiglu_decode_pair_r16_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        w8_linear_swiglu_splitk_exact_t_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C64:
        w8_linear_swiglu_mma_r32_c64_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C80:
        w8_linear_swiglu_mma_r32_c80_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C96:
        w8_linear_swiglu_mma_r32_c96_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C128:
        w8_linear_swiglu_mma_r32_c128_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C64:
        w8_linear_swiglu_mma_r64_c64_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C96:
        w8_linear_swiglu_mma_r64_c96_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C128:
        w8_linear_swiglu_mma_r64_c128_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR128C64:
        w8_linear_swiglu_mma_r128_c64_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR128C80:
        w8_linear_swiglu_mma_r128_c80_launch(x, w, out, stream);
        return;
    }
    throw std::logic_error("W8 LinearSwiGLU: unknown schedule");
}

void w8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const W8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    w8_linear_swiglu_execute_plan(w8_linear_swiglu_resolve_plan(problem), x, w, out, stream);
}

} // namespace sinfer::ops::detail
