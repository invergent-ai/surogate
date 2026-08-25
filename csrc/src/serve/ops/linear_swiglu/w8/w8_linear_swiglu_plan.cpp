#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
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

// surogate vendor patch (PATCHES.md #18): qwen3.5-4b — no exact-T
// instantiations at k=2560; decode plus the runtime-shaped MMA bands
// (the engine's A8 path takes T >= 224 regardless).
constexpr std::array<RouteSpec, 3> kQ4BRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 256, W8LinearSwiGluScheduleId::MmaR32C64},
    {257, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};

static_assert(
    [] {
        std::int64_t expected = 1;
        for (const RouteSpec& route : kQ4BRoutes) {
            if (route.first != expected || route.first > route.last) { return false; }
            expected = static_cast<std::int64_t>(route.last) + 1;
        }
        return expected == static_cast<std::int64_t>(kAnyCols) + 1;
    }(),
    "W8 LinearSwiGLU 4b routes must be exact and closed");

bool supported_shape(const W8LinearSwiGluProblem& problem) noexcept {
    // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b mlp {7168->3584, k=1024}.
    const bool base = problem.gate_up_rows == 12288 && problem.output_rows == 6144 &&
                      problem.k == 2048 && problem.padded_k == 2048;
    const bool q08 = problem.gate_up_rows == 7168 && problem.output_rows == 3584 &&
                     problem.k == 1024 && problem.padded_k == 1024;
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b mlp.
    const bool q4b = problem.gate_up_rows == 18432 && problem.output_rows == 9216 &&
                     problem.k == 2560 && problem.padded_k == 2560;
    return base || q08 || q4b;
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
            "W8 LinearSwiGLU: exact problem or column count is not admitted");
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
    const auto& routes = problem.k == 1024 ? kQ08Routes : kRoutes;
    for (const RouteSpec& route : routes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("W8 LinearSwiGLU: admitted problem has no route");
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

} // namespace ninfer::ops::detail
