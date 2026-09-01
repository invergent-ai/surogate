#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"

#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kTargetRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, W8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

// surogate vendor patch (PATCHES.md #16/#28): qwen3.5-2b qkgv. The 2048-hidden
// exact-T split-K table is instantiated now; the 2..48 band rides it (batch
// decode rounds live at T=2..16 and previously paid the runtime-shaped MMA
// tiles).
constexpr std::array<RouteSpec, 4> kTarget2BRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 48, W8AttnInputScheduleId::SplitKMmaDirect},
    {49, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

// surogate vendor patch (PATCHES.md #28): qwen3.5-4b qkgv (rows 10240,
// hidden 2560) — same structure with its own exact-T table.
constexpr std::array<RouteSpec, 4> kTarget4BRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 48, W8AttnInputScheduleId::SplitKMmaDirect},
    {49, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

// Qwen3-0.6B's ungated fused qkv: rows 4096 = q2048 | k1024 | v1024 at hidden
// 1024. Same four-band structure as the gated target tables; the bands were not
// measured for this shape, they are the family's defaults for a small parent.
constexpr std::array<RouteSpec, 4> kQwen3Routes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, W8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 9> kCompanionRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 96, W8AttnInputScheduleId::SplitKMmaDirect},
    {97, 192, W8AttnInputScheduleId::MmaR32C64},
    {193, 288, W8AttnInputScheduleId::MmaR64C96},
    {289, 320, W8AttnInputScheduleId::MmaR64C64},
    {321, 384, W8AttnInputScheduleId::MmaR64C128},
    {385, 448, W8AttnInputScheduleId::MmaR128C64},
    {449, 560, W8AttnInputScheduleId::MmaR128C80},
    {561, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kTargetRoutes) && catalog_is_closed(kTarget2BRoutes) &&
                  catalog_is_closed(kTarget4BRoutes) && catalog_is_closed(kQwen3Routes),
              "W8 target attention input routes must be exact and closed");
static_assert(catalog_is_closed(kCompanionRoutes),
              "W8 companion attention input routes must be exact and closed");

bool is_companion_shape(const W8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

// The other three-output (ungated) shape: Qwen3-0.6B, 16 query heads and 8 KV
// heads at head dim 128 over hidden 1024. It shares the companion's public entry
// point and nothing else -- different parent, different hidden, different row
// split -- so it is keyed separately rather than folded into that predicate.
bool is_qwen3_ungated_shape(const W8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 1024 && problem.query_rows == 2048 && problem.kv_rows == 1024 &&
           problem.parent_rows == 4096 && problem.padded_k == 1024;
}

bool supported_shape(const W8AttnInputProblem& problem) noexcept {
    const bool target_qkgv =
        problem.query_rows == 4096 && problem.kv_rows == 512 && problem.parent_rows == 9216;
    // surogate vendor patches (PATCHES.md #13/#16): qwen3.5-0.8b/-2b fused
    // qkgv (same 5120-row structure; hidden 1024 or 2048).
    const bool small_qkgv = problem.query_rows == 2048 && problem.kv_rows == 512 &&
                            problem.parent_rows == 5120 &&
                            ((problem.input_rows == 1024 && problem.padded_k == 1024) ||
                             (problem.input_rows == 2048 && problem.padded_k == 2048));
    if (small_qkgv) { return true; }
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b fused qkgv.
    const bool q4b_qkgv = problem.query_rows == 4096 && problem.kv_rows == 1024 &&
                          problem.parent_rows == 10240 && problem.input_rows == 2560 &&
                          problem.padded_k == 2560;
    if (q4b_qkgv) { return true; }
    if (is_qwen3_ungated_shape(problem)) { return true; }
    return problem.input_rows == 2048 && problem.padded_k == 2048 &&
           (target_qkgv || is_companion_shape(problem));
}

} // namespace

const char* w8_attn_input_schedule_name(W8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        return "attn_input_proj.w8.decode.r8.direct.k2048";
    case W8AttnInputScheduleId::SplitKMmaDirect:
        return "attn_input_proj.w8.splitk.mma.r16.direct";
    case W8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.w8.simt.r8.c4";
    case W8AttnInputScheduleId::MmaR32C64:
        return "attn_input_proj.w8.mma.r32.c64";
    case W8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.w8.mma.r32.c128";
    case W8AttnInputScheduleId::MmaR64C64:
        return "attn_input_proj.w8.mma.r64.c64";
    case W8AttnInputScheduleId::MmaR64C96:
        return "attn_input_proj.w8.mma.r64.c96";
    case W8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.w8.mma.r64.c128";
    case W8AttnInputScheduleId::MmaR128C64:
        return "attn_input_proj.w8.mma.r128.c64";
    case W8AttnInputScheduleId::MmaR128C80:
        return "attn_input_proj.w8.mma.r128.c80";
    }
    return "attn_input_proj.w8.unknown";
}

bool w8_attn_input_admits(const W8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8AttnInputPlan w8_attn_input_resolve_plan(const W8AttnInputProblem& problem) {
    if (!w8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "W8 attention input: no registered geometry for parent rows " +
            std::to_string(problem.parent_rows) + " (query " +
            std::to_string(problem.query_rows) + " | kv " + std::to_string(problem.kv_rows) +
            ") over k " + std::to_string(problem.padded_k) + " at T " +
            std::to_string(problem.cols) +
            "; register the shape in w8_attn_input_plan.cpp and instantiate its launchers");
    }
    const auto resolve_from = [&](const auto& routes) -> W8AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("W8 attention input: admitted problem has no covering route");
    };
    if (is_companion_shape(problem)) { return resolve_from(kCompanionRoutes); }
    if (is_qwen3_ungated_shape(problem)) { return resolve_from(kQwen3Routes); }
    if (problem.parent_rows == 5120 && problem.input_rows == 2048) {
        return resolve_from(kTarget2BRoutes);
    }
    if (problem.parent_rows == 10240) { return resolve_from(kTarget4BRoutes); }
    return resolve_from(kTargetRoutes);
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (!(problem.parent_rows == 9216 || problem.parent_rows == 5120 ||
          problem.parent_rows == 10240) ||
        (problem.kv_rows != 512 && problem.kv_rows != 1024) ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "W8 attention input: plan does not match exact four-output problem");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaDirect:
        w8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SimtR8C4:
        w8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C128:
        w8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C128:
        w8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C64:
    case W8AttnInputScheduleId::MmaR64C64:
    case W8AttnInputScheduleId::MmaR64C96:
    case W8AttnInputScheduleId::MmaR128C64:
    case W8AttnInputScheduleId::MmaR128C80:
        throw std::logic_error("W8 attention input: companion schedule in four-output plan");
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    w8_attn_input_execute_plan(w8_attn_input_resolve_plan(problem), x, weight, q, gate, k, v,
                               stream);
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (!(is_companion_shape(problem) || is_qwen3_ungated_shape(problem)) ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "W8 attention input: plan does not match exact three-output problem (parent rows " +
            std::to_string(problem.parent_rows) + ", k " + std::to_string(problem.padded_k) +
            ", T " + std::to_string(problem.cols) + ")");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaDirect:
        w8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::SimtR8C4:
        w8_attn_input_simt_r8_c4_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C128:
        w8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C64:
        w8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C64:
        w8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C96:
        w8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR128C64:
        w8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR128C80:
        w8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C128:
        w8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    w8_attn_input_execute_plan(w8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace sinfer::ops::detail
