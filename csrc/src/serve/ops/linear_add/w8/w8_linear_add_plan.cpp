#include "ops/linear_add/w8/w8_linear_add_plan.h"

#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_launch.h"

#include <string>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8LinearAddScheduleId schedule;
};

constexpr std::array<RouteSpec, 5> kK4096Routes{{
    {1, 1, W8LinearAddScheduleId::SimtR8C4},
    {2, 48, W8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, W8LinearAddScheduleId::MediumSplitK},
    {129, 640, W8LinearAddScheduleId::MmaR32C128},
    {641, kAnyCols, W8LinearAddScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 33> kK6144Routes{{
    {1, 1, W8LinearAddScheduleId::DecodeR16},
    {2, 48, W8LinearAddScheduleId::SplitKMmaExactT},
    {49, 128, W8LinearAddScheduleId::MediumSplitK},
    {129, 191, W8LinearAddScheduleId::MmaR32C128},
    {192, 192, W8LinearAddScheduleId::MmaR32C96},
    {193, 256, W8LinearAddScheduleId::MmaR32C128},
    {257, 384, W8LinearAddScheduleId::MmaR32C64},
    {385, 399, W8LinearAddScheduleId::MmaR32C96},
    {400, 400, W8LinearAddScheduleId::MmaR32C80},
    {401, 447, W8LinearAddScheduleId::MmaR32C96},
    {448, 448, W8LinearAddScheduleId::MmaR32C64},
    {449, 480, W8LinearAddScheduleId::MmaR32C96},
    {481, 640, W8LinearAddScheduleId::MmaR32C128},
    {641, 672, W8LinearAddScheduleId::MmaR48C96},
    {673, 704, W8LinearAddScheduleId::MmaR48C64},
    {705, 784, W8LinearAddScheduleId::MmaR48C112},
    {785, 896, W8LinearAddScheduleId::MmaR48C128},
    {897, 960, W8LinearAddScheduleId::MmaR64C96},
    {961, 1023, W8LinearAddScheduleId::MmaR64C112},
    {1024, 1024, W8LinearAddScheduleId::MmaR64C128},
    {1025, 1120, W8LinearAddScheduleId::MmaR64C112},
    {1121, 1280, W8LinearAddScheduleId::MmaR64C128},
    {1281, 1344, W8LinearAddScheduleId::MmaR128C64},
    {1345, 1408, W8LinearAddScheduleId::MmaR48C128},
    {1409, 1680, W8LinearAddScheduleId::MmaR128C80},
    {1681, 1791, W8LinearAddScheduleId::MmaR48C128},
    {1792, 1792, W8LinearAddScheduleId::MmaR64C128},
    {1793, 1919, W8LinearAddScheduleId::MmaR48C128},
    {1920, 1920, W8LinearAddScheduleId::MmaR64C128},
    {1921, 2016, W8LinearAddScheduleId::MmaR64C96},
    {2017, 2047, W8LinearAddScheduleId::MmaR64C112},
    {2048, 2048, W8LinearAddScheduleId::MmaR64C128},
    {2049, kAnyCols, W8LinearAddScheduleId::MmaR64C128},
}};

// surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b residual projections
// ({1024,2048} and {1024,3584}). The SplitKMmaExactT / MediumSplitK / DecodeR16
// launchers are compile-time (2048 x {4096,6144}) instantiations, so the 0.8B
// shapes route exclusively over the runtime-shaped SIMT and MMA schedules.
// Measured on an idle RTX 5090 (bench/ops/q08_route_sweep_bench, T in
// {472, 888, 1024, 1912}): with only 1024 output rows the r64/r128 row
// tiles underfill the GPU — r32c128 wins through ~1024 tokens (52->84
// TF/s vs 24->54 on r64c128) and r48c128 takes over at large T
// (T=1912: 102.7us vs 128.0us on the k=2048 shape).
constexpr std::array<RouteSpec, 5> kQ08Routes{{
    // surogate vendor patch (PATCHES.md #28/#29): T=1 SIMT; the 2..16
    // batch-decode band rides the exact-T bakes now.
    {1, 1, W8LinearAddScheduleId::SimtR8C4},
    {2, 32, W8LinearAddScheduleId::SplitKMmaExactT},
    {33, 128, W8LinearAddScheduleId::MmaR32C128},
    {129, 1024, W8LinearAddScheduleId::MmaR32C128},
    {1025, kAnyCols, W8LinearAddScheduleId::MmaR48C128},
}};

// qwen3.5-2b {2048,2048}: no exact-T instantiations at k=2048 (those are
// 2048x{4096,6144} bakes), so measured runtime-shaped schedules throughout.
constexpr std::array<RouteSpec, 6> kQ2BRoutes{{
    // surogate vendor patch (PATCHES.md #28/#29): batch-decode band on the
    // exact-T bakes.
    {1, 1, W8LinearAddScheduleId::SimtR8C4},
    {2, 32, W8LinearAddScheduleId::SplitKMmaExactT},
    {33, 512, W8LinearAddScheduleId::MmaR32C96},
    {513, 900, W8LinearAddScheduleId::MmaR48C128},
    {901, 1024, W8LinearAddScheduleId::MmaR32C128},
    {1025, kAnyCols, W8LinearAddScheduleId::MmaR48C128},
}};

// surogate vendor patch (PATCHES.md #29): qwen3.5-4b output/down routes.
constexpr std::array<RouteSpec, 5> kQ4B29Routes{{
    {1, 1, W8LinearAddScheduleId::SimtR8C4},
    {2, 32, W8LinearAddScheduleId::SplitKMmaExactT},
    {33, 512, W8LinearAddScheduleId::MmaR32C96},
    {513, 1024, W8LinearAddScheduleId::MmaR48C128},
    {1025, kAnyCols, W8LinearAddScheduleId::MmaR48C128},
}};

// {2048, 5632}: exact-T bakes over the C=32 batch band; the runtime-tiled
// r32c128 / r48c128 split follows the extent-agnostic table. Measured on an
// idle RTX 5090 (bench/ops/w8_linear_add_bench --k 5632, cold cache, medians):
// see PATCHES.md #90 for the T table the band is cut from.
constexpr std::array<RouteSpec, 4> kR2048K5632Routes{{
    {1, 1, W8LinearAddScheduleId::SimtR8C4},
    {2, 32, W8LinearAddScheduleId::SplitKMmaExactT},
    {33, 1024, W8LinearAddScheduleId::MmaR32C128},
    {1025, kAnyCols, W8LinearAddScheduleId::MmaR48C128},
}};

// Every admitted shape with no bake of its own. The exact-T split-K tables and
// the medium split-K family bake both extents (rows and k), so a shape that is
// not one of theirs takes only the kernels that read both from the weight: the
// SIMT decode and the runtime-tiled MMA. Today that is qwen3-0.6b's mlp down
// {1024, 3072} and gemma-3-270m's {640, 1024|2048}.
// The MMA tiles need k % kW8MmaScaleRowAlignmentK == 0 for their 16-byte
// scale-row staging (w8_launch.h); `mma_alignment_holds` below refuses a shape
// that would reach them otherwise, at compile time for a registered shape and
// by name from `w8_linear_add_resolve_plan` for anything else.
constexpr std::array<RouteSpec, 3> kExtentAgnosticRoutes{{
    {1, 1, W8LinearAddScheduleId::SimtR8C4},
    {2, 1024, W8LinearAddScheduleId::MmaR32C128},
    {1025, kAnyCols, W8LinearAddScheduleId::MmaR48C128},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK4096Routes) && routes_are_closed(kK6144Routes) &&
                  routes_are_closed(kQ08Routes) && routes_are_closed(kExtentAgnosticRoutes) &&
                  routes_are_closed(kQ2BRoutes) && routes_are_closed(kQ4B29Routes) &&
                  routes_are_closed(kR2048K5632Routes),
              "W8 LinearAdd routes must be exact, contiguous, and closed");

enum class RouteTable {
    K4096,
    K6144,
    Q08,
    Q2B,
    Q4B29,
    R2048K5632,
    ExtentAgnostic,
};

constexpr std::span<const RouteSpec> routes_of(RouteTable table) {
    switch (table) {
    case RouteTable::K4096:
        return kK4096Routes;
    case RouteTable::K6144:
        return kK6144Routes;
    case RouteTable::Q08:
        return kQ08Routes;
    case RouteTable::Q2B:
        return kQ2BRoutes;
    case RouteTable::Q4B29:
        return kQ4B29Routes;
    case RouteTable::R2048K5632:
        return kR2048K5632Routes;
    case RouteTable::ExtentAgnostic:
        return kExtentAgnosticRoutes;
    }
    return kExtentAgnosticRoutes;
}

// Which table a (rows, k) geometry takes. A shape is keyed on both extents:
// the exact-T and medium split-K bakes instantiate rows and k, and every
// launcher refuses a geometry it did not bake, so the geometry must never be
// read from k alone (tinyllama's {2048, 5632} once fell through to the 4096
// table that way).
constexpr RouteTable table_for(std::int32_t rows, std::int32_t k) {
    // gemma-3-270m's two shapes take the runtime-tiled routes; its 640 rows have
    // no exact-T bake of their own, and the k=2048 branch below is the 2b's.
    if (rows == 640) { return RouteTable::ExtentAgnostic; }
    if (rows == 1024) {
        // The row count alone does not name the geometry here: the 0.8b's exact-T
        // bakes are k=2048/3584, and the 0.6b's mlp down is k=3072.
        return k == 3072 ? RouteTable::ExtentAgnostic : RouteTable::Q08;
    }
    // qwen3.5-2b output projections (measured on an idle RTX 5090,
    // bench/ops/q08_route_sweep_bench: 472 -> r32c96 63.5us, 888 -> r48c128
    // 102.8us (+26% over r32c128), 1024 -> r32c128, 1912 -> r48c128).
    if (k == 2048) { return RouteTable::Q2B; }
    // surogate vendor patch (PATCHES.md #29): qwen3.5-4b (2560 rows) has its
    // exact-T bakes now (T=2..16, k=4096/9216).
    if (rows == 2560) { return RouteTable::Q4B29; }
    // tinyllama's mlp down: its own exact-T bake since PATCHES.md #90; before
    // that it was the shape that fell through here to the 4096 table.
    if (rows == 2048 && k == 5632) { return RouteTable::R2048K5632; }
    // The exact-T and medium split-K bakes that remain are 2048 rows over k 4096
    // or 6144. Anything else unbaked takes the table whose kernels read both
    // extents from the weight.
    if (rows != 2048 || (k != 4096 && k != 6144)) { return RouteTable::ExtentAgnostic; }
    return k == 6144 ? RouteTable::K6144 : RouteTable::K4096;
}

constexpr bool schedule_uses_mma(W8LinearAddScheduleId schedule) {
    return schedule != W8LinearAddScheduleId::DecodeR16 &&
           schedule != W8LinearAddScheduleId::SimtR8C4;
}

constexpr bool table_uses_mma(std::span<const RouteSpec> routes) {
    for (const RouteSpec& route : routes) {
        if (schedule_uses_mma(route.schedule)) { return true; }
    }
    return false;
}

// The MMA tiles stage each scale row (k / 16 bytes) with a 16-byte cp.async,
// so a table with any MMA band needs k % kW8MmaScaleRowAlignmentK == 0. A
// SIMT-only table would not, which is why the rule is keyed on the table the
// shape takes and not on k alone.
constexpr bool mma_alignment_holds(std::int32_t rows, std::int32_t k) {
    return k % kW8MmaScaleRowAlignmentK == 0 || !table_uses_mma(routes_of(table_for(rows, k)));
}

// The one list of (rows, k) geometries this op serves. Registering a shape
// here is what admits it in the plan, the wrapper, and the conformance test.
constexpr auto kRegisteredShapes = std::to_array<W8LinearAddShape>({
    // 35B residual projections: the exact-T, medium split-K and per-band MMA
    // tables above are measured on these two.
    {2048, 4096},
    {2048, 6144},
    // surogate vendor patches (PATCHES.md #13/#16): qwen3.5-0.8b
    // ({1024,2048|3584}) and qwen3.5-2b attn/gdn output ({2048,2048}; its
    // mlp down {2048,6144} rides the registered base shape).
    {1024, 2048},
    {1024, 3584},
    {2048, 2048},
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b output/down.
    {2560, 4096},
    {2560, 9216},
    // qwen3-0.6b: its attention output {1024, 2048} already rides the 0.8b
    // shape above; only the mlp down {1024, 3072} is new, and it has no bake, so
    // it resolves through the extent-agnostic table.
    {1024, 3072},
    // tinyllama-1.1b: its attention output {2048, 2048} is already the 2b shape
    // above; only the mlp down {2048, 5632} is new.
    {2048, 5632},
    // gemma-3-270m: attention output {640, 1024} and mlp down {640, 2048}. Its
    // hidden is narrower than anything else registered here, so both are new.
    {640, 1024},
    {640, 2048},
});

constexpr bool registered_shapes_are_sound() {
    for (const W8LinearAddShape& shape : kRegisteredShapes) {
        // Positive extents, whole 32-lane scale groups, and the scale-row
        // alignment the shape's own table needs.
        if (shape.rows <= 0 || shape.k <= 0 || shape.k % 32 != 0) { return false; }
        if (!mma_alignment_holds(shape.rows, shape.k)) { return false; }
    }
    return true;
}

static_assert(registered_shapes_are_sound(),
              "every registered W8 linear_add shape must have k % 32 == 0, and k % 256 == 0 "
              "whenever its route table has an MMA band (16-byte scale-row staging)");

bool registered(std::int32_t rows, std::int32_t k) {
    for (const W8LinearAddShape& shape : kRegisteredShapes) {
        if (shape.rows == rows && shape.k == k) { return true; }
    }
    return false;
}

std::int32_t schedule_rows(W8LinearAddScheduleId schedule) {
    switch (schedule) {
    case W8LinearAddScheduleId::DecodeR16:
    case W8LinearAddScheduleId::MediumSplitK:
        break;
    case W8LinearAddScheduleId::SimtR8C4:
        return 8;
    case W8LinearAddScheduleId::MmaR32C64:
    case W8LinearAddScheduleId::MmaR32C80:
    case W8LinearAddScheduleId::MmaR32C96:
    case W8LinearAddScheduleId::MmaR32C128:
        return 32;
    case W8LinearAddScheduleId::MmaR48C64:
    case W8LinearAddScheduleId::MmaR48C96:
    case W8LinearAddScheduleId::MmaR48C112:
    case W8LinearAddScheduleId::MmaR48C128:
        return 48;
    case W8LinearAddScheduleId::MmaR64C96:
    case W8LinearAddScheduleId::MmaR64C112:
    case W8LinearAddScheduleId::MmaR64C128:
        return 64;
    case W8LinearAddScheduleId::MmaR128C64:
    case W8LinearAddScheduleId::MmaR128C80:
        return 128;
    case W8LinearAddScheduleId::SplitKMmaExactT:
        break;
    }
    throw std::logic_error("w8 linear_add: exact-T schedule has no row tile");
}

std::int32_t schedule_cols(W8LinearAddScheduleId schedule);

bool use_full(W8LinearAddScheduleId schedule, const W8LinearAddProblem& problem) {
    return problem.rows % schedule_rows(schedule) == 0 &&
           problem.cols % schedule_cols(schedule) == 0;
}

std::int32_t schedule_cols(W8LinearAddScheduleId schedule) {
    switch (schedule) {
    case W8LinearAddScheduleId::DecodeR16:
    case W8LinearAddScheduleId::MediumSplitK:
        break;
    case W8LinearAddScheduleId::SimtR8C4:
        return 4;
    case W8LinearAddScheduleId::MmaR32C64:
    case W8LinearAddScheduleId::MmaR48C64:
    case W8LinearAddScheduleId::MmaR128C64:
        return 64;
    case W8LinearAddScheduleId::MmaR32C80:
    case W8LinearAddScheduleId::MmaR128C80:
        return 80;
    case W8LinearAddScheduleId::MmaR32C96:
    case W8LinearAddScheduleId::MmaR48C96:
    case W8LinearAddScheduleId::MmaR64C96:
        return 96;
    case W8LinearAddScheduleId::MmaR48C112:
    case W8LinearAddScheduleId::MmaR64C112:
        return 112;
    case W8LinearAddScheduleId::MmaR32C128:
    case W8LinearAddScheduleId::MmaR48C128:
    case W8LinearAddScheduleId::MmaR64C128:
        return 128;
    case W8LinearAddScheduleId::SplitKMmaExactT:
        break;
    }
    throw std::logic_error("w8 linear_add: exact-T schedule is not token-sliced");
}

} // namespace

const char* w8_linear_add_schedule_name(W8LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case W8LinearAddScheduleId::DecodeR16:
        return "linear_add.w8.decode.r16.residual";
    case W8LinearAddScheduleId::SplitKMmaExactT:
        return "linear_add.w8.splitk8.mma.r16.exact_t.residual";
    case W8LinearAddScheduleId::MediumSplitK:
        return "linear_add.w8.medium_splitk.residual";
    case W8LinearAddScheduleId::SimtR8C4:
        return "linear_add.w8.simt.r8.c4.slab1024.s2.code_ca.scale_pair32";
    case W8LinearAddScheduleId::MmaR32C64:
        return "linear_add.w8.mma.r32.c64.residual";
    case W8LinearAddScheduleId::MmaR32C80:
        return "linear_add.w8.mma.r32.c80.residual";
    case W8LinearAddScheduleId::MmaR32C96:
        return "linear_add.w8.mma.r32.c96.residual";
    case W8LinearAddScheduleId::MmaR32C128:
        return "linear_add.w8.mma.r32.c128.k64.wr32.wc16.s2.scale_cache8.lb2";
    case W8LinearAddScheduleId::MmaR48C64:
        return "linear_add.w8.mma.r48.c64.residual";
    case W8LinearAddScheduleId::MmaR48C96:
        return "linear_add.w8.mma.r48.c96.residual";
    case W8LinearAddScheduleId::MmaR48C112:
        return "linear_add.w8.mma.r48.c112.residual";
    case W8LinearAddScheduleId::MmaR48C128:
        return "linear_add.w8.mma.r48.c128.residual";
    case W8LinearAddScheduleId::MmaR64C96:
        return "linear_add.w8.mma.r64.c96.residual";
    case W8LinearAddScheduleId::MmaR64C112:
        return "linear_add.w8.mma.r64.c112.residual";
    case W8LinearAddScheduleId::MmaR64C128:
        return "linear_add.w8.mma.r64.c128.k64.wr64.wc16.s2.scale_cache8.lb2";
    case W8LinearAddScheduleId::MmaR128C64:
        return "linear_add.w8.mma.r128.c64.residual";
    case W8LinearAddScheduleId::MmaR128C80:
        return "linear_add.w8.mma.r128.c80.residual";
    }
    return "linear_add.w8.unknown";
}

bool w8_linear_add_schedule_uses_mma(W8LinearAddScheduleId schedule) noexcept {
    return schedule_uses_mma(schedule);
}

std::span<const W8LinearAddShape> w8_linear_add_registered_shapes() noexcept {
    return kRegisteredShapes;
}

bool w8_linear_add_admits(const W8LinearAddProblem& problem) noexcept {
    return registered(problem.rows, problem.k) && problem.padded_k == problem.k &&
           problem.cols >= 1;
}

W8LinearAddPlan w8_linear_add_resolve_plan(const W8LinearAddProblem& problem) {
    // Named first: for a shape whose table has an MMA band, a misaligned k is
    // the constraint that matters, not the generic "not admitted".
    if (!mma_alignment_holds(problem.rows, problem.k)) {
        throw std::invalid_argument(
            "w8 linear_add: k " + std::to_string(problem.k) + " violates k % " +
            std::to_string(kW8MmaScaleRowAlignmentK) +
            " == 0 -- the MMA tiles stage each scale row (k/16 bytes) with a 16-byte "
            "cp.async, and the route table for this shape sends some T band to them (rows " +
            std::to_string(problem.rows) + ", k " + std::to_string(problem.k) + ", padded_k " +
            std::to_string(problem.padded_k) + ", cols " + std::to_string(problem.cols) + ")");
    }
    if (!w8_linear_add_admits(problem)) {
        throw std::invalid_argument("w8 linear_add: exact problem or column count is not admitted "
                                    "(rows " + std::to_string(problem.rows) + ", k " +
                                    std::to_string(problem.k) + ", padded_k " +
                                    std::to_string(problem.padded_k) + ", cols " +
                                    std::to_string(problem.cols) + ")");
    }
    for (const RouteSpec& route : routes_of(table_for(problem.rows, problem.k))) {
        if (problem.cols >= route.first && problem.cols <= route.last) {
            return {route.schedule};
        }
    }
    throw std::logic_error("w8 linear_add: admitted problem has no covering route");
}

void w8_linear_add_execute_plan(const W8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream) {
    const W8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const W8LinearAddPlan resolved = w8_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("w8 linear_add: plan does not match the exact problem");
    }
    if (plan.schedule == W8LinearAddScheduleId::DecodeR16) {
        w8_linear_add_decode_r16_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == W8LinearAddScheduleId::SplitKMmaExactT) {
        w8_linear_add_splitk_mma_launch(x, w, residual_out, stream);
        return;
    }
    if (plan.schedule == W8LinearAddScheduleId::MediumSplitK) {
        w8_linear_add_medium_splitk_launch(x, w, residual_out, stream);
        return;
    }
    const bool full = use_full(plan.schedule, problem);
    for_each_token_slice(
        x.ne[1], schedule_cols(plan.schedule), [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice  = x.slice(1, offset, count);
            Tensor residual_slice = residual_out.slice(1, offset, count);
            switch (plan.schedule) {
            case W8LinearAddScheduleId::DecodeR16:
            case W8LinearAddScheduleId::MediumSplitK:
                break;
            case W8LinearAddScheduleId::SimtR8C4:
                w8_linear_add_simt_r8_c4_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C64:
                w8_linear_add_mma_r32_c64_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C80:
                w8_linear_add_mma_r32_c80_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C96:
                w8_linear_add_mma_r32_c96_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR32C128:
                w8_linear_add_mma_r32_c128_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C64:
                w8_linear_add_mma_r48_c64_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C96:
                w8_linear_add_mma_r48_c96_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C112:
                w8_linear_add_mma_r48_c112_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR48C128:
                w8_linear_add_mma_r48_c128_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR64C96:
                w8_linear_add_mma_r64_c96_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR64C112:
                w8_linear_add_mma_r64_c112_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR64C128:
                w8_linear_add_mma_r64_c128_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR128C64:
                w8_linear_add_mma_r128_c64_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::MmaR128C80:
                w8_linear_add_mma_r128_c80_launch(full, x_slice, w, residual_slice, stream);
                return;
            case W8LinearAddScheduleId::SplitKMmaExactT:
                break;
            }
            throw std::logic_error("w8 linear_add: unknown tiled schedule");
        });
}

void w8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream) {
    const W8LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    w8_linear_add_execute_plan(w8_linear_add_resolve_plan(problem), x, w, residual_out, stream);
}

} // namespace sinfer::ops::detail
