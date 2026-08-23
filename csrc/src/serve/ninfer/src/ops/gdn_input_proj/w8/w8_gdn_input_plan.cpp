#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"

#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8GdnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, W8GdnInputScheduleId::DecodeR8Direct},
    {2, 96, W8GdnInputScheduleId::SplitKMmaDirect},
    {97, kAnyCols, W8GdnInputScheduleId::MmaR64C128},
}};

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "W8 GDN input routes must be exact and closed");

bool supported_shape(const W8GdnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.qkv_rows == 8192 && problem.z_rows == 4096 &&
           problem.parent_rows == 12288 && problem.padded_k == 2048;
}

} // namespace

const char* w8_gdn_input_schedule_name(W8GdnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case W8GdnInputScheduleId::DecodeR8Direct:
        return "gdn_input_proj.w8.decode.r8.direct.k2048.split2";
    case W8GdnInputScheduleId::SplitKMmaDirect:
        return "gdn_input_proj.w8.mma.splitk.direct.k2048";
    case W8GdnInputScheduleId::MmaR64C128:
        return "gdn_input_proj.w8.mma.r64.c128.split2";
    }
    return "gdn_input_proj.w8.unknown";
}

const char* w8_gdn_input_conv_schedule_name(W8GdnInputConvScheduleId schedule) noexcept {
    switch (schedule) {
    case W8GdnInputConvScheduleId::DecodeFused:
        return "gdn_input_proj_conv.w8.decode.fused";
    case W8GdnInputConvScheduleId::SplitKMmaFused:
        return "gdn_input_proj_conv.w8.mma.splitk.fused";
    case W8GdnInputConvScheduleId::Materialized:
        return "gdn_input_proj_conv.w8.materialized";
    }
    return "gdn_input_proj_conv_snapshot.w8.unknown";
}

bool w8_gdn_input_admits(const W8GdnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8GdnInputPlan w8_gdn_input_resolve_plan(const W8GdnInputProblem& problem) {
    if (!w8_gdn_input_admits(problem)) {
        throw std::invalid_argument("W8 GDN input: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("W8 GDN input: admitted problem has no covering route");
}

W8GdnInputConvPlan w8_gdn_input_conv_resolve_plan(const W8GdnInputProblem& problem,
                                                  std::int32_t batch_size) {
    if (!w8_gdn_input_admits(problem) || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "W8 GDN input conv: exact problem or column count is not admitted");
    }
    if (batch_size > 1) { return {W8GdnInputConvScheduleId::Materialized}; }
    if (problem.cols == 1) { return {W8GdnInputConvScheduleId::DecodeFused}; }
    if (problem.cols <= 16) { return {W8GdnInputConvScheduleId::SplitKMmaFused}; }
    return {W8GdnInputConvScheduleId::Materialized};
}

void w8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    const W8GdnInputProblem problem{x.ne[0], qkv.ne[0], z.ne[0], weight.n, weight.padded_shape[1],
                                    x.ne[1]};
    const W8GdnInputPlan plan = w8_gdn_input_resolve_plan(problem);
    switch (plan.schedule) {
    case W8GdnInputScheduleId::DecodeR8Direct:
        w8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case W8GdnInputScheduleId::SplitKMmaDirect:
        w8_gdn_input_splitk_mma_launch(x, weight, qkv, z, stream);
        return;
    case W8GdnInputScheduleId::MmaR64C128:
        w8_gdn_input_mma_r64_c128_launch(x, weight, qkv, z, stream);
        return;
    }
    throw std::logic_error("W8 GDN input: unknown schedule");
}

} // namespace ninfer::ops::detail
