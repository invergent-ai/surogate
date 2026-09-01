#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace sinfer::ops::detail {
namespace {

constexpr int kRows           = 2048;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 48;
using ProjectionLauncher      = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Hidden, int ActiveCols, int Rows = kRows>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr int KWarps =
        Hidden == 4096 ? (ActiveCols <= 12 ? 16 : 8) : (ActiveCols <= 32 ? 8 : 4);
    constexpr int MinBlocks = Hidden == 4096 ? (KWarps == 16 ? 1 : 2) : (ActiveCols <= 32 ? 2 : 3);
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    constexpr auto ActivationCache =
        Hidden == 4096 && (ActiveCols == 4 || (ActiveCols >= 27 && ActiveCols <= 40)) ? Cache::cg
                                                                                      : Cache::ca;
    using Geometry = W8LinearGeometry<Rows, Hidden>;
    using Schedule = W8SmallTMmaSchedule<KWarps, TileCols, MinBlocks, ScaleAccess, ActivationCache>;
    static_assert((Rows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const W8ContiguousOutput output{residual, Rows};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SmallTMmaResidualEpilogue>
        <<<Rows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, W8SmallTMmaResidualEpilogue{});
}

template <int Hidden, std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kK4096ProjectionLaunchers = make_projection_launchers<4096>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});
constexpr auto kK6144ProjectionLaunchers = make_projection_launchers<6144>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

// surogate vendor patch (PATCHES.md #29): qwen3.5-4b o_proj (2560x4096) and
// mlp down (2560x9216) exact-T tables, T=2..16 (the batch-decode band; the
// runtime-shaped SIMT read weights ceil(T/4) times there).
constexpr int kQ4bLastExactCols = 32; // PATCHES.md #29: C=32 batch band
template <int Hidden, std::size_t... Offsets>
constexpr auto make_q4b_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets), 2560>...};
}
constexpr auto kQ4bK4096Launchers = make_q4b_launchers<4096>(
    std::make_index_sequence<kQ4bLastExactCols - kFirstExactCols + 1>{});
constexpr auto kQ4bK9216Launchers = make_q4b_launchers<9216>(
    std::make_index_sequence<kQ4bLastExactCols - kFirstExactCols + 1>{});

// surogate vendor patch (PATCHES.md #29): qwen3.5-2b o_proj (2048x2048) and
// qwen3.5-0.8b o_proj (1024x2048) / down (1024x3584), same batch-decode band.
template <int Hidden, int Rows, std::size_t... Offsets>
constexpr auto make_small_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets), Rows>...};
}
constexpr auto kQ2bK2048Launchers = make_small_launchers<2048, 2048>(
    std::make_index_sequence<kQ4bLastExactCols - kFirstExactCols + 1>{});
constexpr auto kQ08K2048Launchers = make_small_launchers<2048, 1024>(
    std::make_index_sequence<kQ4bLastExactCols - kFirstExactCols + 1>{});
constexpr auto kQ08K3584Launchers = make_small_launchers<3584, 1024>(
    std::make_index_sequence<kQ4bLastExactCols - kFirstExactCols + 1>{});

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, Tensor& residual_out, const Weight& weight,
                   cudaStream_t stream) {
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), kRows};
    w8_rowsplit_medium_t_splitk_kernel<Hidden, TileCols, KSplits, NGroups, MinBlocks,
                                       W8ContiguousOutput, true>
        <<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void dispatch_medium_shape(const Tensor& x, const Weight& weight, Tensor& residual_out,
                           cudaStream_t stream) {
    // (rows, k) is vetted by w8_linear_add_medium_splitk_launch before any arm
    // is reached; this only picks the bake.
    if (weight.k == 4096) {
        launch_medium<4096, TileCols, KSplits, NGroups, MinBlocks>(x, residual_out, weight, stream);
    } else {
        launch_medium<6144, TileCols, KSplits, NGroups, MinBlocks>(x, residual_out, weight, stream);
    }
}

} // namespace

void w8_linear_add_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                     cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 linear_add split-K MMA requires exact T=2..48");
    }
    if (weight.n == 2560 || weight.n == 1024 || (weight.n == 2048 && weight.k == 2048)) {
        if (x.ne[1] > kQ4bLastExactCols) {
            throw std::invalid_argument(
                "W8 linear_add: small-target exact tables cover T=2..32");
        }
        // Every table bakes its hidden extent, so an unlisted k must be refused
        // rather than silently run through a neighbour's: qwen3-0.6b's mlp down
        // is {1024, 3072} and would otherwise have taken the 2048 table.
        if ((weight.n == 2560 && weight.k != 9216 && weight.k != 4096) ||
            (weight.n == 1024 && weight.k != 3584 && weight.k != 2048)) {
            throw std::invalid_argument("W8 linear_add exact split-K: no table for rows " +
                                        std::to_string(weight.n) + " over k " +
                                        std::to_string(weight.k));
        }
        const auto& launchers = weight.n == 2560
                                    ? (weight.k == 9216 ? kQ4bK9216Launchers : kQ4bK4096Launchers)
                                : weight.n == 1024
                                    ? (weight.k == 3584 ? kQ08K3584Launchers : kQ08K2048Launchers)
                                    : kQ2bK2048Launchers;
        launchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    } else if (weight.n == kRows && weight.k == 6144) {
        kK6144ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    } else if (weight.n == kRows && weight.k == 4096) {
        kK4096ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    } else {
        // Same rule the small-target tables above enforce. These tables bake both
        // extents -- kRows rows over k 4096 or 6144 -- so anything else must be
        // refused rather than run through a neighbour's bake.
        throw std::invalid_argument("W8 linear_add exact split-K: no table for rows " +
                                    std::to_string(weight.n) + " over k " +
                                    std::to_string(weight.k));
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_linear_add_medium_splitk_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                        cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if (t < 49 || t > 128) {
        throw std::invalid_argument("W8 linear_add medium split-K requires T=49..128");
    }
    // The medium bakes are kRows rows over k 4096 or 6144, and dispatch_medium_shape
    // selects between the two by k alone, so this is the one place that refuses
    // any other geometry -- and says which one, rather than complaining about T.
    if (weight.n != kRows || (weight.k != 4096 && weight.k != 6144)) {
        throw std::invalid_argument("W8 linear_add medium split-K: no table for rows " +
                                    std::to_string(weight.n) + " over k " +
                                    std::to_string(weight.k));
    }
    if (t <= 64) {
        dispatch_medium_shape<64, 8, 4, 1>(x, weight, residual_out, stream);
    } else if (t == 65) {
        dispatch_medium_shape<80, 8, 2, 1>(x, weight, residual_out, stream);
    } else if (t <= 72) {
        dispatch_medium_shape<72, 8, 3, 1>(x, weight, residual_out, stream);
    } else if (t <= 80) {
        dispatch_medium_shape<80, 8, 2, 1>(x, weight, residual_out, stream);
    } else if (t <= 96) {
        dispatch_medium_shape<96, 4, 6, 1>(x, weight, residual_out, stream);
    } else if (t <= 112) {
        dispatch_medium_shape<112, 4, 7, 1>(x, weight, residual_out, stream);
    } else if (t <= 120) {
        dispatch_medium_shape<120, 4, 5, 1>(x, weight, residual_out, stream);
    } else if (t <= 125) {
        dispatch_medium_shape<128, 4, 4, 1>(x, weight, residual_out, stream);
    } else {
        dispatch_medium_shape<128, 4, 8, 1>(x, weight, residual_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
