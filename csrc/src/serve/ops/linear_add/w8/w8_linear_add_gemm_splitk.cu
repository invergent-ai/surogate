#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <span>
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

template <int Hidden, int Rows, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets), Rows>...};
}

// One table per baked (rows, k) geometry, T = kFirstExactCols..LastCols.
template <int Hidden, int Rows, int LastCols>
constexpr auto make_launchers() {
    static_assert(LastCols >= kFirstExactCols && LastCols <= kLastExactCols);
    return make_launchers<Hidden, Rows>(std::make_index_sequence<LastCols - kFirstExactCols + 1>{});
}

// The 35B residual projections bake the whole T=2..48 band.
constexpr auto kR2048K4096Launchers = make_launchers<4096, 2048, kLastExactCols>();
constexpr auto kR2048K6144Launchers = make_launchers<6144, 2048, kLastExactCols>();

// Every other geometry bakes the C=32 batch band (PATCHES.md #29): T=2..16 is
// batch decode, where the runtime-shaped SIMT read weights ceil(T/4) times;
// 17..32 is reachable only when the Marlin band declines the call.
constexpr int kSmallLastExactCols = 32;
// surogate vendor patch (PATCHES.md #29): qwen3.5-4b o_proj (2560x4096) and
// mlp down (2560x9216).
constexpr auto kR2560K4096Launchers = make_launchers<4096, 2560, kSmallLastExactCols>();
constexpr auto kR2560K9216Launchers = make_launchers<9216, 2560, kSmallLastExactCols>();
// surogate vendor patch (PATCHES.md #29): qwen3.5-2b o_proj (2048x2048) and
// qwen3.5-0.8b o_proj (1024x2048) / down (1024x3584).
constexpr auto kR2048K2048Launchers = make_launchers<2048, 2048, kSmallLastExactCols>();
constexpr auto kR1024K2048Launchers = make_launchers<2048, 1024, kSmallLastExactCols>();
constexpr auto kR1024K3584Launchers = make_launchers<3584, 1024, kSmallLastExactCols>();
// tinyllama-1.1b mlp down (2048x5632): 11 groups of 512 at KWarps=8.
constexpr auto kR2048K5632Launchers = make_launchers<5632, 2048, kSmallLastExactCols>();

// (rows, k) is the key by construction: every table bakes both extents, so a
// shape not listed here is refused rather than run through a neighbour's bake
// (qwen3-0.6b's {1024, 3072} would otherwise have taken the 2048 table, and
// tinyllama's {2048, 5632} once took the 4096 one).
struct ExactTable {
    std::int32_t rows;
    std::int32_t k;
    std::span<const ProjectionLauncher> launchers;

    constexpr std::int32_t last_cols() const noexcept {
        return kFirstExactCols + static_cast<std::int32_t>(launchers.size()) - 1;
    }
};

constexpr std::array<ExactTable, 8> kExactTables{{
    {2048, 4096, kR2048K4096Launchers},
    {2048, 6144, kR2048K6144Launchers},
    {2560, 4096, kR2560K4096Launchers},
    {2560, 9216, kR2560K9216Launchers},
    {2048, 2048, kR2048K2048Launchers},
    {1024, 2048, kR1024K2048Launchers},
    {1024, 3584, kR1024K3584Launchers},
    {2048, 5632, kR2048K5632Launchers},
}};

constexpr bool exact_tables_are_keyed_once() {
    for (std::size_t i = 0; i < kExactTables.size(); ++i) {
        if (kExactTables[i].launchers.empty()) { return false; }
        for (std::size_t j = i + 1; j < kExactTables.size(); ++j) {
            if (kExactTables[i].rows == kExactTables[j].rows &&
                kExactTables[i].k == kExactTables[j].k) {
                return false;
            }
        }
    }
    return true;
}
static_assert(exact_tables_are_keyed_once(), "one exact-T table per (rows, k)");

const ExactTable* find_table(std::int32_t rows, std::int32_t k) noexcept {
    for (const ExactTable& table : kExactTables) {
        if (table.rows == rows && table.k == k) { return &table; }
    }
    return nullptr;
}

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

std::int32_t w8_linear_add_exact_t_last_cols(std::int32_t rows, std::int32_t k) noexcept {
    const ExactTable* table = find_table(rows, k);
    return table == nullptr ? 0 : table->last_cols();
}

bool w8_linear_add_exact_t_covers(std::int32_t rows, std::int32_t k, std::int32_t t) noexcept {
    return t >= kFirstExactCols && t <= w8_linear_add_exact_t_last_cols(rows, k);
}

void w8_linear_add_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                     cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 linear_add split-K MMA requires exact T=2..48");
    }
    const ExactTable* table = find_table(weight.n, weight.k);
    if (table == nullptr) {
        throw std::invalid_argument("W8 linear_add exact split-K: no table for rows " +
                                    std::to_string(weight.n) + " over k " +
                                    std::to_string(weight.k));
    }
    if (x.ne[1] > table->last_cols()) {
        throw std::invalid_argument("W8 linear_add exact split-K: the table for rows " +
                                    std::to_string(weight.n) + " over k " +
                                    std::to_string(weight.k) + " covers T=2.." +
                                    std::to_string(table->last_cols()));
    }
    table->launchers[static_cast<std::size_t>(x.ne[1] - kFirstExactCols)](x, weight, residual_out,
                                                                          stream);
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
