#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows             = 9216;
constexpr int kCompanionRows          = 6144;
constexpr int kHidden                 = 2048;
constexpr int kRowsPerCta             = 16;
constexpr int kFirstExactCols         = 2;
constexpr int kLastTargetExactCols    = 48;
constexpr int kLastCompanionExactCols = 32;
using TargetOutput                    = W8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput                 = W8SplitOutput3<4096, 1024, 1024>;
using TargetLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                cudaStream_t);
using CompanionLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                   cudaStream_t);

template <int ActiveCols, int Rows, int Hidden, class Output>
void launch_output(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    using Geometry         = W8LinearGeometry<Rows, Hidden>;
    using Schedule         = W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>;
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<Rows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols>
void launch_target_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, kTargetRows, kHidden>(x, weight, output, stream);
}

template <int ActiveCols>
void launch_companion_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, kCompanionRows, kHidden>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_target_launchers(std::index_sequence<Offsets...>) {
    return std::array<TargetLauncher, sizeof...(Offsets)>{
        &launch_target_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_companion_launchers(std::index_sequence<Offsets...>) {
    return std::array<CompanionLauncher, sizeof...(Offsets)>{
        &launch_companion_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

// surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b fused qkgv.
template <int ActiveCols>
void launch_target08_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Output08 = W8SplitOutput4<2048, 512, 2048, 512>;
    const Output08 output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, 5120, 1024>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_target08_launchers(std::index_sequence<Offsets...>) {
    return std::array<TargetLauncher, sizeof...(Offsets)>{
        &launch_target08_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kTarget08Launchers = make_target08_launchers(
    std::make_index_sequence<kLastTargetExactCols - kFirstExactCols + 1>{});

// surogate vendor patch (PATCHES.md #28): qwen3.5-2b fused qkgv exact-T split
// path (hidden 2048) — the table whose absence routed the 2..64 band onto the
// runtime-shaped MMA tiles (141us at T=8 in batch decode).
template <int ActiveCols>
void launch_target2b_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Output2B = W8SplitOutput4<2048, 512, 2048, 512>;
    const Output2B output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, 5120, 2048>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_target2b_launchers(std::index_sequence<Offsets...>) {
    return std::array<TargetLauncher, sizeof...(Offsets)>{
        &launch_target2b_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kTarget2BLaunchers = make_target2b_launchers(
    std::make_index_sequence<kLastTargetExactCols - kFirstExactCols + 1>{});

// surogate vendor patch (PATCHES.md #28): qwen3.5-4b fused qkgv exact-T split
// path (rows 10240, hidden 2560; segment order q,k,gate,v matches the T=1
// decode kernel's Output4B).
template <int ActiveCols>
void launch_target4b_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((10240 % kRowsPerCta) == 0);
    using Output4B = W8SplitOutput4<4096, 1024, 4096, 1024>;
    const Output4B output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, 10240, 2560>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_target4b_launchers(std::index_sequence<Offsets...>) {
    return std::array<TargetLauncher, sizeof...(Offsets)>{
        &launch_target4b_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kTarget4BLaunchers = make_target4b_launchers(
    std::make_index_sequence<kLastTargetExactCols - kFirstExactCols + 1>{});

constexpr auto kTargetLaunchers =
    make_target_launchers(std::make_index_sequence<kLastTargetExactCols - kFirstExactCols + 1>{});
constexpr auto kCompanionLaunchers = make_companion_launchers(
    std::make_index_sequence<kLastCompanionExactCols - kFirstExactCols + 1>{});

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_target_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b fused qkgv.
    if (weight.k == 1024) {
        using Output08 = W8SplitOutput4<2048, 512, 2048, 512>;
        const Output08 output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
        w8_rowsplit_medium_t_splitk_kernel<1024, TileCols, KSplits, NGroups, MinBlocks>
            <<<5120 / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
        return;
    }
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kTargetRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_companion_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kCompanionRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

} // namespace

void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 64) {
        throw std::invalid_argument("W8 attention input split-K MMA requires T=2..64");
    }
    if (x.ne[1] <= kLastTargetExactCols) {
        const auto& launchers = weight.k == 1024   ? kTarget08Launchers
                                : weight.n == 10240 ? kTarget4BLaunchers
                                : weight.n == 5120  ? kTarget2BLaunchers
                                                    : kTargetLaunchers;
        launchers[x.ne[1] - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else {
        launch_target_medium_cols<64, 4, 2, 2>(x, weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 96) {
        throw std::invalid_argument("W8 companion attention input split-K MMA requires T=2..96");
    }
    if (x.ne[1] <= kLastCompanionExactCols) {
        kCompanionLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, k, v, stream);
    } else if (x.ne[1] <= 48) {
        launch_companion_medium_cols<48, 4, 2, 3>(x, weight, q, k, v, stream);
    } else if (x.ne[1] <= 64) {
        launch_companion_medium_cols<64, 4, 2, 2>(x, weight, q, k, v, stream);
    } else {
        launch_companion_medium_cols<96, 2, 4, 3>(x, weight, q, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
