#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace sinfer::ops::detail {
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

// Qwen3-0.6B ungated fused qkv exact-T split path (rows 4096 = q2048 | k1024 |
// v1024, hidden 1024). Same structure as the gated tables above; the three-output
// entry below routes T=2..32 here and the wider bands onto the medium-T tiles.
using Qwen3Output = W8SplitOutput3<2048, 1024, 1024>;

template <int ActiveCols>
void launch_qwen3_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                              Tensor& v, cudaStream_t stream) {
    static_assert((2048 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const Qwen3Output output{static_cast<__nv_bfloat16*>(q.data),
                             static_cast<__nv_bfloat16*>(k.data),
                             static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, 4096, 1024>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_qwen3_launchers(std::index_sequence<Offsets...>) {
    return std::array<CompanionLauncher, sizeof...(Offsets)>{
        &launch_qwen3_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kQwen3Launchers = make_qwen3_launchers(
    std::make_index_sequence<kLastCompanionExactCols - kFirstExactCols + 1>{});

// TinyLlama-1.1B ungated fused qkv exact-T split path (rows 2560 = q2048 |
// k256 | v256, hidden 2048). Same structure as the Qwen3 table above; the bands
// were not measured for this shape, they are that table's, which is the closest
// registered parent.
using TinyOutput = W8SplitOutput3<2048, 256, 256>;

template <int ActiveCols>
void launch_tiny_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    static_assert((2048 % kRowsPerCta) == 0 && (256 % kRowsPerCta) == 0);
    const TinyOutput output{static_cast<__nv_bfloat16*>(q.data),
                            static_cast<__nv_bfloat16*>(k.data),
                            static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, 2560, 2048>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_tiny_launchers(std::index_sequence<Offsets...>) {
    return std::array<CompanionLauncher, sizeof...(Offsets)>{
        &launch_tiny_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kTinyLaunchers = make_tiny_launchers(
    std::make_index_sequence<kLastCompanionExactCols - kFirstExactCols + 1>{});

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_tiny_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    const TinyOutput output{static_cast<__nv_bfloat16*>(q.data),
                            static_cast<__nv_bfloat16*>(k.data),
                            static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_medium_t_splitk_kernel<2048, TileCols, KSplits, NGroups, MinBlocks>
        <<<2560 / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_qwen3_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                              Tensor& v, cudaStream_t stream) {
    const Qwen3Output output{static_cast<__nv_bfloat16*>(q.data),
                             static_cast<__nv_bfloat16*>(k.data),
                             static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_medium_t_splitk_kernel<1024, TileCols, KSplits, NGroups, MinBlocks>
        <<<4096 / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

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
    // Each table bakes a row split and a hidden size. Selecting on k alone let a
    // shape reach a table written for a different model, so the pair decides.
    const bool registered = (weight.n == 5120 && (weight.k == 1024 || weight.k == 2048)) ||
                            (weight.n == 10240 && weight.k == 2560) ||
                            (weight.n == 9216 && weight.k == 2048);
    if (!registered) {
        throw std::invalid_argument(
            "W8 attention input split-K MMA: unregistered gated geometry (n=" +
            std::to_string(weight.n) + ", k=" + std::to_string(weight.k) + ")");
    }
    if (x.ne[1] <= kLastTargetExactCols) {
        const auto& launchers = weight.n == 10240 ? kTarget4BLaunchers
                                : weight.n == 5120
                                    ? (weight.k == 1024 ? kTarget08Launchers : kTarget2BLaunchers)
                                    : kTargetLaunchers;
        launchers[x.ne[1] - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else {
        launch_target_medium_cols<64, 4, 2, 2>(x, weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    // Qwen3-0.6B's ungated parent. Its route table hands this schedule T=2..64.
    if (weight.n == 4096) {
        if (x.ne[1] < kFirstExactCols || x.ne[1] > 64) {
            throw std::invalid_argument("W8 qwen3 attention input split-K MMA requires T=2..64");
        }
        if (x.ne[1] <= kLastCompanionExactCols) {
            kQwen3Launchers[x.ne[1] - kFirstExactCols](x, weight, q, k, v, stream);
        } else if (x.ne[1] <= 48) {
            launch_qwen3_medium_cols<48, 4, 2, 3>(x, weight, q, k, v, stream);
        } else {
            launch_qwen3_medium_cols<64, 4, 2, 2>(x, weight, q, k, v, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // TinyLlama's ungated parent. Its route table hands this schedule T=2..64.
    if (weight.n == 2560) {
        if (x.ne[1] < kFirstExactCols || x.ne[1] > 64) {
            throw std::invalid_argument("W8 tinyllama attention input split-K MMA requires T=2..64");
        }
        if (x.ne[1] <= kLastCompanionExactCols) {
            kTinyLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, k, v, stream);
        } else if (x.ne[1] <= 48) {
            launch_tiny_medium_cols<48, 4, 2, 3>(x, weight, q, k, v, stream);
        } else {
            launch_tiny_medium_cols<64, 4, 2, 2>(x, weight, q, k, v, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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

} // namespace sinfer::ops::detail
