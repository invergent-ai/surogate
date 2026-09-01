#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

namespace sinfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;
constexpr int kHidden        = 2048;
using TargetOutput           = W8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput        = W8SplitOutput3<4096, 1024, 1024>;

template <class Schedule, bool Full, int Rows, int Hidden, class Output>
void launch_variant(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    const dim3 grid(Rows / Schedule::BM, static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::Store, Output>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const std::uint8_t*>(weight.scales),
                                                 output, Rows, Hidden, x.ne[1], Hidden);
}

template <class Schedule, int Rows, int Hidden, class Output>
void launch_route(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<Schedule, true, Rows, Hidden>(x, weight, output, stream);
    } else {
        launch_variant<Schedule, false, Rows, Hidden>(x, weight, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b fused qkgv.
    if (weight.n == 10240) {
        using Output4B = W8SplitOutput4<4096, 1024, 4096, 1024>;
        const Output4B output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 10240, 2560>(x, weight, output, stream);
        return;
    }
    // surogate vendor patches (PATCHES.md #13/#16): qwen3.5-0.8b/-2b qkgv.
    if (weight.n == 5120) {
        using Output08 = W8SplitOutput4<2048, 512, 2048, 512>;
        const Output08 output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
        if (weight.k == 1024) {
            launch_route<Schedule, 5120, 1024>(x, weight, output, stream);
        } else {
            launch_route<Schedule, 5120, 2048>(x, weight, output, stream);
        }
        return;
    }
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kTargetRows, kHidden>(x, weight, output, stream);
}

void w8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b fused qkgv.
    if (weight.n == 10240) {
        using Output4B = W8SplitOutput4<4096, 1024, 4096, 1024>;
        const Output4B output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 10240, 2560>(x, weight, output, stream);
        return;
    }
    // surogate vendor patches (PATCHES.md #13/#16): qwen3.5-0.8b/-2b qkgv.
    if (weight.n == 5120) {
        using Output08 = W8SplitOutput4<2048, 512, 2048, 512>;
        const Output08 output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
        if (weight.k == 1024) {
            launch_route<Schedule, 5120, 1024>(x, weight, output, stream);
        } else {
            launch_route<Schedule, 5120, 2048>(x, weight, output, stream);
        }
        return;
    }
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kTargetRows, kHidden>(x, weight, output, stream);
}

void w8_attn_input_mma_r32_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    // Qwen3-0.6B ungated fused qkv (rows 4096 = q2048 | k1024 | v1024, hidden 1024).
    if (weight.n == 4096) {
        using OutputQwen3 = W8SplitOutput3<2048, 1024, 1024>;
        static_assert((2048 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
        const OutputQwen3 output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 4096, 1024>(x, weight, output, stream);
        return;
    }
    // TinyLlama-1.1B ungated fused qkv (rows 2560 = q2048 | k256 | v256, hidden
    // 2048): 32 query heads and 4 KV heads at head dim 64, the narrowest KV
    // plane the row-split epilogue has carried.
    if (weight.n == 2560) {
        using OutputTiny = W8SplitOutput3<2048, 256, 256>;
        static_assert((2048 % Schedule::BM) == 0 && (256 % Schedule::BM) == 0);
        const OutputTiny output{static_cast<__nv_bfloat16*>(q.data),
                                static_cast<__nv_bfloat16*>(k.data),
                                static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 2560, 2048>(x, weight, output, stream);
        return;
    }
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                       Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    // Qwen3-0.6B ungated fused qkv (rows 4096 = q2048 | k1024 | v1024, hidden 1024).
    if (weight.n == 4096) {
        using OutputQwen3 = W8SplitOutput3<2048, 1024, 1024>;
        static_assert((2048 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
        const OutputQwen3 output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 4096, 1024>(x, weight, output, stream);
        return;
    }
    // TinyLlama-1.1B ungated fused qkv (rows 2560 = q2048 | k256 | v256, hidden
    // 2048). The 256-row KV planes are exactly four of this schedule's 64-row
    // tiles, which is the whole reason this tile size still applies.
    if (weight.n == 2560) {
        using OutputTiny = W8SplitOutput3<2048, 256, 256>;
        static_assert((2048 % Schedule::BM) == 0 && (256 % Schedule::BM) == 0);
        const OutputTiny output{static_cast<__nv_bfloat16*>(q.data),
                                static_cast<__nv_bfloat16*>(k.data),
                                static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 2560, 2048>(x, weight, output, stream);
        return;
    }
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_companion_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_companion_attn_input_mma_r64_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_companion_attn_input_mma_r32_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_companion_attn_input_mma_r64_c96_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_companion_attn_input_mma_r128_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

void w8_companion_attn_input_mma_r128_c80_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2, 2>;
    static_assert((4096 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_route<Schedule, kCompanionRows, kHidden>(x, weight, output, stream);
}

} // namespace sinfer::ops::detail
