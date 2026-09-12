#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"

#include <stdexcept>
#include <string>
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

namespace sinfer::ops::detail {
namespace {

constexpr int kTargetRows    = 9216;
constexpr int kCompanionRows = 6144;
constexpr int kHidden        = 2048;
using TargetOutput           = W8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput        = W8SplitOutput3<4096, 1024, 1024>;

// Load BF16 MMA fragments directly for a narrow batch. This preserves the
// prefill kernel's weight rounding and sequence of k16 MMAs while avoiding its
// shared-memory transpose, barriers, and padded 128-column work. Replicating
// four weight rows in the fragment supplies enough independent CTAs for decode.
template <class Output>
__global__ __launch_bounds__(32) void w8_narrow_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::int8_t* __restrict__ codes,
    const __half* __restrict__ scales, Output output, int tokens) {
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 4;
    const int row = row0 + ((lane >> 2) & 3);
    const int pair = 2 * (lane & 3);
    const int input_col = lane >> 2;
    float acc[4] = {};
#pragma unroll 16
    for (int group = 0; group < kHidden / 32; ++group) {
        const float s0 = __half2float(scales[row * (kHidden / 32) + group]);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int column = group * 32 + half * 16 + pair;
            const auto pack = [&](int r, int c, float scale) {
                const auto* w = codes + r * kHidden + c;
                const auto v = __floats2bfloat162_rn(w[0] * scale, w[1] * scale);
                return reinterpret_cast<const unsigned&>(v);
            };
            const unsigned a0 = pack(row, column, s0);
            const unsigned a1 = a0;
            const unsigned a2 = pack(row, column + 8, s0);
            const unsigned a3 = a2;
            unsigned b0 = 0, b1 = 0;
            if (input_col < tokens) {
                const auto* in = reinterpret_cast<const unsigned*>(x + input_col * kHidden + column);
                b0 = in[0];
                b1 = in[4];
            }
            mma_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
        }
    }
    const auto tile = output.tile(row0);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const int col = pair + (e & 1);
        if (col < tokens && (lane >> 2) < 4) {
            *tile.at(row, col) = __float2bfloat16_rn(acc[e]);
        }
    }
}

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

void w8_attn_input_mma_r4_c8_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    if (weight.n != 2560 || weight.k != 2048) {
        throw std::invalid_argument("W8 narrow attention input MMA: expected TinyLlama geometry");
    }
    using Output = W8SplitOutput3<2048, 256, 256>;
    const Output output{static_cast<__nv_bfloat16*>(q.data),
                        static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_narrow_mma_kernel<<<2560 / 4, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::int8_t*>(weight.qdata),
        static_cast<const __half*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

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
    // Qwen3's ungated fused qkv: rows 4096 = q2048 | k1024 | v1024, at hidden 1024 for the
    // 0.6B and 2048 for the 1.7B. One row split, two K, because 16 query heads and 8 KV heads
    // at head dim 128 is the family's attention at both sizes.
    if (weight.n == 4096) {
        using OutputQwen3 = W8SplitOutput3<2048, 1024, 1024>;
        static_assert((2048 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
        const OutputQwen3 output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
        if (weight.k == 2048) {
            launch_route<Schedule, 4096, 2048>(x, weight, output, stream);
        } else {
            launch_route<Schedule, 4096, 1024>(x, weight, output, stream);
        }
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
    // LFM2-1.2B ungated fused qkv (rows 3072 = q2048 | k512 | v512, hidden 2048): 32 query
    // heads and 8 KV heads at head dim 64.
    if (weight.n == 3072) {
        using OutputLfm2 = W8SplitOutput3<2048, 512, 512>;
        static_assert((2048 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
        const OutputLfm2 output{static_cast<__nv_bfloat16*>(q.data),
                                static_cast<__nv_bfloat16*>(k.data),
                                static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 3072, 2048>(x, weight, output, stream);
        return;
    }
    // Only the companion is left. Falling through to it for any parent that reached here is how
    // a 3072-row weight came to be read by a kernel baked for 6144 rows.
    if (weight.n != kCompanionRows || weight.k != kHidden) {
        throw std::invalid_argument(
            "W8 attention input MMA: unregistered ungated geometry (n=" +
            std::to_string(weight.n) + ", k=" + std::to_string(weight.k) + ")");
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
    // Qwen3's ungated fused qkv: rows 4096 = q2048 | k1024 | v1024, at hidden 1024 for the
    // 0.6B and 2048 for the 1.7B. One row split, two K, because 16 query heads and 8 KV heads
    // at head dim 128 is the family's attention at both sizes.
    if (weight.n == 4096) {
        using OutputQwen3 = W8SplitOutput3<2048, 1024, 1024>;
        static_assert((2048 % Schedule::BM) == 0 && (1024 % Schedule::BM) == 0);
        const OutputQwen3 output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
        if (weight.k == 2048) {
            launch_route<Schedule, 4096, 2048>(x, weight, output, stream);
        } else {
            launch_route<Schedule, 4096, 1024>(x, weight, output, stream);
        }
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
    // LFM2-1.2B ungated fused qkv (rows 3072 = q2048 | k512 | v512, hidden 2048): 32 query
    // heads and 8 KV heads at head dim 64.
    if (weight.n == 3072) {
        using OutputLfm2 = W8SplitOutput3<2048, 512, 512>;
        static_assert((2048 % Schedule::BM) == 0 && (512 % Schedule::BM) == 0);
        const OutputLfm2 output{static_cast<__nv_bfloat16*>(q.data),
                                static_cast<__nv_bfloat16*>(k.data),
                                static_cast<__nv_bfloat16*>(v.data)};
        launch_route<Schedule, 3072, 2048>(x, weight, output, stream);
        return;
    }
    // Only the companion is left. Falling through to it for any parent that reached here is how
    // a 3072-row weight came to be read by a kernel baked for 6144 rows.
    if (weight.n != kCompanionRows || weight.k != kHidden) {
        throw std::invalid_argument(
            "W8 attention input MMA: unregistered ungated geometry (n=" +
            std::to_string(weight.n) + ", k=" + std::to_string(weight.k) + ")");
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
