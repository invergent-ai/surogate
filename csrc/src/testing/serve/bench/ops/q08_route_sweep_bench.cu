// surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b W8 prefill route sweep.
//
// The q08 route tables were installed correctness-first (everything large-T
// on MmaR64C128). This bench instantiates the shared runtime-shaped
// w8_rowsplit_gemm_mma_kernel with every candidate tile schedule directly
// and measures the four dominant prefill GEMMs of the 0.8B target at
// prefill-chunk token counts, so the tables can be replaced with measured
// winners without guessing:
//
//   gate_up   7168 x 1024  (linear_swiglu; ~19% of prefill GPU time)
//   qkvz      8192 x 1024  (gdn input; SplitOutput2 6144/2048)
//   down      1024 x 3584  (linear_add residual)
//   output    1024 x 2048  (linear_add residual)
//   qkgv      5120 x 1024  (attn input; SplitOutput4)
//
// Epilogues are benched as plain stores: epilogue cost is identical across
// tile candidates, so store-only ranking transfers to the fused forms.

#include "core/device.h"
#include "sinfer_bench_common.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::ops::detail;

namespace {

struct Case {
    const char* name;
    std::int32_t rows;
    std::int32_t k;
};

constexpr Case kCases[] = {
    {"gate_up_7168x1024", 7168, 1024},
    {"qkvz_8192x1024", 8192, 1024},
    {"down_1024x3584", 1024, 3584},
    {"output_1024x2048", 1024, 2048},
    {"qkgv_5120x1024", 5120, 1024},
    // qwen3.5-2b-specific shapes (PATCHES.md #16); its gate_up/down are the
    // measured 35B shapes already.
    {"qkgv_5120x2048", 5120, 2048},
    {"qkvz_8192x2048", 8192, 2048},
    {"output_2048x2048", 2048, 2048},
};

constexpr std::int32_t kTokenCounts[] = {472, 888, 1024, 1912};

template <class Schedule>
void launch_case(const Case& c, const __nv_bfloat16* x, const std::uint8_t* codes,
                 const std::uint8_t* scales, __nv_bfloat16* out, std::int32_t tokens,
                 cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(sinfer::ops::div_up(c.rows, Schedule::BM)),
                    static_cast<unsigned>(sinfer::ops::div_up(tokens, Schedule::BN)), 1u);
    const W8ContiguousOutput output{out, c.rows};
    if ((tokens % Schedule::BN) == 0) {
        w8_rowsplit_gemm_mma_kernel<Schedule, true, W8Epilogue::Store>
            <<<grid, Schedule::THREADS, 0, stream>>>(x, codes, scales, output, c.rows, c.k, tokens,
                                                     c.k);
    } else {
        w8_rowsplit_gemm_mma_kernel<Schedule, false, W8Epilogue::Store>
            <<<grid, Schedule::THREADS, 0, stream>>>(x, codes, scales, output, c.rows, c.k, tokens,
                                                     c.k);
    }
}

using LaunchFn = void (*)(const Case&, const __nv_bfloat16*, const std::uint8_t*,
                          const std::uint8_t*, __nv_bfloat16*, std::int32_t, cudaStream_t);

struct Candidate {
    const char* name;
    std::int32_t bm;
    LaunchFn launch;
};

// Every registered schedule family shape (mirrors the linear_add/swiglu
// launcher sets) plus wider row tiles. Template order is
// <BM, BN, WM, WN, MIN_BLOCKS, STAGES=2, BK=64>.
const Candidate kCandidates[] = {
    {"r32c64", 32, &launch_case<W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>>},
    {"r32c96", 32, &launch_case<W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>>},
    {"r32c128", 32, &launch_case<W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>>},
    {"r48c96", 48, &launch_case<W8RowSplitMmaGemmSchedule<48, 96, 48, 16, 2>>},
    {"r48c128", 48, &launch_case<W8RowSplitMmaGemmSchedule<48, 128, 48, 16, 2>>},
    {"r64c64", 64, &launch_case<W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>>},
    {"r64c96", 64, &launch_case<W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>>},
    {"r64c128", 64, &launch_case<W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>>},
    {"r96c96", 96, &launch_case<W8RowSplitMmaGemmSchedule<96, 96, 48, 16, 2>>},
    {"r128c64", 128, &launch_case<W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>>},
    {"r128c80", 128, &launch_case<W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>>},
    {"r128c96a1", 128, &launch_case<W8RowSplitMmaGemmSchedule<128, 96, 64, 16, 2, 2, 64, 1>>},
    {"r128c128a1", 128, &launch_case<W8RowSplitMmaGemmSchedule<128, 128, 64, 16, 2, 2, 64, 1>>},
};

} // namespace

// Materialized-path probe: dequant W8 -> BF16 once (bandwidth-trivial), then
// the engine's BF16 MMA production schedule at full rate. If bf16-mma
// dominates the fused W8 tiles at large T, a materialize route wins.
template <class Geometry>
void launch_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out,
                 std::int32_t tokens, cudaStream_t stream) {
    using Schedule        = Bf16MmaProductionSchedule<Geometry>;
    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int tiles_n     = sinfer::ops::div_up(tokens, Schedule::kBlockCols);
    const Bf16MmaContiguousOutput output{out, Geometry::kOutputRows};
    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gemm_mma_kernel<Geometry, Schedule, false, Bf16MmaContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        (void)attr;
    }
    bf16_gemm_mma_kernel<Geometry, Schedule, false>
        <<<tiles_m * tiles_n, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(x, w, output,
                                                                                   tokens);
}

using Bf16Fn = void (*)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                        cudaStream_t);

Bf16Fn bf16_launcher(std::int32_t rows, std::int32_t k) {
    if (rows == 7168 && k == 1024) return &launch_bf16<Bf16GemvGeometry<7168, 1024>>;
    if (rows == 8192 && k == 1024) return &launch_bf16<Bf16GemvGeometry<8192, 1024>>;
    if (rows == 1024 && k == 3584) return &launch_bf16<Bf16GemvGeometry<1024, 3584>>;
    if (rows == 1024 && k == 2048) return &launch_bf16<Bf16GemvGeometry<1024, 2048>>;
    if (rows == 5120 && k == 1024) return &launch_bf16<Bf16GemvGeometry<5120, 1024>>;
    return nullptr;
}

int main(int argc, char** argv) {
    int warmup = 20, repeat = 100;
    if (argc > 1) warmup = std::atoi(argv[1]);
    if (argc > 2) repeat = std::atoi(argv[2]);

    cudaStream_t stream = nullptr;
    DeviceBuffer flush  = bench::make_zeros(96u << 20);

    std::printf("%-20s %6s", "case", "tokens");
    for (const Candidate& cand : kCandidates) std::printf(" %10s", cand.name);
    std::printf("\n");

    for (const Case& c : kCases) {
        bench::PackedQuantizedWeight weight =
            bench::make_row_split_weight(QType::W8G32_F16S, c.rows, c.k, c.k);
        const std::int32_t max_tokens = 1912;
        DeviceBuffer x   = bench::make_bf16(static_cast<std::size_t>(c.k) * max_tokens);
        DeviceBuffer out = bench::make_zeros(static_cast<std::size_t>(c.rows) * max_tokens * 2);
        const auto* codes  = static_cast<const std::uint8_t*>(weight.weight.qdata);
        const auto* scales = static_cast<const std::uint8_t*>(weight.weight.scales);

        DeviceBuffer wbf16 = bench::make_bf16(static_cast<std::size_t>(c.rows) * c.k);
        const Bf16Fn bf16   = bf16_launcher(c.rows, c.k);
        for (const std::int32_t tokens : kTokenCounts) {
            std::printf("%-20s %6d", c.name, tokens);
            const double gflop = 2.0 * c.rows * c.k * static_cast<double>(tokens) / 1e9;
            for (const Candidate& cand : kCandidates) {
                const auto launch = [&](cudaStream_t s) {
                    cand.launch(c, static_cast<const __nv_bfloat16*>(x.p), codes, scales,
                                static_cast<__nv_bfloat16*>(out.p), tokens, s);
                };
                const auto timing = bench::measure_launch(launch, stream, warmup, repeat);
                const double us   = timing.median_us;
                std::printf(" %7.1f/%2.0f", us, gflop / (us * 1e-6) / 1e3);
            }
            if (bf16 != nullptr) {
                const auto launch = [&](cudaStream_t s) {
                    bf16(static_cast<const __nv_bfloat16*>(x.p),
                         static_cast<const __nv_bfloat16*>(wbf16.p),
                         static_cast<__nv_bfloat16*>(out.p), tokens, s);
                };
                const auto timing = bench::measure_launch(launch, stream, warmup, repeat);
                std::printf(" | bf16 %7.1f/%2.0f", timing.median_us,
                            gflop / (timing.median_us * 1e-6) / 1e3);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }
    return 0;
}
