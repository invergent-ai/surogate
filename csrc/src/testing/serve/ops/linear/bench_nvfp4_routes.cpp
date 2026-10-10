// The W4A4 GEMMs a generic NVFP4 shape can take -- cuBLASLt and every CUTLASS SM120 tile
// `nvfp4_cutlass_gemm` offers -- on the shapes given, at the token widths a decode round and a
// prompt chunk run, so `nvfp4_cutlass_tile_for` picks a route from a measurement rather than from
// the prompt widths alone. All routes read the same quantised activation; each CUTLASS tile is
// first checked against cuBLASLt, and one that declines a shape or disagrees is not timed. The
// activation quantiser is timed once per width, since every W4A4 route pays it.
//
//   sinfer_nvfp4_route_bench                       correctness on two Gemma 4 12B shapes
//   sinfer_nvfp4_route_bench bench n k [n k ...]   correctness, then times
//
// SUROGATE_NVFP4_BENCH_T=8,32 replaces the default widths. The bench pins
// SUROGATE_SERVE_NVFP4_CUTLASS=off so `nvfp4_cublaslt_gemm` is cuBLASLt itself.

#include "core/arena.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/w8a8/w4fp4_cutlass_gemm.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::ops::detail;

namespace {

struct Tile {
    const char* name;
    int index;
};

// `nvfp4_cutlass_gemm`'s tile indices, named as SUROGATE_SERVE_NVFP4_CUTLASS spells them.
constexpr Tile kTiles[] = {
    {"cutlass_128", 0},   {"cutlass_256", 1},     {"cutlass_256sk", 2},
    {"cutlass_128sk", 3}, {"cutlass_256swap", 4}, {"cutlass_128swap", 5},
};

std::vector<int> widths() {
    std::vector<int> out;
    if (const char* env = std::getenv("SUROGATE_NVFP4_BENCH_T"); env != nullptr && *env != '\0') {
        std::string text(env);
        std::size_t at = 0;
        while (at < text.size()) {
            const std::size_t comma = text.find(',', at);
            out.push_back(std::atoi(text.substr(at, comma - at).c_str()));
            if (comma == std::string::npos) { break; }
            at = comma + 1;
        }
        return out;
    }
    return {2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048};
}

std::vector<float> read_bf16(const void* device, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    cuda_check(cudaMemcpy(bits.data(), device, count * 2, cudaMemcpyDeviceToHost), "read output");
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t widened = static_cast<std::uint32_t>(bits[i]) << 16U;
        std::memcpy(&out[i], &widened, sizeof(float));
    }
    return out;
}

/// Largest |a - b| over the largest |b|. The routes multiply the same codes and scales and
/// differ only in accumulation order, so anything past BF16 rounding is a wrong result.
double relative_error(const std::vector<float>& actual, const std::vector<float>& reference) {
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(actual[i])) { return INFINITY; }
        worst = std::max(worst, std::fabs(static_cast<double>(actual[i]) - reference[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(reference[i])));
    }
    return scale == 0.0 ? worst : worst / scale;
}

struct Problem {
    std::int32_t n = 0;
    std::int32_t k = 0;
    DeviceBuffer payload;
    Weight weight{};
};

Problem make_problem(std::int32_t n, std::int32_t k) {
    Problem p;
    p.n = n;
    p.k = k;
    const quantized_weight::PackedWeight host = sinfer::test::linear::make_nvfp4_weight(n, k, 23U);
    p.payload = DeviceBuffer(host.payload.size());
    p.payload.copy_from_host(host.payload.data(), p.payload.bytes);
    p.weight = host.device_weight(p.payload.p);
    return p;
}

/// One width's operands: the BF16 activation, its W4A4 quantisation in the tiled scale layout
/// both routes read, and an output plane.
struct Operands {
    DeviceBuffer x_buffer;
    DeviceArena workspace;
    Nvfp4W4a4Workspace quantised{};
    DeviceBuffer out_buffer;
    Tensor x;
    Tensor out;

    Operands(const Problem& p, int t)
        : x_buffer([&] {
              std::vector<float> values(static_cast<std::size_t>(p.k) * t);
              fill_uniform(values, 31U + static_cast<std::uint32_t>(t), -2.0F, 2.0F);
              return to_device_bf16(values);
          }()),
          workspace(nvfp4_w4a4_workspace_capacity_bytes(t, p.k) + 4096),
          out_buffer(static_cast<std::size_t>(p.n) * t * 2),
          x(x_buffer.p, DType::BF16, {p.k, t}),
          out(out_buffer.p, DType::BF16, {p.n, t}) {
        quantised = allocate_nvfp4_w4a4_workspace(workspace, t, p.k);
        quantise(p);
        cuda_check(cudaDeviceSynchronize(), "quantise");
    }

    void quantise(const Problem& p) {
        launch_nvfp4_w4a4_quantize(x, p.weight, quantised, nullptr, Nvfp4ScaleLayout::Tiled);
    }

    void cublaslt(const Problem& p) {
        nvfp4_cublaslt_gemm(p.weight, 0, p.n, quantised.codes, quantised.scales,
                            static_cast<__nv_bfloat16*>(out.data), p.n, out.ne[1], 0.0F, nullptr);
    }

    bool cutlass(const Problem& p, int tile) {
        const float alpha = 1.0F / (p.weight.input_scale_divisor * p.weight.weight_scale_divisor);
        return nvfp4_cutlass_gemm(quantised.codes, quantised.scales,
                                  static_cast<const std::uint8_t*>(p.weight.qdata),
                                  static_cast<const std::uint8_t*>(p.weight.scales), alpha, nullptr,
                                  out.data, out.ne[1], p.n, p.k, tile, nullptr);
    }
};

bool run_cutlass(Operands& o, const Problem& p, int tile) {
    try {
        if (!o.cutlass(p, tile)) { return false; }
        cuda_check(cudaDeviceSynchronize(), "cutlass");
        return true;
    } catch (const std::exception&) {
        (void)cudaGetLastError();
        return false;
    }
}

/// Every CUTLASS tile at `t` against cuBLASLt. Returns the number of tiles that ran and disagreed.
int check(const Problem& p, int t, std::vector<bool>* usable) {
    Operands o(p, t);
    const std::size_t count = static_cast<std::size_t>(p.n) * t;
    o.cublaslt(p);
    cuda_check(cudaDeviceSynchronize(), "cuBLASLt");
    const std::vector<float> expected = read_bf16(o.out.data, count);
    int failures = 0;
    for (std::size_t i = 0; i < std::size(kTiles); ++i) {
        cuda_check(cudaMemset(o.out.data, 0xff, count * 2), "poison");
        if (!run_cutlass(o, p, kTiles[i].index)) {
            if (usable != nullptr) { (*usable)[i] = false; }
            std::printf("n=%6d k=%6d T=%4d  %-16s declines\n", p.n, p.k, t, kTiles[i].name);
            continue;
        }
        const double error = relative_error(read_bf16(o.out.data, count), expected);
        const bool bad     = !(error <= 1e-2);
        if (bad) {
            ++failures;
            if (usable != nullptr) { (*usable)[i] = false; }
        }
        std::printf("n=%6d k=%6d T=%4d  %-16s relative error %.2e%s\n", p.n, p.k, t,
                    kTiles[i].name, error, bad ? "  WRONG" : "");
    }
    return failures;
}

double time_us(const std::function<void()>& launch, double bytes) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    launch();
    cuda_check(cudaDeviceSynchronize(), "warm-up");
    cuda_check(cudaEventRecord(start), "record");
    launch();
    cuda_check(cudaEventRecord(stop), "record");
    cuda_check(cudaEventSynchronize(stop), "sync");
    float first_ms = 0;
    cuda_check(cudaEventElapsedTime(&first_ms, start, stop), "elapsed");
    // At least ~1 GB of weight reads or 5 calls, at most 2 s of GPU time.
    int iters = std::max(5, static_cast<int>(1e9 / std::max(bytes, 1.0)));
    iters     = std::min(iters, std::max(1, static_cast<int>(2000.0 / std::max(first_ms, 1e-3F))));
    cuda_check(cudaEventRecord(start), "record");
    for (int i = 0; i < iters; ++i) { launch(); }
    cuda_check(cudaEventRecord(stop), "record");
    cuda_check(cudaEventSynchronize(stop), "sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cuda_check(cudaEventDestroy(start), "event");
    cuda_check(cudaEventDestroy(stop), "event");
    return ms * 1000.0 / iters;
}

void bench(const Problem& p) {
    // Codes at half a byte, one UE4M3 scale per 16.
    const double bytes = static_cast<double>(p.n) * p.k * (0.5 + 1.0 / 16);
    std::vector<bool> usable(std::size(kTiles), true);
    int failures = 0;
    for (const int t : {8, 32, 128}) { failures += check(p, t, &usable); }
    if (failures != 0) { std::printf("n=%6d k=%6d  %d tile checks WRONG\n", p.n, p.k, failures); }
    for (const int t : widths()) {
        Operands o(p, t);
        const double flops = 2.0 * p.n * p.k * t;
        const auto report  = [&](const char* name, double us) {
            std::printf("n=%6d k=%6d T=%4d  %-16s %9.1f us  %6.1f GB/s  %6.1f TFLOP/s\n", p.n, p.k,
                        t, name, us, bytes / us * 1e-3, flops / us * 1e-6);
        };
        const double quantise = time_us([&] { o.quantise(p); }, bytes);
        std::printf("n=%6d k=%6d T=%4d  %-16s %9.1f us\n", p.n, p.k, t, "quantise", quantise);
        double best           = time_us([&] { o.cublaslt(p); }, bytes);
        const char* best_name = "cublaslt";
        report("cublaslt", best);
        for (std::size_t i = 0; i < std::size(kTiles); ++i) {
            if (!usable[i] || !run_cutlass(o, p, kTiles[i].index)) { continue; }
            const double us = time_us([&] { (void)o.cutlass(p, kTiles[i].index); }, bytes);
            report(kTiles[i].name, us);
            if (us < best) {
                best      = us;
                best_name = kTiles[i].name;
            }
        }
        std::printf("n=%6d k=%6d T=%4d  best %s %.1f us\n", p.n, p.k, t, best_name, best);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    // Read once by the route; pinned before the first GEMM so `nvfp4_cublaslt_gemm` is cuBLASLt.
    setenv("SUROGATE_SERVE_NVFP4_CUTLASS", "off", 1);
    try {
        nvfp4_cublaslt_prewarm();
        if (argc > 1 && std::string(argv[1]) == "bench") {
            for (int i = 2; i + 1 < argc; i += 2) {
                bench(make_problem(std::atoi(argv[i]), std::atoi(argv[i + 1])));
            }
            return 0;
        }
        // Gemma 4 12B's fused gate/up and its down projection: the two widths every tile takes.
        int failures = 0;
        for (const auto& [n, k] : {std::pair{30720, 3840}, std::pair{3840, 15360}}) {
            const Problem p = make_problem(n, k);
            for (const int t : {8, 32, 256}) { failures += check(p, t, nullptr); }
        }
        std::printf("%s NVFP4 CUTLASS tiles against cuBLASLt (%d wrong)\n",
                    failures == 0 ? "OK" : "FAIL", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sinfer_nvfp4_route_bench: %s\n", error.what());
        return 1;
    }
}
