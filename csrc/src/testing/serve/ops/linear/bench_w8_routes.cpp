// Every W8G32 launcher on the shapes given, at the token widths a w8_dispatch.cpp band covers, so
// a shape gets its entry from a measurement on the GPU it serves on rather than by analogy with
// one that was measured somewhere else. Each launcher is first checked against SIMT r8_c4, which
// is exact at any shape; one that declines a shape (throws) or disagrees is reported and not
// timed. `dispatch` is ops::linear as the dispatcher routes the shape today, Marlin band included.
//
//   sinfer_w8_route_bench                       correctness on two aligned Gemma shapes
//   sinfer_w8_route_bench bench n k [n k ...]   correctness, then times
//
// SUROGATE_W8_BENCH_T=1,8,32 replaces the default widths. A shape the dispatcher keeps off the
// MMA kernels (rows not whole 128-row tiles, K not whole 256s) is best given one per process: a
// kernel that faults there leaves the CUDA context unusable for the shapes after it.

#include "api/ops/linear.h"
#include "core/arena.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

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

namespace sinfer::ops::detail {
void launch_w8_consistent(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace sinfer::ops::detail

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::ops::detail;

namespace {

struct Route {
    const char* name;
    W8Launch launch;
};

constexpr Route kRoutes[] = {
    {"simt_r8_c4", launch_w8_simt_r8_c4},     {"simt_r8_c8", launch_w8_simt_r8_c8},
    {"mma_r32_c64", launch_w8_mma_r32_c64},   {"mma_r32_c128", launch_w8_mma_r32_c128},
    {"mma_r64_c96", launch_w8_mma_r64_c96},   {"mma_r64_c128", launch_w8_mma_r64_c128},
    {"mma_r128_c64", launch_w8_mma_r128_c64}, {"consistent", launch_w8_consistent},
};

std::vector<int> widths() {
    std::vector<int> out;
    if (const char* env = std::getenv("SUROGATE_W8_BENCH_T"); env != nullptr && *env != '\0') {
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
    return {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048};
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

/// Largest |a - b| over the largest |b|: the launchers differ only in accumulation order, so
/// anything past a few BF16 ulps of the output's scale is a wrong result, not rounding.
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
    const quantized_weight::PackedWeight host = sinfer::test::linear::make_w8g32_f16s_weight(n, k, 17U);
    p.payload = DeviceBuffer(host.payload.size());
    p.payload.copy_from_host(host.payload.data(), p.payload.bytes);
    p.weight = host.device_weight(p.payload.p);
    return p;
}

DeviceBuffer make_activation(std::int32_t k, std::int32_t t) {
    std::vector<float> values(static_cast<std::size_t>(k) * t);
    fill_uniform(values, 29U + static_cast<std::uint32_t>(t), -2.0F, 2.0F);
    return to_device_bf16(values);
}

bool run(const Route& route, const Tensor& x, const Weight& w, Tensor& out) {
    try {
        route.launch(x, w, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), route.name);
        return true;
    } catch (const std::exception&) {
        (void)cudaGetLastError();
        return false;
    }
}

/// Every route at `t` against SIMT r8_c4. Returns the number of routes that ran and disagreed.
int check(const Problem& p, int t, std::vector<bool>* usable) {
    DeviceBuffer x_buffer = make_activation(p.k, t);
    DeviceBuffer reference_buffer(static_cast<std::size_t>(p.n) * t * 2);
    DeviceBuffer out_buffer(static_cast<std::size_t>(p.n) * t * 2);
    const Tensor x(x_buffer.p, DType::BF16, {p.k, t});
    Tensor reference(reference_buffer.p, DType::BF16, {p.n, t});
    Tensor out(out_buffer.p, DType::BF16, {p.n, t});
    if (!run(kRoutes[0], x, p.weight, reference)) {
        std::printf("n=%6d k=%6d T=%4d  simt_r8_c4 declined the shape\n", p.n, p.k, t);
        return 1;
    }
    const std::vector<float> expected = read_bf16(reference.data, static_cast<std::size_t>(p.n) * t);
    int failures = 0;
    for (std::size_t r = 1; r < std::size(kRoutes); ++r) {
        cuda_check(cudaMemset(out.data, 0xff, static_cast<std::size_t>(p.n) * t * 2), "poison");
        if (!run(kRoutes[r], x, p.weight, out)) {
            if (usable != nullptr) { (*usable)[r] = false; }
            std::printf("n=%6d k=%6d T=%4d  %-13s declines\n", p.n, p.k, t, kRoutes[r].name);
            continue;
        }
        const double error =
            relative_error(read_bf16(out.data, static_cast<std::size_t>(p.n) * t), expected);
        const bool bad = !(error <= 2e-2);
        if (bad) {
            ++failures;
            if (usable != nullptr) { (*usable)[r] = false; }
        }
        std::printf("n=%6d k=%6d T=%4d  %-13s relative error %.2e%s\n", p.n, p.k, t,
                    kRoutes[r].name, error, bad ? "  WRONG" : "");
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
    const double bytes = static_cast<double>(p.n) * p.k * (1.0 + 2.0 / 32);
    std::vector<bool> usable(std::size(kRoutes), true);
    int failures = 0;
    for (const int t : {1, 8, 32, 128}) { failures += check(p, t, &usable); }
    if (failures != 0) { std::printf("n=%6d k=%6d  %d route checks WRONG\n", p.n, p.k, failures); }
    for (const int t : widths()) {
        DeviceBuffer x_buffer = make_activation(p.k, t);
        DeviceBuffer out_buffer(static_cast<std::size_t>(p.n) * t * 2);
        const Tensor x(x_buffer.p, DType::BF16, {p.k, t});
        Tensor out(out_buffer.p, DType::BF16, {p.n, t});
        double best = INFINITY;
        const char* best_name = "";
        for (std::size_t r = 0; r < std::size(kRoutes); ++r) {
            if (!usable[r] || !run(kRoutes[r], x, p.weight, out)) { continue; }
            const double us = time_us([&] { kRoutes[r].launch(x, p.weight, out, nullptr); }, bytes);
            std::printf("n=%6d k=%6d T=%4d  %-13s %9.1f us  %6.1f GB/s\n", p.n, p.k, t,
                        kRoutes[r].name, us, bytes / us * 1e-3);
            if (us < best) {
                best      = us;
                best_name = kRoutes[r].name;
            }
        }
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            QType::W8G32_F16S, p.n, p.k, ops::LinearPolicy::A16Only, t, t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        const double dispatch = time_us(
            [&] { ops::linear(x, p.weight, out, ops::LinearPolicy::A16Only, workspace, nullptr); },
            bytes);
        std::printf("n=%6d k=%6d T=%4d  %-13s %9.1f us  %6.1f GB/s  (best %s, %.2fx)\n", p.n, p.k,
                    t, "dispatch", dispatch, bytes / dispatch * 1e-3, best_name, dispatch / best);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        if (argc > 1 && std::string(argv[1]) == "bench") {
            for (int i = 2; i + 1 < argc; i += 2) {
                bench(make_problem(std::atoi(argv[i]), std::atoi(argv[i + 1])));
            }
            return 0;
        }
        // Gemma 3 4B's attention output and feed-forward down projection: shapes every route
        // takes, so each must agree with SIMT here.
        int failures = 0;
        for (const auto& [n, k] : {std::pair{2560, 2048}, std::pair{2560, 10240}}) {
            const Problem p = make_problem(n, k);
            for (const int t : {1, 8, 32, 128}) { failures += check(p, t, nullptr); }
        }
        std::printf("%s W8 routes against SIMT (%d wrong)\n", failures == 0 ? "OK" : "FAIL",
                    failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sinfer_w8_route_bench: %s\n", error.what());
        return 1;
    }
}
