// Hopper block-FP8 GEMM micro-benchmark: DeepGEMM's kernel (fp8_block_sm90_deepgemm.h) against the
// CUTLASS tiles (sm90_gemm with SUROGATE_SERVE_FP8_BLOCK_DEEPGEMM=0), per shape, token count and
// residual mode, timed from a CUDA graph of 200 launches (and eagerly). Weights rotate through
// enough copies to stay out of L2, as in a decode round.
//
//   sinfer_fp8_block_gemm_bench [n k tokens ...]   (default: Qwen3-8B's four linears)

#include "ops/linear/fp8_block/fp8_block_sm90_deepgemm.h"
#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace fb = sinfer::ops::detail::fp8_block;

namespace {

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

template <class T>
T* upload(const std::vector<T>& host) {
    T* p = nullptr;
    CHECK(cudaMalloc(&p, host.size() * sizeof(T)));
    CHECK(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

std::uint8_t e4m3(float v) {
    const __nv_fp8_e4m3 q(v);
    return *reinterpret_cast<const std::uint8_t*>(&q);
}

struct Problem {
    int n, k, tokens;
};

void run(const Problem& p, std::mt19937& rng) {
    std::uniform_real_distribution<float> uw(-1.0f, 1.0f), us(0.5f, 2.0f);
    const int copies = std::max(2, static_cast<int>((256ull << 20) / (static_cast<std::uint64_t>(p.n) * p.k)) + 1);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(p.n) * p.k);
    for (auto& c : codes) { c = e4m3(uw(rng)); }
    std::vector<float> wscales(static_cast<std::size_t>(p.n / 128) * (p.k / 128));
    for (auto& s : wscales) { s = us(rng) * 0.01f; }
    std::vector<const std::uint8_t*> w(copies);
    std::vector<const float*> ws(copies);
    for (int c = 0; c < copies; ++c) { w[c] = upload(codes), ws[c] = upload(wscales); }
    std::vector<std::uint8_t> acodes(static_cast<std::size_t>(p.tokens) * p.k);
    for (auto& c : acodes) { c = e4m3(uw(rng) * 4.0f); }
    std::vector<float> ascales(static_cast<std::size_t>(p.k / 128) * fb::sm90_scale_stride(p.tokens));
    for (auto& s : ascales) { s = us(rng) * 0.05f; }
    const auto* a  = upload(acodes);
    const auto* as = upload(ascales);
    std::vector<__nv_bfloat16> init(static_cast<std::size_t>(p.tokens) * p.n);
    for (auto& v : init) { v = __float2bfloat16(uw(rng)); }
    auto* out_dg = upload(init);
    auto* out_ct = upload(init);
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (const bool residual : {false, true}) {
        // One call of each for the comparison, from the same starting output.
        CHECK(cudaMemcpy(out_dg, init.data(), init.size() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(out_ct, init.data(), init.size() * 2, cudaMemcpyHostToDevice));
        const bool dg_ok = fb::sm90::deepgemm({a, as, w[0], ws[0], out_dg, residual, p.tokens, p.n, p.k, stream});
        const bool ct_ok = fb::sm90_gemm(a, as, w[0], ws[0], out_ct, residual, p.tokens, p.n, p.k, stream);
        CHECK(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> got_dg(init.size()), got_ct(init.size());
        CHECK(cudaMemcpy(got_dg.data(), out_dg, init.size() * 2, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(got_ct.data(), out_ct, init.size() * 2, cudaMemcpyDeviceToHost));
        double max_diff = 0, max_ref = 0;
        for (std::size_t i = 0; i < init.size(); ++i) {
            const double x = __bfloat162float(got_dg[i]), y = __bfloat162float(got_ct[i]);
            max_diff = std::max(max_diff, std::abs(x - y)), max_ref = std::max(max_ref, std::abs(y));
        }
        const auto launch = [&](int route, int c) {
            route == 0 ? (void)fb::sm90::deepgemm({a, as, w[c], ws[c], out_dg, residual, p.tokens, p.n, p.k, stream})
                       : (void)fb::sm90_gemm(a, as, w[c], ws[c], out_ct, residual, p.tokens, p.n, p.k, stream);
        };
        const int iters = 200;
        double eager[2] = {0, 0}, graph[2] = {0, 0};
        for (int route = 0; route < 2; ++route) {
            for (int i = 0; i < 20; ++i) { launch(route, i % copies); }
            float ms = 0;
            CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < iters; ++i) { launch(route, i % copies); }
            CHECK(cudaEventRecord(stop, stream));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            eager[route] = ms * 1000.0 / iters;
            // The engine replays decode rounds from CUDA graphs: the same launches, captured.
            cudaGraph_t g;
            cudaGraphExec_t exec;
            CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
            for (int i = 0; i < iters; ++i) { launch(route, i % copies); }
            CHECK(cudaStreamEndCapture(stream, &g));
            CHECK(cudaGraphInstantiate(&exec, g, 0));
            CHECK(cudaGraphLaunch(exec, stream));
            CHECK(cudaEventRecord(start, stream));
            CHECK(cudaGraphLaunch(exec, stream));
            CHECK(cudaEventRecord(stop, stream));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            graph[route] = ms * 1000.0 / iters;
            CHECK(cudaGraphExecDestroy(exec));
            CHECK(cudaGraphDestroy(g));
        }
        std::printf("n=%6d k=%6d tokens=%5d residual=%d  deepgemm %7.1f us  cutlass %7.1f us  ratio %.2f  "
                    "(graphs; eager %.1f / %.1f; ok %d/%d, max diff %.3g of %.3g)\n",
                    p.n, p.k, p.tokens, residual, graph[0], graph[1], graph[0] / graph[1], eager[0], eager[1],
                    dg_ok, ct_ok, max_diff, max_ref);
    }
    for (int c = 0; c < copies; ++c) {
        CHECK(cudaFree(const_cast<std::uint8_t*>(w[c])));
        CHECK(cudaFree(const_cast<float*>(ws[c])));
    }
    CHECK(cudaFree(const_cast<std::uint8_t*>(a)));
    CHECK(cudaFree(const_cast<float*>(as)));
    CHECK(cudaFree(out_dg));
    CHECK(cudaFree(out_ct));
    CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    setenv("SUROGATE_SERVE_FP8_BLOCK_DEEPGEMM", "0", 1); // sm90_gemm: the CUTLASS tiles only
    if (!fb::sm90_gemm_available()) {
        std::printf("no sm_90 device or kernel: skipped\n");
        return 77;
    }
    std::vector<Problem> problems;
    for (int i = 1; i + 2 < argc; i += 3) {
        problems.push_back({std::atoi(argv[i]), std::atoi(argv[i + 1]), std::atoi(argv[i + 2])});
    }
    if (problems.empty()) {
        const std::pair<int, int> shapes[] = {{6144, 4096}, {4096, 4096}, {24576, 4096}, {4096, 12288}};
        for (const int tokens : {64, 192, 576}) {
            for (const auto& [n, k] : shapes) { problems.push_back({n, k, tokens}); }
        }
    }
    std::mt19937 rng(7);
    for (const auto& p : problems) { run(p, rng); }
    return 0;
}
