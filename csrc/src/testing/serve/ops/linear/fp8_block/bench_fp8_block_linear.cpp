// Block-FP8 linear micro-benchmark through the engine's own route (fp8_block::linear and
// linear_add): the activation quantisation plus whichever kernel this device runs for the round
// -- the decode GEMV, the narrow tensor-core kernel, a CUTLASS GEMM (Hopper, sm_12x) or the
// engine's tile -- timed from a CUDA graph of 50 launches over weight copies that keep L2 cold.
// Compare routes by running it under their switches: SUROGATE_SERVE_FP8_BLOCK_SM120=0 against
// the default, or SUROGATE_SERVE_FP8_BLOCK_SM120_TILE=p32, c32, c64 or c128.
//
//   sinfer_fp8_block_linear_bench [--per-row] [n k tokens ...]   (default: Qwen3-8B's four linears)
//
// --per-row times weights with one scale per row (the compressed-tensors per-channel kind) in
// place of the 128 x 128 block grid.

#include "core/arena.h"
#include "ops/linear/fp8_block/fp8_block.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace fb = sinfer::ops::detail::fp8_block;
using sinfer::DType;
using sinfer::QType;
using sinfer::QuantLayout;
using sinfer::Tensor;
using sinfer::Weight;
using sinfer::WorkspaceArena;

namespace {

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));        \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

struct Problem {
    int n, k, tokens;
};

void* upload(const void* host, std::size_t bytes) {
    void* p = nullptr;
    CHECK(cudaMalloc(&p, bytes));
    CHECK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice));
    return p;
}

bool g_per_row = false;

// A block-FP8 weight [n, k] in the artifact's layout: codes, then the scale grid 256-aligned.
Weight block_weight(int n, int k, const std::vector<std::uint8_t>& payload, std::size_t scale_off) {
    Weight w{};
    w.payload = upload(payload.data(), payload.size());
    w.payload_bytes = payload.size();
    w.qtype = g_per_row ? QType::FP8_E4M3FN_ROW_F32S : QType::FP8_E4M3FN_BLK128_F32S;
    w.layout = QuantLayout::Fp8Block128;
    w.group_size = 128;
    w.group = 128;
    w.ndim = 2;
    w.qdata = w.payload;
    w.scales = static_cast<const std::uint8_t*>(w.payload) + scale_off;
    w.scale_dtype = DType::FP32;
    w.scale_ne[0] = g_per_row ? k : 128;
    w.scale_ne[1] = g_per_row ? 1 : 128;
    w.n = n;
    w.k = k;
    w.shape[0] = n;
    w.shape[1] = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    return w;
}

void run(const Problem& p, std::mt19937& rng) {
    std::uniform_real_distribution<float> uw(-1.0f, 1.0f), us(0.5f, 2.0f), ux(-3.0f, 3.0f);
    const std::size_t codes = static_cast<std::size_t>(p.n) * p.k;
    const std::size_t scale_off = (codes + 255) / 256 * 256;
    const std::size_t cells = g_per_row ? static_cast<std::size_t>(p.n) : static_cast<std::size_t>(p.n / 128) * (p.k / 128);
    std::vector<std::uint8_t> payload(scale_off + cells * 4);
    for (std::size_t i = 0; i < codes; ++i) { payload[i] = __nv_fp8_e4m3(uw(rng)).__x; }
    for (std::size_t i = 0; i < cells; ++i) {
        const float v = us(rng) * 0.01f;
        std::memcpy(payload.data() + scale_off + 4 * i, &v, 4);
    }
    const int copies = std::max(2, static_cast<int>((256ull << 20) / codes) + 1);
    std::vector<Weight> w;
    for (int c = 0; c < copies; ++c) { w.push_back(block_weight(p.n, p.k, payload, scale_off)); }
    std::vector<__nv_bfloat16> hx(static_cast<std::size_t>(p.k) * p.tokens);
    for (auto& v : hx) { v = __float2bfloat16(ux(rng)); }
    auto* x = static_cast<__nv_bfloat16*>(upload(hx.data(), hx.size() * 2));
    __nv_bfloat16* out = nullptr;
    CHECK(cudaMalloc(&out, static_cast<std::size_t>(p.n) * p.tokens * 2));
    CHECK(cudaMemset(out, 0, static_cast<std::size_t>(p.n) * p.tokens * 2));
    Tensor xt(x, DType::BF16, {p.k, p.tokens});
    Tensor ot(out, DType::BF16, {p.n, p.tokens});
    WorkspaceArena workspace(std::max<std::size_t>(256, fb::linear_workspace_capacity_bytes(p.n, p.k, p.tokens)));
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (const bool residual : {false, true}) {
        const auto launch = [&](int c) {
            residual ? fb::linear_add(xt, w[c], ot, &workspace, stream) : fb::linear(xt, w[c], ot, &workspace, stream);
        };
        for (int i = 0; i < 10; ++i) { launch(i % copies); }
        CHECK(cudaStreamSynchronize(stream));
        const int iters = 50;
        cudaGraph_t g;
        cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        for (int i = 0; i < iters; ++i) { launch(i % copies); }
        CHECK(cudaStreamEndCapture(stream, &g));
        CHECK(cudaGraphInstantiate(&exec, g, 0));
        CHECK(cudaGraphLaunch(exec, stream));
        float ms = 0;
        CHECK(cudaEventRecord(start, stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaEventRecord(stop, stream));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(g));
        const double us = ms * 1000.0 / iters;
        const double tflops = 2.0 * p.n * p.k * p.tokens / (us * 1e-6) / 1e12;
        const double gbs = (static_cast<double>(codes) + 2.0 * p.tokens * (p.k + p.n)) / (us * 1e-6) / 1e9;
        std::printf("%s n=%6d k=%6d tokens=%5d residual=%d  %9.1f us  %6.1f TFLOPS  %6.1f GB/s\n",
                    g_per_row ? "row" : "blk", p.n, p.k, p.tokens, residual, us, tflops, gbs);
    }
    for (auto& c : w) { CHECK(cudaFree(const_cast<void*>(c.payload))); }
    CHECK(cudaFree(x));
    CHECK(cudaFree(out));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    std::vector<Problem> problems;
    int first = 1;
    if (argc > 1 && std::strcmp(argv[1], "--per-row") == 0) {
        g_per_row = true;
        first = 2;
    }
    for (int i = first; i + 2 < argc; i += 3) {
        problems.push_back({std::atoi(argv[i]), std::atoi(argv[i + 1]), std::atoi(argv[i + 2])});
    }
    if (problems.empty()) {
        const int shapes[][2] = {{6144, 4096}, {4096, 4096}, {24576, 4096}, {4096, 12288}};
        for (const int tokens : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
            for (const auto& s : shapes) { problems.push_back({s[0], s[1], tokens}); }
        }
    }
    std::mt19937 rng(7);
    for (const auto& p : problems) { run(p, rng); }
    return 0;
}
