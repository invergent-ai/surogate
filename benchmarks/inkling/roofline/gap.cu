// (4) Per-kernel gap of a chain of tiny dependent kernels: stream launches vs one CUDA graph, each without and with
// programmatic dependent launch (PDL). Kernel k reads buf[k] and writes buf[k + 1] = buf[k] + 1 (one thread of CTA
// 0; the other CTAs only occupy the grid), so the chain is a real data dependency: the final value proves the order
// (with PDL the consumer calls cudaGridDependencySynchronize() before the read, as csrc/src/serve/core/pdl.cuh).
// Reported: incremental ns per kernel = (T(N) - T(1)) / (N - 1), and the fixed cost T(1) of one launch / replay.
#include "common.cuh"

#include <cstring>

namespace rf {
namespace {

template <bool PDL> __global__ void chain_kernel(uint32_t* __restrict__ buf, int k) {
#if __CUDA_ARCH__ >= 900
    if (PDL) {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif
    if (blockIdx.x == 0 && threadIdx.x == 0) buf[k + 1] = buf[k] + 1u;
}

template <bool PDL> void launch_one(uint32_t* buf, int k, int grid, int block, cudaStream_t s) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim  = dim3(grid);
    cfg.blockDim = dim3(block);
    cfg.stream   = s;
    cudaLaunchAttribute attr{};
    attr.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    if (PDL) {
        cfg.attrs    = &attr;
        cfg.numAttrs = 1;
    }
    RF_CHECK(cudaLaunchKernelEx(&cfg, chain_kernel<PDL>, buf, k));
}

struct GapResult {
    double ns_total = 0;
    bool ok         = false;
};

template <bool PDL>
GapResult measure(bool graph, int n, int grid, int block, uint32_t* buf, cudaStream_t s) {
    auto chain = [&] {
        for (int k = 0; k < n; ++k) launch_one<PDL>(buf, k, grid, block, s);
    };
    cudaGraphExec_t exec = nullptr;
    if (graph) {
        cudaGraph_t g = nullptr;
        RF_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        chain();
        RF_CHECK(cudaStreamEndCapture(s, &g));
        RF_CHECK(cudaGraphInstantiate(&exec, g, 0));
        RF_CHECK(cudaGraphDestroy(g));
    }
    auto run = [&] {
        if (graph) {
            RF_CHECK(cudaGraphLaunch(exec, s));
        } else {
            chain();
        }
    };
    // correctness: buf[0] = 7, run once, expect buf[n] = 7 + n
    const uint32_t seed = 7;
    RF_CHECK(cudaMemsetAsync(buf, 0, (n + 1) * sizeof(uint32_t), s));
    RF_CHECK(cudaMemcpyAsync(buf, &seed, 4, cudaMemcpyHostToDevice, s));
    run();
    uint32_t got = 0;
    RF_CHECK(cudaMemcpyAsync(&got, buf + n, 4, cudaMemcpyDeviceToHost, s));
    RF_CHECK(cudaStreamSynchronize(s));
    const Timing t = time_it(s, run, 3);
    if (exec) RF_CHECK(cudaGraphExecDestroy(exec));
    return GapResult{t.best_ns, got == seed + static_cast<uint32_t>(n)};
}

template <bool PDL> void gap_case(bool graph, int n, int grid, int block, uint32_t* buf, cudaStream_t s) {
    const GapResult one  = measure<PDL>(graph, 1, grid, block, buf, s);
    const GapResult many = measure<PDL>(graph, n, grid, block, buf, s);
    const double per_k   = (many.ns_total - one.ns_total) / (n - 1);
    const char* mode     = graph ? (PDL ? "graph_pdl" : "graph") : (PDL ? "stream_pdl" : "stream");
    const bool ok        = one.ok && many.ok;
    emit("launch_gap", {kv("mode", mode), kv("grid", grid), kv("block", block), kv("chain", n),
                        kv("ns_per_kernel", per_k), kv("ns_chain_total", many.ns_total),
                        kv("ns_single", one.ns_total), kv("ok", ok)});
    note("  launch_gap %-10s grid=%4d block=%3d chain=%4d  %7.2f us/kernel  (single %7.2f us)  %s\n", mode, grid,
         block, n, per_k / 1e3, one.ns_total / 1e3, ok ? "ok" : "CHAIN VALUE WRONG");
}

} // namespace

void run_gap() {
    const cudaDeviceProp p = device_props();
    cudaStream_t s{};
    RF_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int n = 400; // ~ 8 kernels per visit x 48 visits + head / sampling
    uint32_t* buf = nullptr;
    RF_CHECK(cudaMalloc(&buf, (n + 2) * sizeof(uint32_t)));
    const int sms    = p.multiProcessorCount;
    const int grids[][2] = {{1, 32}, {sms, 128}, {4 * sms, 256}};
    for (const auto& g : grids) {
        gap_case<false>(false, n, g[0], g[1], buf, s);
        gap_case<true>(false, n, g[0], g[1], buf, s);
        gap_case<false>(true, n, g[0], g[1], buf, s);
        gap_case<true>(true, n, g[0], g[1], buf, s);
    }
    RF_CHECK(cudaFree(buf));
    RF_CHECK(cudaStreamDestroy(s));
}

} // namespace rf
