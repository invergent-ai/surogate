// The wo plan's OP GATE ('wo BF16 KERNEL'). What one wo GEMM (2304 x 3072, BF16)
// adds to a decode-like PDL graph, generic kernel vs narrow kernel, same binary:
//   graph S    = 48 x a 4 us spin kernel (stands for the attention / fold before wo; PDL, triggers its dependents at once);
//   graph G, N = 48 x (spin, then wo on the next of 8 weight copies = 113 MB > L2, so every call streams from DRAM).
// Cost per call = (t(graph) - t(S)) / 48, median of launches, arms interleaved S G N G N. Prints the per-call saving at
// T = 1 and T = 8 and the gate line: T = 1 saving >= 2.4 us. Exit 77 without a GPU.

#include "api/ops/inkling_mix.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

__global__ void spin_kernel(std::uint64_t ns) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    std::uint64_t t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    constexpr int kN = 2304, kK = 3072, kCopies = 8, kCalls = 48, kLaunches = 120;
    constexpr std::uint64_t kSpinNs = 4000;
    const std::size_t wbytes = static_cast<std::size_t>(kN) * kK * 2;
    std::vector<DeviceBuffer> w;
    for (int c = 0; c < kCopies; ++c) {
        w.emplace_back(wbytes);
        cuda_check(cudaMemset(w.back().p, 0x11 + c, wbytes), "fill");
    }
    cudaStream_t s = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0), "event");
    cuda_check(cudaEventCreate(&e1), "event");
    double gate_saving = 0.0;
    for (const int T : {1, 8}) {
        DeviceBuffer x(static_cast<std::size_t>(kK) * T * 2), out(static_cast<std::size_t>(kN) * T * 2);
        cuda_check(cudaMemset(x.p, 0x22, x.bytes), "x");
        const auto capture = [&](bool with_wo, bool narrow) {
            ops::inkling_gemm_inv_use_narrow(narrow);
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            cuda_check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
            for (int i = 0; i < kCalls; ++i) {
                CUDA_CHECK(pdl::launch_dependent({dim3(1), dim3(1), 0, s}, spin_kernel, kSpinNs));
                if (with_wo) {
                    ops::inkling_gemm_inv(w[i % kCopies].p, kN, kK, x.p, T, out.p, DType::BF16, kN, {}, nullptr, 0, s);
                }
            }
            cuda_check(cudaStreamEndCapture(s, &g), "end capture");
            cuda_check(cudaGraphInstantiate(&ge, g, 0), "instantiate");
            cudaGraphDestroy(g);
            return ge;
        };
        cudaGraphExec_t gs = capture(false, false), gg = capture(true, false), gn = capture(true, true);
        ops::inkling_gemm_inv_use_narrow(false);
        const auto timed = [&](cudaGraphExec_t ge) {
            std::vector<float> us;
            for (int i = 0; i < kLaunches + 8; ++i) {
                cuda_check(cudaEventRecord(e0, s), "record");
                cuda_check(cudaGraphLaunch(ge, s), "launch");
                cuda_check(cudaEventRecord(e1, s), "record");
                cuda_check(cudaEventSynchronize(e1), "sync");
                float ms = 0;
                cuda_check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                if (i >= 8) { us.push_back(ms * 1000.0F); }
            }
            return median(us);
        };
        const double base = timed(gs);
        double g_sum = 0, n_sum = 0;
        for (int rep = 0; rep < 2; ++rep) {
            const double tg = (timed(gg) - base) / kCalls, tn = (timed(gn) - base) / kCalls;
            std::printf("T=%d rep %d: generic %.2f us / call, narrow %.2f (saving %.2f)\n", T, rep, tg, tn, tg - tn);
            g_sum += tg;
            n_sum += tn;
        }
        const double saving = (g_sum - n_sum) / 2.0;
        std::printf("T=%d: spin-only graph %.1f us; generic %.2f, narrow %.2f us / call; saving %.2f us\n", T, base,
                    g_sum / 2.0, n_sum / 2.0, saving);
        if (T == 1) { gate_saving = saving; }
        cudaGraphExecDestroy(gs);
        cudaGraphExecDestroy(gg);
        cudaGraphExecDestroy(gn);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(s);
    std::printf("OP GATE (T = 1 saving >= 2.40 us per call): %.2f us: %s\n", gate_saving,
                gate_saving >= 2.40 ? "MET" : "NOT MET");
    return 0;
}
