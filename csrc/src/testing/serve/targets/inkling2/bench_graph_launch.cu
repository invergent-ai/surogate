// Why is the inkling2 decode CUDA graph slower than eager? An nsys profile: cudaGraphLaunch
// takes ~0.81 ms of host time for the 745-node decode graph and the GPU idles until it returns. This bench captures
// straight chains of N kernels (each spinning ~`spin_us` like a decode kernel) in several shapes and reports the host
// time of the launch call(s) and the wall time to completion:
//   plain   : <<<>>> launches;          pdl     : programmatic-serialization launches (the engine's pdl::launch_dependent)
//   pdl+mem : pdl plus 2 D2D memcpy + 1 memset nodes (as the decode graph has);  plain+mem
//   pdl/k   : the pdl chain split into k graphs launched back to back (k = 4, 12, 48)
// Usage: sinfer_inkling_graph_launch_bench [N=745] [spin_us=8] [iterations=50]

#include "core/pdl.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sinfer;

namespace {

#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);    \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

__global__ void spin_kernel(float* p, int i, long long cycles) {
    pdl::wait_for_dependencies();
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) { p[i & 1023] += 1.0F; }
}

struct Shape {
    const char* name;
    bool pdl;
    bool mem;
    int pieces;
};

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void record(cudaStream_t s, const Shape& sh, int first, int count, int n, float* buf, float* buf2, long long cycles) {
    for (int i = first; i < first + count; ++i) {
        if (sh.mem && (i == 0 || i == n / 2)) {
            CK(cudaMemcpyAsync(buf2, buf, 4096, cudaMemcpyDeviceToDevice, s));
            if (i == 0) { CK(cudaMemsetAsync(buf2 + 2048, 0, 256, s)); }
        }
        if (sh.pdl) {
            CK(pdl::launch_dependent({dim3(1), dim3(32), 0, s}, spin_kernel, buf, i, cycles));
        } else {
            spin_kernel<<<1, 32, 0, s>>>(buf, i, cycles);
            CK(cudaGetLastError());
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const int n          = argc > 1 ? std::atoi(argv[1]) : 745;
    const double spin_us = argc > 2 ? std::atof(argv[2]) : 8.0;
    const int iters      = argc > 3 ? std::atoi(argv[3]) : 50;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    int dev = 0, clock_khz = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
    const long long cycles = static_cast<long long>(spin_us * clock_khz / 1000.0);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float *buf = nullptr, *buf2 = nullptr;
    CK(cudaMalloc(&buf, 1 << 16));
    CK(cudaMalloc(&buf2, 1 << 16));
    std::printf("N=%d kernels, spin %.1f us (%lld cycles at %d kHz), %d iterations, PDL %s\n", n, spin_us, cycles,
                clock_khz, iters, pdl::pdl_supported() ? "on" : "off");
    // eager reference
    for (const bool p : {false, true}) {
        const Shape sh{p ? "eager pdl" : "eager plain", p, false, 1};
        record(s, sh, 0, n, n, buf, buf2, cycles);
        CK(cudaStreamSynchronize(s));
        double host = 0, wall = 0;
        for (int it = 0; it < iters; ++it) {
            const double t0 = now_us();
            record(s, sh, 0, n, n, buf, buf2, cycles);
            const double t1 = now_us();
            CK(cudaStreamSynchronize(s));
            host += t1 - t0;
            wall += now_us() - t0;
        }
        std::printf("%-12s launch host %8.1f us   wall %8.1f us\n", sh.name, host / iters, wall / iters);
    }
    const Shape shapes[] = {{"plain", false, false, 1}, {"pdl", true, false, 1}, {"plain+mem", false, true, 1},
                            {"pdl+mem", true, true, 1},  {"pdl/4", true, false, 4}, {"pdl/12", true, false, 12},
                            {"pdl/48", true, false, 48}, {"pdl+mem/12", true, true, 12}};
    for (const Shape& sh : shapes) {
        std::vector<cudaGraphExec_t> execs;
        const int per = (n + sh.pieces - 1) / sh.pieces;
        for (int first = 0; first < n; first += per) {
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            record(s, sh, first, std::min(per, n - first), n, buf, buf2, cycles);
            CK(cudaStreamEndCapture(s, &g));
            cudaGraphExec_t e;
            CK(cudaGraphInstantiate(&e, g, 0));
            CK(cudaGraphUpload(e, s));
            CK(cudaGraphDestroy(g));
            execs.push_back(e);
        }
        for (int w = 0; w < 5; ++w) {
            for (auto e : execs) { CK(cudaGraphLaunch(e, s)); }
        }
        CK(cudaStreamSynchronize(s));
        double host = 0, wall = 0, first_launch = 0;
        for (int it = 0; it < iters; ++it) {
            const double t0 = now_us();
            for (std::size_t k = 0; k < execs.size(); ++k) {
                CK(cudaGraphLaunch(execs[k], s));
                if (k == 0) { first_launch += now_us() - t0; }
            }
            const double t1 = now_us();
            CK(cudaStreamSynchronize(s));
            host += t1 - t0;
            wall += now_us() - t0;
        }
        std::printf("%-12s launch host %8.1f us (first piece %7.1f)   wall %8.1f us   graphs %zu\n", sh.name,
                    host / iters, first_launch / iters, wall / iters, execs.size());
        for (auto e : execs) { CK(cudaGraphExecDestroy(e)); }
    }
    CK(cudaFree(buf));
    CK(cudaFree(buf2));
    return 0;
}
