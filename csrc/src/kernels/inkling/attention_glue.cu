// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer kernels (lane g2): small glue kernels of the inkling_attention op wrapper -- the FP8-KV emulation
// (e4m3 round trip of K / V), segment offsets for uniform rows and for the window-bounded chunk key buffer.

#include <stdexcept>
#include <string>

#include "kernels/inkling/inkling.h"
#include "kernels/inkling/inkling_common.cuh"

namespace inkling {
namespace {

inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling ") + what + ": " + cudaGetErrorString(e));
}

__global__ void round_e4m3_kernel(bf16* y, const bf16* x, long n) {
    for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < n; i += static_cast<long>(gridDim.x) * blockDim.x)
        y[i] = __float2bfloat16_rn(e4m3_round(__bfloat162float(x[i])));
}

__global__ void fill_uniform_cu_kernel(int* cu, int num_segs, int step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= num_segs) cu[i] = i * step;
}

__global__ void near_cu_kernel(int* cu_near, const int* cu_k, int num_segs, int win_start, int base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_segs) return;
    const int g = cu_k[i] + win_start;
    cu_near[i] = (i == 0 ? max(g, base) : g) - base;
}

}  // namespace

void round_e4m3(bf16* y, const bf16* x, long n, cudaStream_t stream) {
    if (n <= 0) return;
    const long blocks = (n + 255) / 256;
    round_e4m3_kernel<<<static_cast<int>(blocks < 65535 ? blocks : 65535), 256, 0, stream>>>(y, x, n);
    check_launch("round_e4m3");
}

void fill_uniform_cu(int* cu, int num_segs, int step, cudaStream_t stream) {
    fill_uniform_cu_kernel<<<(num_segs + 1 + 127) / 128, 128, 0, stream>>>(cu, num_segs, step);
    check_launch("fill_uniform_cu");
}

void near_cu_from_window(int* cu_near, const int* cu_k, int num_segs, int win_start, int base, cudaStream_t stream) {
    near_cu_kernel<<<(num_segs + 1 + 127) / 128, 128, 0, stream>>>(cu_near, cu_k, num_segs, win_start, base);
    check_launch("near_cu_from_window");
}

}  // namespace inkling
