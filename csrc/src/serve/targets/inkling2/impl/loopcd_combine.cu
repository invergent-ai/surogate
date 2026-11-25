// LoopCD-Logits combine (loopcd.h): one block per token over the softcapped head columns (z, z_ref).
// Adaptive strength: w = w_max (1 - (p1 - p2)) with p = softmax(z) over the token domain; p1 - p2 =
// (1 - exp(z2 - z1)) / sum_v exp(z_v - z1), z1 >= z2 the two largest logits (a tie gives p1 - p2 = 0).

#include "targets/inkling2/impl/loopcd.h"

#include "core/device.h"

#include <cfloat>
#include <cstdint>

namespace sinfer::targets::inkling2::detail {
namespace {

constexpr int kThreads = 512;

struct Top2 {
    float a = -FLT_MAX; ///< largest
    float b = -FLT_MAX; ///< second largest (equal to a on a tie)
};

__device__ __forceinline__ Top2 merge(Top2 x, Top2 y) {
    Top2 out;
    if (x.a >= y.a) {
        out.a = x.a;
        out.b = fmaxf(x.b, y.a);
    } else {
        out.a = y.a;
        out.b = fmaxf(y.b, x.a);
    }
    return out;
}

__global__ void loopcd_combine_kernel(const float* __restrict__ pairs, int rows, int domain, int adaptive,
                                      float strength, float* __restrict__ logits, int ldo) {
    const int t          = blockIdx.x;
    const float* z       = pairs + static_cast<std::int64_t>(2 * t) * rows;
    const float* z_ref   = z + rows;
    float* out           = logits + static_cast<std::int64_t>(t) * ldo;
    __shared__ Top2 s_top[kThreads];
    __shared__ float s_sum[kThreads];
    __shared__ float s_w;

    float w = strength;
    if (adaptive) {
        Top2 mine;
        for (int r = threadIdx.x; r < domain; r += kThreads) {
            const float v = z[r];
            if (v > mine.a) {
                mine.b = mine.a;
                mine.a = v;
            } else if (v > mine.b) {
                mine.b = v;
            }
        }
        s_top[threadIdx.x] = mine;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) { s_top[threadIdx.x] = merge(s_top[threadIdx.x], s_top[threadIdx.x + stride]); }
            __syncthreads();
        }
        const Top2 top = s_top[0];
        float sum      = 0.0F;
        for (int r = threadIdx.x; r < domain; r += kThreads) { sum += expf(z[r] - top.a); }
        s_sum[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) { s_sum[threadIdx.x] += s_sum[threadIdx.x + stride]; }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float margin = (1.0F - expf(top.b - top.a)) / s_sum[0];
            s_w                = strength * (1.0F - margin);
        }
        __syncthreads();
        w = s_w;
    }
    for (int r = threadIdx.x; r < rows; r += kThreads) {
        const float v = z[r];
        out[r]        = r < domain ? v + w * (v - z_ref[r]) : v;
    }
}

} // namespace

void loopcd_combine(const float* pairs, std::int32_t rows, std::int32_t domain, std::int32_t n, bool adaptive,
                    float strength, float* logits, std::int32_t ldo, cudaStream_t stream) {
    if (n <= 0) { return; }
    loopcd_combine_kernel<<<n, kThreads, 0, stream>>>(pairs, rows, domain, adaptive ? 1 : 0, strength, logits, ldo);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::targets::inkling2::detail
