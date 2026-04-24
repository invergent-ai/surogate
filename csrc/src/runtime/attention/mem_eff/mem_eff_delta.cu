// Precompute-delta kernel for the mem-efficient attention backward.
// delta[b, h, m] = sum_k ( out[b, m, h, k] * dout[b, m, h, k] )
// Output dtype is fp32, input dtype bf16.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace surogate {
namespace mem_eff {

namespace {

__global__ void delta_bf16_kernel(const nv_bfloat16* __restrict__ out,
                                  const nv_bfloat16* __restrict__ dout,
                                  float* __restrict__ delta,
                                  int head_dim_v,
                                  int num_heads,
                                  int num_queries,
                                  long o_strideB,
                                  long o_strideM,
                                  long o_strideH,
                                  long dO_strideB,
                                  long dO_strideM,
                                  long dO_strideH,
                                  long delta_strideB,
                                  long delta_strideH) {
    const int m = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    if (m >= num_queries) {
        return;
    }

    const nv_bfloat16* o = out + b * o_strideB + m * o_strideM + h * o_strideH;
    const nv_bfloat16* d = dout + b * dO_strideB + m * dO_strideM + h * dO_strideH;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim_v; i += blockDim.x) {
        sum += __bfloat162float(o[i]) * __bfloat162float(d[i]);
    }

    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        smem[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = lane < nwarps ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) {
            delta[b * delta_strideB + h * delta_strideH + m] = v;
        }
    }
}

}  // namespace

// Compute delta = sum(out * dout, dim=-1) in fp32. All pointers refer
// to contiguous-head_dim BF16 (out / dout) and FP32 (delta) buffers
// in the same layout the backward kernel expects.
void compute_delta_bf16(const void* out,
                        const void* dout,
                        float* delta,
                        int num_batches,
                        int num_heads,
                        int num_queries,
                        int head_dim_v,
                        long o_strideB,
                        long o_strideM,
                        long o_strideH,
                        long dO_strideB,
                        long dO_strideM,
                        long dO_strideH,
                        long delta_strideB,
                        long delta_strideH,
                        cudaStream_t stream) {
    dim3 grid(num_queries, num_heads, num_batches);
    dim3 block(128);
    delta_bf16_kernel<<<grid, block, 0, stream>>>(static_cast<const nv_bfloat16*>(out),
                                                  static_cast<const nv_bfloat16*>(dout),
                                                  delta,
                                                  head_dim_v,
                                                  num_heads,
                                                  num_queries,
                                                  o_strideB,
                                                  o_strideM,
                                                  o_strideH,
                                                  dO_strideB,
                                                  dO_strideM,
                                                  dO_strideH,
                                                  delta_strideB,
                                                  delta_strideH);
}

}  // namespace mem_eff
}  // namespace surogate
