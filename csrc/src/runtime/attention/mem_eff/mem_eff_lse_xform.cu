// Scatter / gather between two LSE layouts for the mem-eff attention
// backend:
//
//   Kernel layout (what the cutlass fwd writes, bwd reads):
//     lse_kernel[num_docs, num_heads, lse_dim]  (fp32, contiguous)
//     lse_dim = ceil(max_doc_seqlen, 8)
//
//   Runtime layout (DSL's [B, Hq, T]):
//     lse_runtime[B, num_heads, T]  (fp32, contiguous)
//     Global token position gp = cu_seqlens[i] + q_in_doc sits at
//     lse_runtime[gp/T, head, gp%T].
//
// The scatter path runs after the fwd kernel; the gather path runs
// before the bwd kernel so the kernel can consume saved LSE values
// without us having to change the runtime's DSL-declared LSE shape.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace surogate {
namespace mem_eff {

namespace {

__global__ void lse_scatter_kernel(const float* __restrict__ lse_kernel,
                                   float* __restrict__ lse_runtime,
                                   const int32_t* __restrict__ cu_seqlens,
                                   int num_heads,
                                   int T,
                                   int lse_dim) {
    const int i = blockIdx.z;  // doc index
    const int h = blockIdx.y;  // head index
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t start = cu_seqlens[i];
    const int32_t end = cu_seqlens[i + 1];
    const int doc_len = end - start;
    if (q >= doc_len) {
        return;
    }
    const long src = static_cast<long>(i) * num_heads * lse_dim + static_cast<long>(h) * lse_dim + q;
    const int gp = start + q;
    const int b_row = gp / T;
    const int pos_in_row = gp % T;
    const long dst = static_cast<long>(b_row) * num_heads * T + static_cast<long>(h) * T + pos_in_row;
    lse_runtime[dst] = lse_kernel[src];
}

__global__ void lse_gather_kernel(float* __restrict__ lse_kernel,
                                  const float* __restrict__ lse_runtime,
                                  const int32_t* __restrict__ cu_seqlens,
                                  int num_heads,
                                  int T,
                                  int lse_dim) {
    const int i = blockIdx.z;
    const int h = blockIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t start = cu_seqlens[i];
    const int32_t end = cu_seqlens[i + 1];
    const int doc_len = end - start;
    if (q >= doc_len) {
        return;
    }
    const long dst = static_cast<long>(i) * num_heads * lse_dim + static_cast<long>(h) * lse_dim + q;
    const int gp = start + q;
    const int b_row = gp / T;
    const int pos_in_row = gp % T;
    const long src = static_cast<long>(b_row) * num_heads * T + static_cast<long>(h) * T + pos_in_row;
    lse_kernel[dst] = lse_runtime[src];
}

}  // namespace

void lse_scatter_kernel_to_runtime(const float* lse_kernel,
                                   float* lse_runtime,
                                   const int32_t* cu_seqlens,
                                   int num_docs,
                                   int num_heads,
                                   int max_doc_seqlen,
                                   int lse_dim,
                                   int T,
                                   cudaStream_t stream) {
    const int tpb = 128;
    dim3 grid((max_doc_seqlen + tpb - 1) / tpb, num_heads, num_docs);
    dim3 block(tpb);
    lse_scatter_kernel<<<grid, block, 0, stream>>>(lse_kernel, lse_runtime, cu_seqlens, num_heads, T, lse_dim);
}

void lse_gather_runtime_to_kernel(float* lse_kernel,
                                  const float* lse_runtime,
                                  const int32_t* cu_seqlens,
                                  int num_docs,
                                  int num_heads,
                                  int max_doc_seqlen,
                                  int lse_dim,
                                  int T,
                                  cudaStream_t stream) {
    const int tpb = 128;
    dim3 grid((max_doc_seqlen + tpb - 1) / tpb, num_heads, num_docs);
    dim3 block(tpb);
    lse_gather_kernel<<<grid, block, 0, stream>>>(lse_kernel, lse_runtime, cu_seqlens, num_heads, T, lse_dim);
}

namespace {

// Delta target layout (varlen): the backward kernel does
//   delta_ptr += q_start;           // cu_seqlens[doc]
//   batch_id = 0;
//   delta_ptr += head_id * delta_strideH;
// and then reads delta_ptr[q_in_block + thread]. So delta must be
// laid out as flat [num_heads, delta_strideH] with each head's
// contiguous region holding global packed positions
// [0, total_q) indexed directly by q_start + q_in_doc.
//
// This is DIFFERENT from LSE's layout (forward does NOT reset
// batch_id, so LSE stays [num_docs, num_heads, lse_dim]) — we must
// not reuse lse_gather_kernel here even though both buffers have
// the same element count.
__global__ void delta_flat_gather_kernel(float* __restrict__ delta_flat,
                                         const float* __restrict__ delta_runtime,
                                         const int32_t* __restrict__ cu_seqlens,
                                         int num_heads,
                                         int T,
                                         int delta_strideH) {
    const int i = blockIdx.z;  // doc index
    const int h = blockIdx.y;  // head index
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t start = cu_seqlens[i];
    const int32_t end = cu_seqlens[i + 1];
    const int doc_len = end - start;
    if (q >= doc_len) {
        return;
    }
    const int gp = start + q;
    const long dst = static_cast<long>(h) * delta_strideH + gp;
    const int b_row = gp / T;
    const int pos_in_row = gp % T;
    const long src = static_cast<long>(b_row) * num_heads * T + static_cast<long>(h) * T + pos_in_row;
    delta_flat[dst] = delta_runtime[src];
}

}  // namespace

void delta_gather_runtime_to_flat(float* delta_flat,
                                  const float* delta_runtime,
                                  const int32_t* cu_seqlens,
                                  int num_docs,
                                  int num_heads,
                                  int max_doc_seqlen,
                                  int delta_strideH,
                                  int T,
                                  cudaStream_t stream) {
    const int tpb = 128;
    dim3 grid((max_doc_seqlen + tpb - 1) / tpb, num_heads, num_docs);
    dim3 block(tpb);
    delta_flat_gather_kernel<<<grid, block, 0, stream>>>(delta_flat,
                                                         delta_runtime,
                                                         cu_seqlens,
                                                         num_heads,
                                                         T,
                                                         delta_strideH);
}

}  // namespace mem_eff
}  // namespace surogate
