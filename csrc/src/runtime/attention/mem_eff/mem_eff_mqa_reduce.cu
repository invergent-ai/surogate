// MQA backward helper: the cutlass backward kernel writes dK, dV at
// Hq head positions (one per Q head) because it doesn't model the
// MQA head-sharing. For MQA (Hkv=1), every Q head contributes to the
// same K/V head; we accumulate the Hq partial gradients into a single
// Hkv slot here.
//
// Input:  grad_k_partial, grad_v_partial — shape [total_tokens, Hq, Hs]
//         packed bf16 (one slot per Q head).
// Output: grad_k_out, grad_v_out — pointers into the interleaved d_qkv
//         buffer's K/V sections, layout [total_tokens, Hkv(=1), Hs]
//         with strideM = (Hq + 2*Hkv) * Hs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace surogate {
namespace mem_eff {

namespace {

__global__ void mqa_reduce_kv_kernel(const nv_bfloat16* __restrict__ dk_partial,
                                     const nv_bfloat16* __restrict__ dv_partial,
                                     nv_bfloat16* __restrict__ dk_out,
                                     nv_bfloat16* __restrict__ dv_out,
                                     int total_tokens,
                                     int Hq,
                                     int Hs,
                                     int partial_strideM,
                                     int partial_strideH,
                                     int out_strideM) {
    const int m = blockIdx.x;
    const int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (m >= total_tokens || d >= Hs) {
        return;
    }

    const nv_bfloat16* dk_base = dk_partial + m * partial_strideM;
    const nv_bfloat16* dv_base = dv_partial + m * partial_strideM;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (int h = 0; h < Hq; ++h) {
        k_sum += __bfloat162float(dk_base[h * partial_strideH + d]);
        v_sum += __bfloat162float(dv_base[h * partial_strideH + d]);
    }
    dk_out[m * out_strideM + d] = __float2bfloat16(k_sum);
    dv_out[m * out_strideM + d] = __float2bfloat16(v_sum);
}

}  // namespace

void mqa_reduce_kv_bf16(const void* dk_partial,
                        const void* dv_partial,
                        void* dk_out,
                        void* dv_out,
                        int total_tokens,
                        int Hq,
                        int Hs,
                        int partial_strideM,
                        int partial_strideH,
                        int out_strideM,
                        cudaStream_t stream) {
    const int tpb = 128;
    const int dim_blocks = (Hs + tpb - 1) / tpb;
    dim3 grid(total_tokens, dim_blocks);
    dim3 block(tpb);
    mqa_reduce_kv_kernel<<<grid, block, 0, stream>>>(static_cast<const nv_bfloat16*>(dk_partial),
                                                     static_cast<const nv_bfloat16*>(dv_partial),
                                                     static_cast<nv_bfloat16*>(dk_out),
                                                     static_cast<nv_bfloat16*>(dv_out),
                                                     total_tokens,
                                                     Hq,
                                                     Hs,
                                                     partial_strideM,
                                                     partial_strideH,
                                                     out_strideM);
}

}  // namespace mem_eff
}  // namespace surogate
