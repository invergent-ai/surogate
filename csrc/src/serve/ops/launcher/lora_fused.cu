#include "ops/launcher/lora_fused.h"

#include "core/device.h"
#include "ops/kernel/lora_fused.cuh"

namespace sinfer::ops::detail {

namespace {

LoraFusedParams fill_params(const Tensor& x, const LoraBank* const* banks, Tensor* const* outs,
                            std::int32_t pair_count, const Tensor& ids,
                            const std::int32_t* uniform) {
    LoraFusedParams params;
    params.x          = static_cast<const __nv_bfloat16*>(x.data);
    params.ids        = static_cast<const std::int32_t*>(ids.data);
    params.uniform    = uniform;
    params.pair_count = pair_count;
    params.k          = x.ne[0];
    params.tokens     = x.ne[1];
    std::int32_t rows = 0;
    for (std::int32_t p = 0; p < pair_count; ++p) {
        LoraFusedPair& pair = params.pairs[p];
        pair.a              = static_cast<const __nv_bfloat16*>(banks[p]->a);
        pair.b              = static_cast<const __nv_bfloat16*>(banks[p]->b);
        pair.out            = static_cast<__nv_bfloat16*>(outs[p]->data);
        pair.a_stride       = banks[p]->a_stride;
        pair.b_stride       = banks[p]->b_stride;
        pair.n              = banks[p]->n;
        pair.rank           = banks[p]->rank;
        pair.row_begin      = rows;
        rows += banks[p]->n;
    }
    params.total_rows = rows;
    return params;
}

} // namespace

void lora_fused_delta_launch(const Tensor& x, const LoraBank* const* banks, Tensor* const* outs,
                             std::int32_t pair_count, const Tensor& ids,
                             const std::int32_t* uniform, cudaStream_t stream) {
    const LoraFusedParams params = fill_params(x, banks, outs, pair_count, ids, uniform);
    const dim3 block(kLoraFusedThreads, 1, 1);
    const dim3 grid(
        static_cast<unsigned>((params.total_rows + kLoraFusedThreads - 1) / kLoraFusedThreads),
        static_cast<unsigned>(params.tokens), 1);
    lora_fused_delta_kernel<<<grid, block, 0, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}


void lora_split_delta_launch(const Tensor& x, const LoraBank* const* banks, Tensor* const* outs,
                             std::int32_t pair_count, const Tensor& ids,
                             const std::int32_t* uniform, Tensor& low, cudaStream_t stream) {
    const LoraFusedParams params = fill_params(x, banks, outs, pair_count, ids, uniform);
    std::int32_t total_rank      = 0;
    for (std::int32_t p = 0; p < pair_count; ++p) { total_rank += banks[p]->rank; }
    const dim3 block(kLoraFusedThreads, 1, 1);
    const dim3 shrink_grid(static_cast<unsigned>(total_rank),
                           static_cast<unsigned>(params.tokens), 1);
    lora_split_shrink_kernel<<<shrink_grid, block, 0, stream>>>(
        params, static_cast<__nv_bfloat16*>(low.data));
    CUDA_CHECK(cudaGetLastError());
    const dim3 expand_grid(
        static_cast<unsigned>((params.total_rows + kLoraFusedThreads - 1) / kLoraFusedThreads),
        static_cast<unsigned>(params.tokens), 1);
    lora_split_expand_kernel<<<expand_grid, block, 0, stream>>>(
        params, static_cast<const __nv_bfloat16*>(low.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
