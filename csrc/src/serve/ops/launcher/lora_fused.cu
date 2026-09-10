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
        pair.out_stride = outs[p]->nb[1] / 2;
        pair.rank           = banks[p]->rank;
        pair.row_begin      = rows;
        pair.gain = banks[p]->gain;
        pair.bias = banks[p]->bias;
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

namespace sinfer::ops {
namespace {
__global__ void embedding_adapter_kernel(const int* tokens, LoraBank bank, const int* slots,
                                          const int* uniform, __nv_bfloat16* out, int count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (row >= bank.n || token >= count) { return; }
    const int slot = slots ? slots[token] : *uniform;
    const int id = tokens[token];
    if (slot < 0 || id < 0 || id >= bank.k) { return; }
    const auto* a = static_cast<const __nv_bfloat16*>(bank.a) + slot * bank.a_stride;
    const auto* b = static_cast<const __nv_bfloat16*>(bank.b) + slot * bank.b_stride + row * bank.rank;
    float value = __bfloat162float(out[token * bank.n + row]);
    for (int r = 0; r < bank.rank; ++r) { value = fmaf(__bfloat162float(a[r * bank.k + id]), __bfloat162float(b[r]), value); }
    const auto index = static_cast<std::int64_t>(slot) * bank.n + row;
    out[token * bank.n + row] = __float2bfloat16(value * (bank.gain ? 1.0F + bank.gain[index] : 1.0F) + (bank.bias ? bank.bias[index] : 0.0F));
}
__global__ void shift_adapter_kernel(LoraBank bank, const int* slots, const int* uniform,
                                      __nv_bfloat16* out, int count, std::int64_t stride) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (row >= bank.n || token >= count) { return; }
    const int slot = slots ? slots[token] : *uniform;
    if (slot < 0) { return; }
    auto* cell = out + token * stride + row;
    *cell = __float2bfloat16(__bfloat162float(*cell) + bank.bias[static_cast<std::int64_t>(slot) * bank.n + row]);
}
}
void lora_embedding(const Tensor& token_ids, const LoraBank& bank, const Tensor& slots,
                     const std::int32_t* uniform, Tensor& out, cudaStream_t stream) {
    embedding_adapter_kernel<<<dim3((bank.n + 127) / 128, out.ne[1]), 128, 0, stream>>>(
        static_cast<const int*>(token_ids.data), bank, static_cast<const int*>(slots.data), uniform,
        static_cast<__nv_bfloat16*>(out.data), out.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
void lora_shift(const LoraBank& bank, const Tensor& slots, const std::int32_t* uniform,
                Tensor& out, cudaStream_t stream) {
    shift_adapter_kernel<<<dim3((bank.n + 127) / 128, out.ne[1]), 128, 0, stream>>>(bank,
        static_cast<const int*>(slots.data), uniform, static_cast<__nv_bfloat16*>(out.data),
        out.ne[1], out.nb[1] / 2);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::ops

#include "api/ops/lora_router.h"
#include <stdexcept>
#include "ops/sparse_moe/sparse_moe_route.cuh"
namespace sinfer::ops {
namespace {
__global__ void router_bias_adapter_kernel(const float* scores, int score_stride, int* ids, float* alpha,
    const float* base_bias, const float* expert_scale, int experts, int topk, float routed_scale,
    const LoraBank* banks, const int* slots, int slot_stride) {
    const int token = blockIdx.x, lane = threadIdx.x;
    const int slot = slots[token * slot_stride];
    if (slot < 0 || !banks[1].bias) { return; }
    scores += token * score_stride; ids += token * topk; alpha += token * topk;
    const float* delta = banks[1].bias + static_cast<std::int64_t>(slot) * experts;
    __shared__ float ranked[512];
    for (int i = lane; i < experts; i += 32) { ranked[i] = sigmoid(scores[i]) + base_bias[i] + delta[i]; }
    __syncwarp();
    for (int k = 0; k < topk; ++k) {
        detail::SparseMoeRankedValue best{-CUDART_INF_F, 0x7fffffff, lane};
        for (int i = lane; i < experts; i += 32) {
            detail::SparseMoeRankedValue value{ranked[i], i, lane};
            if (detail::sparse_moe_ranked_better(value, best)) { best = value; }
        }
        best = detail::sparse_moe_warp_best(best);
        if (lane == 0) { ids[k] = best.id; ranked[best.id] = -CUDART_INF_F; }
        __syncwarp();
    }
    float value = lane < topk ? sigmoid(scores[ids[lane]]) : 0.0F;
    float denominator = warp_reduce_sum(value);
    denominator = __shfl_sync(0xffffffffU, denominator, 0);
    if (lane < topk) { alpha[lane] = routed_scale * value / denominator * (expert_scale ? expert_scale[ids[lane]] : 1.0F); }
}
}
void lora_router_bias(const Tensor& scores, const Tensor& ids, const Tensor& alpha,
                       const float* base_bias, const float* expert_scale,
                       const SparseMoeGeometry& geometry, const LoraBank* banks,
                       const std::int32_t* slots, int slot_stride, cudaStream_t stream) {
    if (geometry.experts > 512) { throw std::invalid_argument("adapter router exceeds supported expert count"); }
    router_bias_adapter_kernel<<<scores.ne[1], 32, 0, stream>>>(static_cast<const float*>(scores.data),
        scores.ne[0], static_cast<int*>(ids.data), static_cast<float*>(alpha.data), base_bias,
        expert_scale, geometry.experts, geometry.experts_per_token, geometry.routed_scale, banks, slots, slot_stride);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::ops

#include "api/ops/lora_replacement.h"
namespace sinfer::ops {
namespace {
__global__ void replacement_linear_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
    const void* const* weights, const int* slots, const int* uniform, int n, int k) {
    const int token = blockIdx.y, lane = threadIdx.x & 31;
    const int slot = slots ? slots[token] : *uniform;
    if (slot < 0 || !weights[slot]) { return; }
    const auto* weight = static_cast<const __nv_bfloat16*>(weights[slot]);
    for (int row = blockIdx.x * 4 + threadIdx.x / 32; row < n; row += gridDim.x * 4) {
        float value = 0;
        for (int col = lane; col < k; col += 32) {
            value = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(row) * k + col]),
                         __bfloat162float(x[static_cast<std::int64_t>(token) * k + col]), value);
        }
        value = warp_reduce_sum(value);
        if (lane == 0) { out[static_cast<std::int64_t>(token) * n + row] = __float2bfloat16(value); }
    }
}
__global__ void replacement_embedding_kernel(const int* ids, __nv_bfloat16* out,
    const void* const* weights, const int* slots, const int* uniform, int hidden) {
    const int token = blockIdx.y, slot = slots ? slots[token] : *uniform;
    if (slot < 0 || !weights[slot]) { return; }
    const auto* table = static_cast<const __nv_bfloat16*>(weights[slot]);
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < hidden; row += gridDim.x * blockDim.x) {
        out[static_cast<std::int64_t>(token) * hidden + row] = table[static_cast<std::int64_t>(ids[token]) * hidden + row];
    }
}
}
void lora_replace_linear(const Tensor& x, Tensor& out, const void* const* weights,
    const Tensor& slots, const std::int32_t* uniform, cudaStream_t stream) {
    replacement_linear_kernel<<<dim3(std::min(1024, (out.ne[0] + 3) / 4), out.ne[1]), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), weights,
        static_cast<const int*>(slots.data), uniform, out.ne[0], x.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
void lora_replace_embedding(const Tensor& ids, Tensor& out, const void* const* weights,
    const Tensor& slots, const std::int32_t* uniform, cudaStream_t stream) {
    replacement_embedding_kernel<<<dim3((out.ne[0] + 127) / 128, out.ne[1]), 128, 0, stream>>>(
        static_cast<const int*>(ids.data), static_cast<__nv_bfloat16*>(out.data), weights,
        static_cast<const int*>(slots.data), uniform, out.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace sinfer::ops
