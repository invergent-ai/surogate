#include "ops/linear/fp8/fp8_cublaslt.h"

#include "core/device.h"

namespace ninfer::ops::detail {
namespace {

// One thread per output element, rows fastest: coalesced over the staging and the output.
__global__ void fp8_cublaslt_finish_kernel(const float* __restrict__ staging,
                                           const __nv_bfloat16* __restrict__ weight_row_scales,
                                           std::int32_t row_begin, std::int32_t rows,
                                           const float* __restrict__ activation_scales,
                                           std::int32_t tokens, __nv_bfloat16* __restrict__ out,
                                           std::int32_t out_ld, bool accumulate) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    if (index >= total) { return; }
    const std::int32_t row   = static_cast<std::int32_t>(index % rows);
    const std::int32_t token = static_cast<std::int32_t>(index / rows);
    float value = staging[index] * __bfloat162float(weight_row_scales[row_begin + row]) *
                  activation_scales[token];
    __nv_bfloat16* destination = out + static_cast<std::int64_t>(token) * out_ld + row;
    if (accumulate) { value += __bfloat162float(*destination); }
    *destination = __float2bfloat16_rn(value);
}

} // namespace

void fp8_cublaslt_finish(const float* staging, const __nv_bfloat16* weight_row_scales,
                         std::int32_t row_begin, std::int32_t rows, const float* activation_scales,
                         std::int32_t tokens, __nv_bfloat16* out, std::int32_t out_ld,
                         bool accumulate, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * tokens;
    constexpr int kThreads   = 256;
    const dim3 grid(static_cast<unsigned>((total + kThreads - 1) / kThreads));
    fp8_cublaslt_finish_kernel<<<grid, kThreads, 0, stream>>>(staging, weight_row_scales, row_begin,
                                                             rows, activation_scales, tokens, out,
                                                             out_ld, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
