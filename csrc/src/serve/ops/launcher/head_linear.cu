// sinfer::ops -- head_linear launcher: tile choice, grid and the launch. The only translation
// unit that includes this op's kernel header.
#include "ops/launcher/head_linear.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/head_linear.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace sinfer::ops::detail {
namespace {

template <int kTokens>
void launch(const Tensor& x, const Weight& w, std::int32_t heads, std::int32_t n, std::int32_t k,
            std::int32_t tokens, float out_scale, Tensor& out, cudaStream_t stream) {
    // Row strides of the two W8 planes, in elements: codes are padded to the artifact's K
    // alignment, and there is one FP16 scale per group of that padded row.
    const int code_stride  = w.padded_shape[1];
    const int scale_stride = code_stride / kHeadLinearGroup;
    const dim3 grid(static_cast<unsigned>((n + kHeadLinearWarps - 1) / kHeadLinearWarps),
                    static_cast<unsigned>(heads),
                    static_cast<unsigned>((tokens + kTokens - 1) / kTokens));
    const std::size_t shared = static_cast<std::size_t>(kTokens) * k * sizeof(__nv_bfloat16);
    head_linear_kernel<kTokens><<<grid, kHeadLinearThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::int8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), heads, n, k, code_stride, scale_stride, tokens,
        out_scale, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void head_linear_launch(const Tensor& x, const Weight& w, std::int32_t heads, std::int32_t n,
                        std::int32_t k, float out_scale, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // A decode round is one column and wants no staging it will not use; a prefill wants the
    // weight row read once per eight columns rather than once per column.
    if (tokens == 1) {
        launch<1>(x, w, heads, n, k, tokens, out_scale, out, stream);
    } else if (tokens <= 4) {
        launch<4>(x, w, heads, n, k, tokens, out_scale, out, stream);
    } else {
        launch<8>(x, w, heads, n, k, tokens, out_scale, out, stream);
    }
}

} // namespace sinfer::ops::detail
