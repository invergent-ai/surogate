#include "ops/launcher/dflash2.h"

#include "core/device.h"
#include "ops/kernel/dflash2.cuh"

namespace sinfer::ops::detail {

void dflash_conv_launch(const Tensor& input, const Tensor& dynamic, const Tensor& base,
                        std::int32_t stage, std::int32_t width, const Tensor* residual, Tensor& out,
                        std::int32_t groups, cudaStream_t stream) {
    const std::int32_t hidden  = input.ne[0];
    const std::int32_t columns = input.ne[1];
    const long long threads    = static_cast<long long>(hidden / 8) * columns;
    constexpr int kBlock       = 256;
    dflash_conv_kernel<<<static_cast<unsigned>((threads + kBlock - 1) / kBlock), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data), static_cast<const __nv_bfloat16*>(dynamic.data),
        static_cast<const __nv_bfloat16*>(base.data),
        residual != nullptr ? static_cast<const __nv_bfloat16*>(residual->data) : nullptr,
        static_cast<__nv_bfloat16*>(out.data), hidden, columns, groups, stage, width);
    CUDA_CHECK(cudaGetLastError());
}

void dflash_select_top_k_launch(const Tensor& logits, std::int32_t domain, Tensor& ids,
                                Tensor& values, cudaStream_t stream) {
    const std::int32_t k       = ids.ne[0];
    const std::int32_t columns = ids.ne[1];
    const auto* source         = static_cast<const __nv_bfloat16*>(logits.data);
    auto* id_out               = static_cast<int*>(ids.data);
    auto* value_out            = static_cast<float*>(values.data);
    if (k <= 16) {
        dflash_top_k_kernel<16><<<columns, 256, 0, stream>>>(source, logits.ne[0], domain, k,
                                                             id_out, value_out);
    } else {
        dflash_top_k_kernel<32><<<columns, 256, 0, stream>>>(source, logits.ne[0], domain, k,
                                                             id_out, value_out);
    }
    CUDA_CHECK(cudaGetLastError());
}

void dflash_select_walk_launch(const Tensor& anchors, const Tensor& ids, const Tensor& values,
                               const Tensor& hidden, const Tensor& predecessor,
                               const Tensor& successor, std::int32_t positions, Tensor& drafts,
                               cudaStream_t stream) {
    const std::int32_t k     = ids.ne[0];
    const std::int32_t lanes = ids.ne[1] / positions;
    dflash_walk_kernel<<<lanes, 32 * k, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data), static_cast<const std::int32_t*>(ids.data),
        static_cast<const float*>(values.data), static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(predecessor.data),
        static_cast<const __nv_bfloat16*>(successor.data), k, positions, hidden.ne[0],
        predecessor.ne[1], static_cast<std::int32_t*>(drafts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
