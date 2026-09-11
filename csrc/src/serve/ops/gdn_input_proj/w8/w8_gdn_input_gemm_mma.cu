#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

namespace sinfer::ops::detail {
namespace {

template <int QkvRows, int ZRows>
void launch_wide(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    // Match the eight-way K reduction used by small batches. A mixed round
    // must not change a prompt's projected values when decode columns join it.
    // 32 columns keep the shared tiles within the SM89 static-memory limit.
    constexpr int columns = 32;
    for_each_token_slice(x.ne[1], columns, [&](int begin, int count) {
        const Tensor input = x.slice(1, begin, count);
        Tensor first = qkv.slice(1, begin, count), second = z.slice(1, begin, count);
        const W8SplitOutput2<QkvRows, ZRows> output{
            static_cast<__nv_bfloat16*>(first.data), static_cast<__nv_bfloat16*>(second.data)};
        w8_rowsplit_medium_t_splitk_kernel<0, columns, 8, 1, 1>
            <<<dim3((QkvRows + ZRows) / 16, (count + columns - 1) / columns), 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output, count, weight.k);
    });
}

} // namespace

void w8_gdn_input_wide_splitk_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     cudaStream_t stream) {
    if (weight.n == 8192) { launch_wide<6144, 2048>(x, weight, qkv, z, stream); }
    else { launch_wide<8192, 4096>(x, weight, qkv, z, stream); }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
