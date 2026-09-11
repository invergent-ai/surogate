#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

namespace sinfer::ops::detail {
namespace {

template <int Hidden, int QkvRows, int ZRows>
void launch_wide(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 cudaStream_t stream) {
    // Match the eight-way K reduction used by small batches. A mixed round
    // must not change a prompt's projected values when decode columns join it.
    // Reuse each activation tile across 32 output rows. The shared arena
    // still fits the SM89 static-memory limit, with no extra workspace.
    constexpr int columns   = 32;
    constexpr int row_tiles = 2;
    using Output = W8SplitOutput2<QkvRows, ZRows>;
    for_each_token_slice(x.ne[1], columns, [&](int begin, int count) {
        const Tensor input = x.slice(1, begin, count);
        Tensor first = qkv.slice(1, begin, count), second = z.slice(1, begin, count);
        const Output output{
            static_cast<__nv_bfloat16*>(first.data), static_cast<__nv_bfloat16*>(second.data)};
        w8_rowsplit_medium_t_splitk_kernel<Hidden, columns, 8, 1, 1, Output, false, row_tiles>
            <<<dim3((QkvRows + ZRows) / (16 * row_tiles), (count + columns - 1) / columns), 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output, count, weight.k);
    });
}

} // namespace

void w8_gdn_input_wide_splitk_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    if (weight.n == 8192) {
        if (weight.k == 1024) { launch_wide<1024, 6144, 2048>(x, weight, qkv, z, stream); }
        else { launch_wide<2048, 6144, 2048>(x, weight, qkv, z, stream); }
    } else {
        if (weight.k == 2560) { launch_wide<2560, 8192, 4096>(x, weight, qkv, z, stream); }
        else { launch_wide<2048, 8192, 4096>(x, weight, qkv, z, stream); }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
