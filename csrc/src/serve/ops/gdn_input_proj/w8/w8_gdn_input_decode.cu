#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/w8/w8_k2048_decode.cuh"
#include "ops/linear/w8a8/w4fp4_decode.cuh"
#include "ops/linear/w8a8/w4fp4_plane.h"

namespace sinfer::ops::detail {
namespace {

using Output = W8SplitOutput2<8192, 4096>;
// surogate vendor patch (PATCHES.md #13/#16): qwen3.5-0.8b and -2b share the
// fused qkvz row structure (6144 conv channels + 2048 z rows); only K differs.
using Output08 = W8SplitOutput2<6144, 2048>;

struct W8GdnDecodeConvEpilogue {
    GdnConvEpilogue<SnapshotHistoryPublish> conv;
    __nv_bfloat16* z;
    std::int32_t z_offset;  // conv channels before the z segment (8192 or 6144)

    template <class IgnoredOutput>
    __device__ __forceinline__ void operator()(const IgnoredOutput&, std::int32_t, std::int32_t row,
                                               float accumulator) const {
        if (row < z_offset) {
            const float projected[1]{accumulator};
            conv.store(row, projected);
        } else {
            z[row - z_offset] = __float2bfloat16_rn(accumulator);
        }
    }
};

GdnConvEpilogue<SnapshotHistoryPublish>
make_conv_epilogue(const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
                   const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query,
                   Tensor& key, Tensor& value, std::int32_t channels, std::int32_t value_rows) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        channels,
        2048,
        2048,
        value_rows,
        0,
        1,
        0,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data), channels},
    };
}

} // namespace

void w8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    constexpr int kRowsPerCta = 8;
    if (weight.n == 8192) {
        constexpr int kRows08 = 8192;
        const Output08 output{static_cast<__nv_bfloat16*>(qkv.data),
                              static_cast<__nv_bfloat16*>(z.data)};
        const auto* xin     = static_cast<const __nv_bfloat16*>(x.data);
        const auto* codes   = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales  = static_cast<const std::uint8_t*>(weight.scales);
        const dim3 grid(kRows08 / kRowsPerCta);
        if (weight.k == 1024) {
            w8_k2048_decode_kernel<kRows08, kRowsPerCta, Output08, W8DecodeStoreEpilogue, 1024>
                <<<grid, kRowsPerCta * 32, 0, stream>>>(xin, codes, scales, output);
        } else {
            w8_k2048_decode_kernel<kRows08, kRowsPerCta, Output08>
                <<<grid, kRowsPerCta * 32, 0, stream>>>(xin, codes, scales, output);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int kRows = 12288;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b shares the 35B fused
    // row structure at hidden 2560.
    if (weight.k == 2560) {
        // surogate vendor patch (PATCHES.md #22): fp4 profile decode.
        if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
            const W4Fp4Plane plane = w4fp4_plane_for(weight, stream);
            if (plane.codes != nullptr) {
                w4fp4_decode_kernel<kRows, kRowsPerCta, Output, W8DecodeStoreEpilogue, 2560>
                    <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data), plane.codes, plane.sf,
                        plane.row_scales, output);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        w8_k2048_decode_kernel<kRows, kRowsPerCta, Output, W8DecodeStoreEpilogue, 2560>
            <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    w8_k2048_decode_kernel<kRows, kRowsPerCta>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

void w8_gdn_input_decode_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    constexpr int kRowsPerCta = 8;
    if (weight.n == 8192) {
        constexpr int kRows08 = 8192;
        const Output08 ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                      static_cast<__nv_bfloat16*>(z.data)};
        const W8GdnDecodeConvEpilogue epilogue{
            make_conv_epilogue(conv_weight, conv_states, valid_columns, initial_slot,
                               snapshot_base_slot, query, key, value, 6144, 2048),
            static_cast<__nv_bfloat16*>(z.data),
            6144,
        };
        const auto* xin    = static_cast<const __nv_bfloat16*>(x.data);
        const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
        const dim3 grid(kRows08 / kRowsPerCta);
        if (weight.k == 1024) {
            w8_k2048_decode_kernel<kRows08, kRowsPerCta, Output08, W8GdnDecodeConvEpilogue, 1024>
                <<<grid, kRowsPerCta * 32, 0, stream>>>(xin, codes, scales, ignored_output,
                                                        epilogue);
        } else {
            w8_k2048_decode_kernel<kRows08, kRowsPerCta, Output08, W8GdnDecodeConvEpilogue>
                <<<grid, kRowsPerCta * 32, 0, stream>>>(xin, codes, scales, ignored_output,
                                                        epilogue);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int kRows = 12288;
    const Output ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                static_cast<__nv_bfloat16*>(z.data)};
    const W8GdnDecodeConvEpilogue epilogue{
        make_conv_epilogue(conv_weight, conv_states, valid_columns, initial_slot,
                           snapshot_base_slot, query, key, value, 8192, 4096),
        static_cast<__nv_bfloat16*>(z.data),
        8192,
    };
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b shares the 35B fused
    // row structure at hidden 2560.
    if (weight.k == 2560) {
        // surogate vendor patch (PATCHES.md #22): fp4 profile decode.
        if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
            const W4Fp4Plane plane = w4fp4_plane_for(weight, stream);
            if (plane.codes != nullptr) {
                w4fp4_decode_kernel<kRows, kRowsPerCta, Output, W8GdnDecodeConvEpilogue, 2560>
                    <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data), plane.codes, plane.sf,
                        plane.row_scales, ignored_output, epilogue);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        w8_k2048_decode_kernel<kRows, kRowsPerCta, Output, W8GdnDecodeConvEpilogue, 2560>
            <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    w8_k2048_decode_kernel<kRows, kRowsPerCta, Output, W8GdnDecodeConvEpilogue>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
