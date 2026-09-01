#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_k2048_decode.cuh"
#include "ops/linear/w8a8/w4fp4_decode.cuh"
#include "ops/linear/w8a8/w4fp4_plane.h"

namespace sinfer::ops::detail {

namespace {

template <int RowsPerCta>
void launch_companion_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    constexpr int kRows = 6144;
    static_assert((4096 % RowsPerCta) == 0 && (1024 % RowsPerCta) == 0);
    using Output = W8SplitOutput3<4096, 1024, 1024>;
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_k2048_decode_kernel<kRows, RowsPerCta><<<kRows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    // surogate vendor patch (PATCHES.md #13/#16): qwen3.5-0.8b/-2b fused qkgv
    // (5120 rows = q2048|k512|gate2048|v512; hidden 1024 or 2048).
    // surogate vendor patch (PATCHES.md #18): qwen3.5-4b fused qkgv.
    if (weight.n == 10240) {
        constexpr int kRows4B       = 10240;
        constexpr int kRowsPerCta4B = 8;
        using Output4B              = W8SplitOutput4<4096, 1024, 4096, 1024>;
        const Output4B output{static_cast<__nv_bfloat16*>(q.data),
                              static_cast<__nv_bfloat16*>(k.data),
                              static_cast<__nv_bfloat16*>(gate.data),
                              static_cast<__nv_bfloat16*>(v.data)};
        // surogate vendor patch (PATCHES.md #22): fp4 profile decode reads
        // the derived NVFP4 plane (half the weight traffic).
        if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
            const W4Fp4Plane plane = w4fp4_plane_for(weight, stream);
            if (plane.codes != nullptr) {
                w4fp4_decode_kernel<kRows4B, kRowsPerCta4B, Output4B, W8DecodeStoreEpilogue, 2560>
                    <<<kRows4B / kRowsPerCta4B, kRowsPerCta4B * 32, 0, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data), plane.codes, plane.sf,
                        plane.row_scales, output);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        w8_k2048_decode_kernel<kRows4B, kRowsPerCta4B, Output4B, W8DecodeStoreEpilogue, 2560>
            <<<kRows4B / kRowsPerCta4B, kRowsPerCta4B * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (weight.n == 5120) {
        constexpr int kRows08       = 5120;
        constexpr int kRowsPerCta08 = 8;
        using Output08              = W8SplitOutput4<2048, 512, 2048, 512>;
        const Output08 output{static_cast<__nv_bfloat16*>(q.data),
                              static_cast<__nv_bfloat16*>(k.data),
                              static_cast<__nv_bfloat16*>(gate.data),
                              static_cast<__nv_bfloat16*>(v.data)};
        const auto* xin    = static_cast<const __nv_bfloat16*>(x.data);
        const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
        const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
        const dim3 grid(kRows08 / kRowsPerCta08);
        if (weight.k == 1024) {
            w8_k2048_decode_kernel<kRows08, kRowsPerCta08, Output08, W8DecodeStoreEpilogue, 1024>
                <<<grid, kRowsPerCta08 * 32, 0, stream>>>(xin, codes, scales, output);
        } else {
            w8_k2048_decode_kernel<kRows08, kRowsPerCta08, Output08>
                <<<grid, kRowsPerCta08 * 32, 0, stream>>>(xin, codes, scales, output);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int kRows       = 9216;
    constexpr int kRowsPerCta = 8;
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    using Output = W8SplitOutput4<4096, 512, 4096, 512>;
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(gate.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_k2048_decode_kernel<kRows, kRowsPerCta>
        <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

void w8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                 Tensor& v, cudaStream_t stream) {
    // Qwen3-0.6B's ungated fused qkv: 4096 rows = q2048 | k1024 | v1024 at k 1024.
    if (weight.n == 4096) {
        constexpr int kRowsQwen3       = 4096;
        constexpr int kRowsPerCtaQwen3 = 8;
        static_assert((2048 % kRowsPerCtaQwen3) == 0 && (1024 % kRowsPerCtaQwen3) == 0);
        using OutputQwen3 = W8SplitOutput3<2048, 1024, 1024>;
        const OutputQwen3 output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
        w8_k2048_decode_kernel<kRowsQwen3, kRowsPerCtaQwen3, OutputQwen3, W8DecodeStoreEpilogue,
                               1024>
            <<<kRowsQwen3 / kRowsPerCtaQwen3, kRowsPerCtaQwen3 * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), output);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    launch_companion_decode<8>(x, weight, q, k, v, stream);
}

void w8_companion_attn_input_decode_r4_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_companion_decode<4>(x, weight, q, k, v, stream);
}

void w8_companion_attn_input_decode_r16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_companion_decode<16>(x, weight, q, k, v, stream);
}

} // namespace sinfer::ops::detail
