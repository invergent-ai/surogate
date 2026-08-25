// surogate vendor patch (PATCHES.md #17/#19): W8A8-int IMMA linear_swiglu.
//
// Large-T prefill path: per-token int8 activation quantization, then ONE
// IMMA GEMM whose row map interleaves gate/up (logical 2i/2i+1) so each
// output pair meets inside a warp fragment and the epilogue writes
// silu(gate) * up directly — no pair buffer, no pairing pass (v1's
// unfused pass cost ~6% of the op and 2x the output traffic).

#include "ops/linear_swiglu/w8a8/w8a8_linear_swiglu.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/linear/w8a8/w8a8_act_quant.h"
#include "ops/linear/w8a8/w8a8_imma_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

struct SwigluPairColumnMajor {
    __nv_bfloat16* out;
    int intermediate;
    static constexpr bool kPairedRows = true;

    __device__ __forceinline__ void store_pair(int pair_row, int token, float gate,
                                               float up) const {
        out[static_cast<std::int64_t>(token) * intermediate + pair_row] =
            __float2bfloat16_rn(silu(gate) * up);
    }
};

} // namespace

std::size_t w8a8_linear_swiglu_workspace_bytes(std::int32_t gate_up_rows, std::int32_t input_rows,
                                               std::int32_t max_tokens) noexcept {
    (void)gate_up_rows;  // fused epilogue: no pair buffer, only act-quant scratch
    return w8a8_act_quant_bytes(input_rows, max_tokens);
}

void w8a8_linear_swiglu_dispatch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t k            = x.ne[0];
    const std::int32_t tokens       = x.ne[1];
    const std::int32_t gate_up_rows = gate_up_weight.n;
    const std::int32_t intermediate = gate_up_rows / 2;

    const std::size_t quant_bytes = w8a8_act_quant_bytes(k, tokens);
    auto scope                    = workspace.scope();
    auto* base = static_cast<std::uint8_t*>(workspace.alloc_bytes(quant_bytes, 16).data);

    const W8A8QuantizedActivations quantized = w8a8_act_quant(x, base, stream);

    const W8A8SwigluPairRowMap row_map{intermediate};
    const SwigluPairColumnMajor epilogue{static_cast<__nv_bfloat16*>(out.data), intermediate};
    if (tokens >= kW8A8WideMinTokens && gate_up_rows >= 4096) {
        using Cfg = W8A8ImmaWideConfig;
        const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                        static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
        w8a8_imma_gemm_kernel<W8A8SwigluPairRowMap, SwigluPairColumnMajor, Cfg>
            <<<grid, Cfg::THREADS, 0, stream>>>(
                static_cast<const std::int8_t*>(gate_up_weight.qdata),
                static_cast<const std::uint8_t*>(gate_up_weight.scales), quantized.codes,
                quantized.scales, gate_up_rows, k, tokens, row_map, epilogue);
    } else {
        using Cfg = W8A8ImmaConfig;
        const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                        static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
        w8a8_imma_gemm_kernel<W8A8SwigluPairRowMap, SwigluPairColumnMajor, Cfg>
            <<<grid, Cfg::THREADS, 0, stream>>>(
                static_cast<const std::int8_t*>(gate_up_weight.qdata),
                static_cast<const std::uint8_t*>(gate_up_weight.scales), quantized.codes,
                quantized.scales, gate_up_rows, k, tokens, row_map, epilogue);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
