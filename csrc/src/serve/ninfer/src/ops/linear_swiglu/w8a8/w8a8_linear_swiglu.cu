// surogate vendor patch (PATCHES.md #17): W8A8-int IMMA linear_swiglu.
//
// Large-T prefill path: per-token int8 activation quantization, the IMMA
// GEMM over the fused gate_up weight into a workspace buffer, then a
// pairing pass out[i] = silu(gate[i]) * up[i]. The pairing is unfused in
// v1 (the gate/up halves land in different CTAs; a smem-exchange fused
// epilogue is a tracked follow-up) and costs ~6% of the op.

#include "ops/linear_swiglu/w8a8/w8a8_linear_swiglu.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/linear/w8a8/w8a8_act_quant.h"
#include "ops/linear/w8a8/w8a8_imma_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

struct StoreColumnMajor {
    __nv_bfloat16* out;
    int rows;

    __device__ __forceinline__ void operator()(int row, int token, float value) const {
        out[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

__global__ void swiglu_pair_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                   __nv_bfloat16* __restrict__ out, int intermediate,
                                   int gate_up_rows) {
    const int token   = static_cast<int>(blockIdx.y);
    const int i       = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= intermediate) { return; }
    const std::int64_t base = static_cast<std::int64_t>(token) * gate_up_rows;
    const float gate        = __bfloat162float(gate_up[base + i]);
    const float up          = __bfloat162float(gate_up[base + intermediate + i]);
    out[static_cast<std::int64_t>(token) * intermediate + i] =
        __float2bfloat16_rn(silu(gate) * up);
}

constexpr std::size_t align16(std::size_t bytes) noexcept {
    return (bytes + 15u) & ~std::size_t{15u};
}

} // namespace

std::size_t w8a8_linear_swiglu_workspace_bytes(std::int32_t gate_up_rows, std::int32_t input_rows,
                                               std::int32_t max_tokens) noexcept {
    return w8a8_act_quant_bytes(input_rows, max_tokens) +
           align16(static_cast<std::size_t>(gate_up_rows) * max_tokens * sizeof(__nv_bfloat16));
}

void w8a8_linear_swiglu_dispatch(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t k            = x.ne[0];
    const std::int32_t tokens       = x.ne[1];
    const std::int32_t gate_up_rows = gate_up_weight.n;
    const std::int32_t intermediate = gate_up_rows / 2;

    const std::size_t quant_bytes = w8a8_act_quant_bytes(k, tokens);
    const std::size_t gemm_bytes =
        align16(static_cast<std::size_t>(gate_up_rows) * tokens * sizeof(__nv_bfloat16));
    auto scope = workspace.scope();
    auto* base =
        static_cast<std::uint8_t*>(workspace.alloc_bytes(quant_bytes + gemm_bytes, 16).data);

    const W8A8QuantizedActivations quantized = w8a8_act_quant(x, base, stream);
    auto* gemm_out = reinterpret_cast<__nv_bfloat16*>(base + quant_bytes);

    using Cfg = W8A8ImmaConfig;
    const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                    static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
    w8a8_imma_gemm_kernel<W8A8IdentityRowMap, StoreColumnMajor>
        <<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const std::int8_t*>(gate_up_weight.qdata),
            static_cast<const std::uint8_t*>(gate_up_weight.scales), quantized.codes,
            quantized.scales, gate_up_rows, k, tokens, W8A8IdentityRowMap{},
            StoreColumnMajor{gemm_out, gate_up_rows});
    CUDA_CHECK(cudaGetLastError());

    const dim3 pair_grid(static_cast<unsigned>(div_up(intermediate, 256)),
                         static_cast<unsigned>(tokens), 1u);
    swiglu_pair_kernel<<<pair_grid, 256, 0, stream>>>(
        gemm_out, static_cast<__nv_bfloat16*>(out.data), intermediate, gate_up_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
