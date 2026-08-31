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
#include "ops/linear/w8a8/w4fp4_cutlass_gemm.h"
#include "ops/linear/w8a8/w4fp4_gemm.cuh"
#include "ops/linear/w8a8/w4fp4_plane.h"
#include "ops/linear/w8a8/w8fp8_gemm.cuh"
#include "ops/linear/w8a8/w8fp8_plane.h"

#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

// surogate patch (PATCHES.md #25): pairing pass for the cutlass path
// (D is [tokens, 2*intermediate] row-major: gate rows then up rows).
__global__ void w4fp4_swiglu_pair_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                         __nv_bfloat16* __restrict__ out, int intermediate,
                                         int tokens) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(intermediate) * tokens) { return; }
    const int row           = static_cast<int>(i % intermediate);
    const int token         = static_cast<int>(i / intermediate);
    const std::int64_t base = static_cast<std::int64_t>(token) * 2 * intermediate;
    const float gate        = __bfloat162float(gate_up[base + row]);
    const float up          = __bfloat162float(gate_up[base + intermediate + row]);
    out[static_cast<std::int64_t>(token) * intermediate + row] =
        __float2bfloat16_rn(silu(gate) * up);
}

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
    // Fused-epilogue paths need only act-quant scratch. The cutlass fp4 path
    // stages [tokens, gate_up_rows] BF16 plus its atom-SF quant buffers —
    // sized only when the fp4 profile is active so the default profile keeps
    // its exact query == high-water contract (enforced by the op tests).
    if (w8_prefill_quant_mode() != PrefillQuantMode::Fp4) {
        return w8a8_act_quant_bytes(input_rows, max_tokens);
    }
    const std::size_t fused = w8a8_act_quant_bytes(input_rows, max_tokens);
    const std::size_t cutlass =
        w4fp4_cutlass_workspace_bytes(gate_up_rows, input_rows, max_tokens, true);
    return fused > cutlass ? fused : cutlass;
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

    const W8A8SwigluPairRowMap row_map{intermediate};
    const SwigluPairColumnMajor epilogue{static_cast<__nv_bfloat16*>(out.data), intermediate};

    // surogate patch (PATCHES.md #25): cutlass NVFP4 path.
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane cut_plane = w4fp4_plane_for(gate_up_weight, stream);
        if (cut_plane.codes != nullptr && cut_plane.sf_atom != nullptr &&
            w4fp4_alpha_one() != nullptr) {
            auto cut_scope = workspace.scope();
            const std::size_t quant_bytes =
                w4fp4_cutlass_workspace_bytes(gate_up_rows, k, tokens, false);
            auto* qbase = workspace.alloc_bytes(quant_bytes, 16).data;
            const W4Fp4AtomActivations acts = w4fp4_act_quant_atom(x, qbase, stream);
            auto* stage = static_cast<__nv_bfloat16*>(
                workspace
                    .alloc_bytes(static_cast<std::size_t>(gate_up_rows) * tokens *
                                     sizeof(__nv_bfloat16),
                                 16)
                    .data);
            if (w4fp4_cutlass_gemm_store(acts.codes, acts.sf_atom, cut_plane.codes,
                                         cut_plane.sf_atom, w4fp4_alpha_one(), stage, tokens,
                                         gate_up_rows, k, stream)) {
                const std::int64_t total =
                    static_cast<std::int64_t>(intermediate) * tokens;
                w4fp4_swiglu_pair_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                                           stream>>>(stage,
                                                     static_cast<__nv_bfloat16*>(out.data),
                                                     intermediate, tokens);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
    }

    // surogate vendor patch (PATCHES.md #21): NVFP4 profile (kt4 fallback).
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane fp4_plane = w4fp4_plane_for(gate_up_weight, stream);
        if (fp4_plane.codes != nullptr) {
            const W4Fp4QuantizedActivations fp4 = w4fp4_act_quant(x, base, stream);
            if (tokens >= kW8A8WideMinTokens && gate_up_rows >= 4096) {
                using Cfg = W4Fp4WideConfig;
                const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                                static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
                w4fp4_gemm_kernel<W8A8SwigluPairRowMap, SwigluPairColumnMajor, Cfg>
                    <<<grid, Cfg::THREADS, 0, stream>>>(
                        fp4_plane.codes, fp4_plane.sf, fp4_plane.row_scales, fp4.codes, fp4.sf,
                        fp4.scales, gate_up_rows, k, tokens, row_map, epilogue);
            } else {
                using Cfg = W4Fp4Config;
                const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                                static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
                w4fp4_gemm_kernel<W8A8SwigluPairRowMap, SwigluPairColumnMajor, Cfg>
                    <<<grid, Cfg::THREADS, 0, stream>>>(
                        fp4_plane.codes, fp4_plane.sf, fp4_plane.row_scales, fp4.codes, fp4.sf,
                        fp4.scales, gate_up_rows, k, tokens, row_map, epilogue);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    // surogate vendor patch (PATCHES.md #20): folded-scale FP8 plane.
    const W8Fp8Plane plane = w8fp8_plane_for(gate_up_weight, stream);
    if (plane.codes != nullptr) {
        const W8Fp8QuantizedActivations fp8 = w8fp8_act_quant(x, base, stream);
        if (tokens >= kW8A8WideMinTokens && gate_up_rows >= 4096) {
            using Cfg = W8A8ImmaWideConfig;
            const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                            static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
            w8fp8_gemm_kernel<W8A8SwigluPairRowMap, SwigluPairColumnMajor, Cfg>
                <<<grid, Cfg::THREADS, 0, stream>>>(plane.codes, plane.row_scales, fp8.codes,
                                                    fp8.scales, gate_up_rows, k, tokens, row_map,
                                                    epilogue);
        } else {
            using Cfg = W8A8ImmaConfig;
            const dim3 grid(static_cast<unsigned>(div_up(gate_up_rows, Cfg::BM)),
                            static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
            w8fp8_gemm_kernel<W8A8SwigluPairRowMap, SwigluPairColumnMajor, Cfg>
                <<<grid, Cfg::THREADS, 0, stream>>>(plane.codes, plane.row_scales, fp8.codes,
                                                    fp8.scales, gate_up_rows, k, tokens, row_map,
                                                    epilogue);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const W8A8QuantizedActivations quantized = w8a8_act_quant(x, base, stream);
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

} // namespace sinfer::ops::detail
