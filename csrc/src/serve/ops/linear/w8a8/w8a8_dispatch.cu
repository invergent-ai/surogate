// surogate vendor patch (PATCHES.md #17): W8A8-int IMMA dispatchers for the
// split-output and residual projection families. Each is act-quant plus one
// w8a8_imma_gemm launch with a direct-write epilogue (no intermediate
// buffer; workspace holds only the quantized activations).

#include "ops/linear/w8a8/w8a8_dispatch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8a8/w8a8_imma_gemm.cuh"
#include "ops/linear/w8a8/w4fp4_cutlass_gemm.h"
#include "ops/linear/w8a8/w4fp4_gemm.cuh"
#include "ops/linear/w8a8/w4fp4_plane.h"
#include "ops/linear/w8a8/w8fp8_gemm.cuh"
#include "ops/linear/w8a8/w8fp8_plane.h"

#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

struct Split2ColumnMajor {
    __nv_bfloat16* first;
    __nv_bfloat16* second;
    int first_rows;
    int second_rows;

    __device__ __forceinline__ void operator()(int row, int token, float value) const {
        if (row < first_rows) {
            first[static_cast<std::int64_t>(token) * first_rows + row] =
                __float2bfloat16_rn(value);
        } else {
            second[static_cast<std::int64_t>(token) * second_rows + (row - first_rows)] =
                __float2bfloat16_rn(value);
        }
    }
};

struct Split4ColumnMajor {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;
    int q_rows;
    int kv_rows;

    __device__ __forceinline__ void operator()(int row, int token, float projected) const {
        // Fused weight row order: [q | k | gate | v].
        const std::int64_t t = token;
        if (row < q_rows) {
            query[t * q_rows + row] = __float2bfloat16_rn(projected);
        } else if (row < q_rows + kv_rows) {
            key[t * kv_rows + (row - q_rows)] = __float2bfloat16_rn(projected);
        } else if (row < 2 * q_rows + kv_rows) {
            gate[t * q_rows + (row - q_rows - kv_rows)] = __float2bfloat16_rn(projected);
        } else {
            value[t * kv_rows + (row - 2 * q_rows - kv_rows)] = __float2bfloat16_rn(projected);
        }
    }
};

struct ResidualColumnMajor {
    __nv_bfloat16* residual;
    int rows;

    __device__ __forceinline__ void operator()(int row, int token, float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + row;
        residual[index] =
            __float2bfloat16_rn(__bfloat162float(residual[index]) + value);
    }
};

// surogate patch (PATCHES.md #25): post-GEMM splits for the cutlass path.
// D is [tokens, parent_rows] row-major; consumers want per-tensor
// column-major [rows, tokens].
__global__ void w4fp4_split2_kernel(const __nv_bfloat16* __restrict__ d, __nv_bfloat16* first,
                                    __nv_bfloat16* second, int first_rows, int second_rows,
                                    int tokens) {
    const int parent = first_rows + second_rows;
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(parent) * tokens) { return; }
    const int row   = static_cast<int>(i % parent);
    const int token = static_cast<int>(i / parent);
    const float v   = __bfloat162float(d[static_cast<std::int64_t>(token) * parent + row]);
    if (row < first_rows) {
        first[static_cast<std::int64_t>(token) * first_rows + row] = __float2bfloat16_rn(v);
    } else {
        second[static_cast<std::int64_t>(token) * second_rows + (row - first_rows)] =
            __float2bfloat16_rn(v);
    }
}

__global__ void w4fp4_split4_kernel(const __nv_bfloat16* __restrict__ d, __nv_bfloat16* q,
                                    __nv_bfloat16* k, __nv_bfloat16* g, __nv_bfloat16* v,
                                    int q_rows, int kv_rows, int tokens) {
    const int parent = 2 * q_rows + 2 * kv_rows;
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(parent) * tokens) { return; }
    const int row   = static_cast<int>(i % parent);
    const int token = static_cast<int>(i / parent);
    const __nv_bfloat16 value = d[static_cast<std::int64_t>(token) * parent + row];
    if (row < q_rows) {
        q[static_cast<std::int64_t>(token) * q_rows + row] = value;
    } else if (row < q_rows + kv_rows) {
        k[static_cast<std::int64_t>(token) * kv_rows + (row - q_rows)] = value;
    } else if (row < 2 * q_rows + kv_rows) {
        g[static_cast<std::int64_t>(token) * q_rows + (row - q_rows - kv_rows)] = value;
    } else {
        v[static_cast<std::int64_t>(token) * kv_rows + (row - 2 * q_rows - kv_rows)] = value;
    }
}

struct W4Fp4CutlassEntry {
    W4Fp4AtomActivations acts;
    const W4Fp4Plane* plane;
    __nv_bfloat16* stage;  // null for the residual family
};

// Quantizes activations and (optionally) claims the stage buffer. Returns
// false when the cutlass path is unavailable (caller falls back).
bool w4fp4_cutlass_begin(const Tensor& x, const W4Fp4Plane& plane, bool with_stage,
                         std::int32_t parent_rows, WorkspaceArena& workspace,
                         cudaStream_t stream, W4Fp4CutlassEntry& out) {
    if (plane.sf_atom == nullptr || w4fp4_alpha_one() == nullptr) { return false; }
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::size_t quant_bytes =
        w4fp4_cutlass_workspace_bytes(parent_rows, k, tokens, false);
    auto* base = workspace.alloc_bytes(quant_bytes, 16).data;
    out.acts   = w4fp4_act_quant_atom(x, base, stream);
    out.plane  = &plane;
    out.stage  = nullptr;
    if (with_stage) {
        out.stage = static_cast<__nv_bfloat16*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(parent_rows) * tokens *
                                 sizeof(__nv_bfloat16),
                             16)
                .data);
    }
    return true;
}

template <class Epilogue>
void launch(const Tensor& x, const Weight& weight, Epilogue epilogue, WorkspaceArena& workspace,
            cudaStream_t stream) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    auto scope                = workspace.scope();
    auto* quant_base = workspace.alloc_bytes(w8a8_act_quant_bytes(k, tokens), 16).data;

    // surogate vendor patch (PATCHES.md #21): NVFP4 profile (opt-in via
    // SUROGATE_SERVE_PREFILL_QUANT=fp4) — hardware block-scale mma.
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane fp4_plane = w4fp4_plane_for(weight, stream);
        if (fp4_plane.codes != nullptr) {
            const W4Fp4QuantizedActivations fp4 = w4fp4_act_quant(x, quant_base, stream);
            if (tokens >= kW8A8WideMinTokens && weight.n >= 2560) {
                using Cfg = W4Fp4WideConfig;
                const dim3 grid(static_cast<unsigned>(div_up(weight.n, Cfg::BM)),
                                static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
                w4fp4_gemm_kernel<W8A8IdentityRowMap, Epilogue, Cfg>
                    <<<grid, Cfg::THREADS, 0, stream>>>(
                        fp4_plane.codes, fp4_plane.sf, fp4_plane.row_scales, fp4.codes, fp4.sf,
                        fp4.scales, weight.n, k, tokens, W8A8IdentityRowMap{}, epilogue);
            } else {
                using Cfg = W4Fp4Config;
                const dim3 grid(static_cast<unsigned>(div_up(weight.n, Cfg::BM)),
                                static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
                w4fp4_gemm_kernel<W8A8IdentityRowMap, Epilogue, Cfg>
                    <<<grid, Cfg::THREADS, 0, stream>>>(
                        fp4_plane.codes, fp4_plane.sf, fp4_plane.row_scales, fp4.codes, fp4.sf,
                        fp4.scales, weight.n, k, tokens, W8A8IdentityRowMap{}, epilogue);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    // surogate vendor patch (PATCHES.md #20): folded-scale FP8 plane, when
    // derived, replaces the IMMA path (same epilogues, same workspace).
    const W8Fp8Plane plane = w8fp8_plane_for(weight, stream);
    if (plane.codes != nullptr) {
        const W8Fp8QuantizedActivations fp8 = w8fp8_act_quant(x, quant_base, stream);
        if (tokens >= kW8A8WideMinTokens && weight.n >= 2560) {
            using Cfg = W8A8ImmaWideConfig;
            const dim3 grid(static_cast<unsigned>(div_up(weight.n, Cfg::BM)),
                            static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
            w8fp8_gemm_kernel<W8A8IdentityRowMap, Epilogue, Cfg>
                <<<grid, Cfg::THREADS, 0, stream>>>(plane.codes, plane.row_scales, fp8.codes,
                                                    fp8.scales, weight.n, k, tokens,
                                                    W8A8IdentityRowMap{}, epilogue);
        } else {
            using Cfg = W8A8ImmaConfig;
            const dim3 grid(static_cast<unsigned>(div_up(weight.n, Cfg::BM)),
                            static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
            w8fp8_gemm_kernel<W8A8IdentityRowMap, Epilogue, Cfg>
                <<<grid, Cfg::THREADS, 0, stream>>>(plane.codes, plane.row_scales, fp8.codes,
                                                    fp8.scales, weight.n, k, tokens,
                                                    W8A8IdentityRowMap{}, epilogue);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const W8A8QuantizedActivations quantized = w8a8_act_quant(x, quant_base, stream);

    // Wide tiles need grid volume: at 2048 output rows BM128 leaves the GPU
    // underfilled (measured: the residual projections ran at half rate).
    // surogate vendor patch (PATCHES.md #18): 2560 rows fill fine (20 row
    // tiles; probe: o_proj 241->214us, down 519->441us vs the base config).
    if (tokens >= kW8A8WideMinTokens && weight.n >= 2560) {
        using Cfg = W8A8ImmaWideConfig;
        const dim3 grid(static_cast<unsigned>(div_up(weight.n, Cfg::BM)),
                        static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
        w8a8_imma_gemm_kernel<W8A8IdentityRowMap, Epilogue, Cfg>
            <<<grid, Cfg::THREADS, 0, stream>>>(
                static_cast<const std::int8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), quantized.codes,
                quantized.scales, weight.n, k, tokens, W8A8IdentityRowMap{}, epilogue);
    } else {
        using Cfg = W8A8ImmaConfig;
        const dim3 grid(static_cast<unsigned>(div_up(weight.n, Cfg::BM)),
                        static_cast<unsigned>(div_up(tokens, Cfg::BN)), 1u);
        w8a8_imma_gemm_kernel<W8A8IdentityRowMap, Epilogue, Cfg>
            <<<grid, Cfg::THREADS, 0, stream>>>(
                static_cast<const std::int8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), quantized.codes,
                quantized.scales, weight.n, k, tokens, W8A8IdentityRowMap{}, epilogue);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8a8_gemm_split2(const Tensor& x, const Weight& weight, Tensor& first, Tensor& second,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    // surogate patch (PATCHES.md #25): cutlass NVFP4 path (fp4 profile).
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane plane = w4fp4_plane_for(weight, stream);
        auto scope             = workspace.scope();
        W4Fp4CutlassEntry entry;
        if (plane.codes != nullptr &&
            w4fp4_cutlass_begin(x, plane, true, weight.n, workspace, stream, entry) &&
            w4fp4_cutlass_gemm_store(entry.acts.codes, entry.acts.sf_atom, plane.codes,
                                     plane.sf_atom, w4fp4_alpha_one(), entry.stage, x.ne[1],
                                     weight.n, x.ne[0], stream)) {
            const std::int64_t total = static_cast<std::int64_t>(weight.n) * x.ne[1];
            w4fp4_split2_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
                entry.stage, static_cast<__nv_bfloat16*>(first.data),
                static_cast<__nv_bfloat16*>(second.data), first.ne[0], second.ne[0], x.ne[1]);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    launch(x, weight,
           Split2ColumnMajor{static_cast<__nv_bfloat16*>(first.data),
                             static_cast<__nv_bfloat16*>(second.data), first.ne[0], second.ne[0]},
           workspace, stream);
}

void w8a8_gemm_split4(const Tensor& x, const Weight& weight, Tensor& query, Tensor& key,
                      Tensor& gate, Tensor& value, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    // surogate patch (PATCHES.md #25): cutlass NVFP4 path (fp4 profile).
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane plane = w4fp4_plane_for(weight, stream);
        auto scope             = workspace.scope();
        W4Fp4CutlassEntry entry;
        if (plane.codes != nullptr &&
            w4fp4_cutlass_begin(x, plane, true, weight.n, workspace, stream, entry) &&
            w4fp4_cutlass_gemm_store(entry.acts.codes, entry.acts.sf_atom, plane.codes,
                                     plane.sf_atom, w4fp4_alpha_one(), entry.stage, x.ne[1],
                                     weight.n, x.ne[0], stream)) {
            const std::int64_t total = static_cast<std::int64_t>(weight.n) * x.ne[1];
            w4fp4_split4_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
                entry.stage, static_cast<__nv_bfloat16*>(query.data),
                static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(gate.data),
                static_cast<__nv_bfloat16*>(value.data), query.ne[0], key.ne[0], x.ne[1]);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    launch(x, weight,
           Split4ColumnMajor{static_cast<__nv_bfloat16*>(query.data),
                             static_cast<__nv_bfloat16*>(key.data),
                             static_cast<__nv_bfloat16*>(gate.data),
                             static_cast<__nv_bfloat16*>(value.data), query.ne[0], key.ne[0]},
           workspace, stream);
}

void w8a8_gemm_residual(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    // surogate patch (PATCHES.md #25): cutlass NVFP4 path — the residual add
    // is the epilogue's beta=1 (C = D = the residual tensor).
    if (w8_prefill_quant_mode() == PrefillQuantMode::Fp4) {
        const W4Fp4Plane plane = w4fp4_plane_for(weight, stream);
        auto scope             = workspace.scope();
        W4Fp4CutlassEntry entry;
        if (plane.codes != nullptr &&
            w4fp4_cutlass_begin(x, plane, false, weight.n, workspace, stream, entry) &&
            w4fp4_cutlass_gemm_residual(entry.acts.codes, entry.acts.sf_atom, plane.codes,
                                        plane.sf_atom, w4fp4_alpha_one(), residual_out.data,
                                        x.ne[1], weight.n, x.ne[0], stream)) {
            return;
        }
    }
    launch(x, weight,
           ResidualColumnMajor{static_cast<__nv_bfloat16*>(residual_out.data),
                               residual_out.ne[0]},
           workspace, stream);
}

} // namespace sinfer::ops::detail
