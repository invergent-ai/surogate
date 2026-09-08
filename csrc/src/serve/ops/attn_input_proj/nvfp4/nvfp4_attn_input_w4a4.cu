#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_split.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops::detail {
namespace {

using Geometry = Nvfp4AttnInputGeometry;

constexpr std::int32_t kQueryRows  = 6144;
constexpr std::int32_t kKeyRows    = 1024;
constexpr std::int32_t kGateRows   = 6144;
constexpr std::int32_t kKeyBegin   = kQueryRows;
constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

static_assert((kQueryRows % 128) == 0);
static_assert((kKeyRows % 128) == 0);
static_assert((kGateRows % 128) == 0);

struct Nvfp4W4a4AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

using M32N64            = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

template <class Schedule>
void launch_gemm(const Weight& weight, Nvfp4W4a4AttentionOutput output,
                 Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                 cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

void launch_mma(const Weight& weight, Nvfp4W4a4AttentionOutput output,
                Nvfp4W4a4MaterializedActivation activation, std::int32_t tokens,
                cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<M32N64>(weight, output, activation, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, output, activation, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, output, activation, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<M128N128Resident>(weight, output, activation, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<M128N128Pipelined>(weight, output, activation, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, output, activation, tokens, stream);
    }
}

} // namespace

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // Shapes outside the registered geometry have no in-house ladder; the segments are taken
    // from the output views so any q|k|gate|v split works (#84).
    if (is_nvfp4_generic_problem(weight.n, weight.k) || q.ne[0] != kQueryRows ||
        k.ne[0] != kKeyRows || gate.ne[0] != kGateRows || v.ne[0] != kKeyRows) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream, Nvfp4ScaleLayout::Tiled);
        const std::int32_t q_rows    = q.ne[0];
        const std::int32_t k_rows    = k.ne[0];
        const std::int32_t gate_rows = gate.ne[0];
        const std::int32_t v_rows    = v.ne[0];
        if (q_rows + k_rows + gate_rows + v_rows != weight.n) {
            throw std::invalid_argument("nvfp4 attn_input_proj: segments do not cover the weight");
        }
        const struct {
            std::int32_t begin;
            std::int32_t rows;
            Tensor* out;
        } segments[] = {
            {0, q_rows, &q},
            {q_rows, k_rows, &k},
            {q_rows + k_rows, gate_rows, &gate},
            {q_rows + k_rows + gate_rows, v_rows, &v},
        };
        for (const auto& segment : segments) {
            nvfp4_cublaslt_gemm(weight, segment.begin, segment.rows, workspace.codes,
                                workspace.scales,
                                static_cast<__nv_bfloat16*>(segment.out->data), segment.rows,
                                tokens, 0.0F, stream);
        }
        return;
    }
    if (nvfp4_cublaslt_route(tokens)) {
        launch_nvfp4_w4a4_quantize(x, weight, workspace, stream, Nvfp4ScaleLayout::Tiled);
        nvfp4_cublaslt_gemm(weight, 0, kQueryRows, workspace.codes, workspace.scales,
                            static_cast<__nv_bfloat16*>(q.data), kQueryRows, tokens, 0.0F, stream);
        nvfp4_cublaslt_gemm(weight, kKeyBegin, kKeyRows, workspace.codes, workspace.scales,
                            static_cast<__nv_bfloat16*>(k.data), kKeyRows, tokens, 0.0F, stream);
        nvfp4_cublaslt_gemm(weight, kGateBegin, kGateRows, workspace.codes, workspace.scales,
                            static_cast<__nv_bfloat16*>(gate.data), kGateRows, tokens, 0.0F, stream);
        nvfp4_cublaslt_gemm(weight, kValueBegin, kKeyRows, workspace.codes, workspace.scales,
                            static_cast<__nv_bfloat16*>(v.data), kKeyRows, tokens, 0.0F, stream);
        return;
    }
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const Nvfp4W4a4TmaSplit split = nvfp4_w4a4_tma_split(tokens);
    if (split.tma_tokens > 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_attention(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(v.data), split.tma_tokens, alpha, stream);
    }
    if (split.tail_tokens > 0) {
        const Nvfp4W4a4Workspace tail =
            nvfp4_w4a4_workspace_at(workspace, Geometry::kInputRows, split.tma_tokens);
        launch_mma(weight,
                   Nvfp4W4a4AttentionOutput{
                       nvfp4_w4a4_column(q.data, kQueryRows, split.tma_tokens),
                       nvfp4_w4a4_column(k.data, kKeyRows, split.tma_tokens),
                       nvfp4_w4a4_column(gate.data, kGateRows, split.tma_tokens),
                       nvfp4_w4a4_column(v.data, kKeyRows, split.tma_tokens),
                   },
                   Nvfp4W4a4MaterializedActivation{tail.codes, tail.scales}, split.tail_tokens,
                   stream);
    }
}

} // namespace sinfer::ops::detail
