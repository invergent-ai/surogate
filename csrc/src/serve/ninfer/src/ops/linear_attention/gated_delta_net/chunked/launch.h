#pragma once

#include "core/layout.h"
#include "ops/linear_attention/gated_delta_net/common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define NINFER_GATED_DELTA_NET_PROPAGATE(expr)                                                     \
    do {                                                                                           \
        const cudaError_t ninfer_gated_delta_net_error = (expr);                                   \
        if (ninfer_gated_delta_net_error != cudaSuccess) { return ninfer_gated_delta_net_error; }  \
    } while (0)

namespace ninfer::ops::detail::gated_delta_net::chunked {

inline constexpr std::size_t kWorkspaceAlign = 256;

struct workspace_layout {
    TensorRegion g_cumsum;
    TensorRegion W;
    TensorRegion U;
    TensorRegion v_new;
    TensorRegion h_chunk;
    std::size_t total_bytes = 0;
};

inline workspace_layout compute_workspace_layout(std::int32_t value_heads, std::int32_t tokens) {
    const std::int32_t chunks = tokens / kChunkSize;
    LayoutBuilder builder;
    workspace_layout w{};
    w.g_cumsum =
        builder.add_tensor(DType::FP32, {value_heads, tokens}, kWorkspaceAlign, "g_cumsum");
    w.W = builder.add_tensor(DType::BF16, {kStateDim, value_heads, tokens}, kWorkspaceAlign, "W");
    w.U = builder.add_tensor(DType::BF16, {kStateDim, value_heads, tokens}, kWorkspaceAlign, "U");
    w.v_new =
        builder.add_tensor(DType::BF16, {kStateDim, value_heads, tokens}, kWorkspaceAlign, "v_new");
    w.h_chunk     = builder.add_tensor(DType::BF16, {kStateDim, kStateDim, value_heads, chunks},
                                       kWorkspaceAlign, "h_chunk");
    w.total_bytes = builder.finish(kWorkspaceAlign, "Gated DeltaNet chunk workspace");
    return w;
}

inline std::size_t workspace_bytes(std::int32_t value_heads, std::int32_t tokens) {
    return compute_workspace_layout(value_heads, tokens).total_bytes;
}

struct prepare_wy_wu_config {
    std::int32_t H_qk = 0;
    std::int32_t H_v  = 0;
    std::int32_t L    = 0;

    const __nv_bfloat16* k = nullptr;
    const __nv_bfloat16* v = nullptr;
    const float* g_in      = nullptr;
    const float* beta      = nullptr;

    __nv_bfloat16* W    = nullptr;
    __nv_bfloat16* U    = nullptr;
    float* g_cumsum_out = nullptr;

    cudaStream_t stream = nullptr;
};

struct state_passing_config {
    std::int32_t H_qk = 0;
    std::int32_t H_v  = 0;
    std::int32_t L    = 0;

    const __nv_bfloat16* W = nullptr;
    const __nv_bfloat16* U = nullptr;
    const __nv_bfloat16* k = nullptr;
    const float* g_cumsum  = nullptr;
    const float* state_in  = nullptr;

    __nv_bfloat16* v_new   = nullptr;
    __nv_bfloat16* h_chunk = nullptr;
    float* state_out       = nullptr;

    cudaStream_t stream = nullptr;
};

struct chunk_output_config {
    std::int32_t H_qk = 0;
    std::int32_t H_v  = 0;
    std::int32_t L    = 0;

    const __nv_bfloat16* q       = nullptr;
    const __nv_bfloat16* k       = nullptr;
    const __nv_bfloat16* v_new   = nullptr;
    const float* g_cumsum        = nullptr;
    const __nv_bfloat16* h_chunk = nullptr;

    __nv_bfloat16* attn_out = nullptr;

    float scale = 0.0f;

    cudaStream_t stream = nullptr;
};

struct stage_validator {
    const char* name;
    std::int32_t H_qk;
    std::int32_t H_v;
    std::int32_t T;

    cudaError_t check_shape() const {
        if (T <= 0 || !are_head_counts_valid(H_qk, H_v)) {
            std::fprintf(stderr, "%s: invalid shape (H_qk=%d H_v=%d T=%d)\n", name, H_qk, H_v, T);
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_full_chunks() const {
        if ((T % kChunkSize) != 0) {
            std::fprintf(stderr,
                         "%s: Gated DeltaNet chunked path requires T to be a multiple of %d; "
                         "route tail tokens through AR instead (T=%lld)\n",
                         name, kChunkSize, static_cast<long long>(T));
            return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }

    cudaError_t check_grid(std::int64_t grid_x, std::int64_t grid_y,
                           std::int64_t grid_z = 1) const {
        if (grid_x > static_cast<std::int64_t>(0xffffffff)) {
            std::fprintf(stderr, "%s: grid.x too large (%lld)\n", name,
                         static_cast<long long>(grid_x));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_y > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.y too large (%lld)\n", name,
                         static_cast<long long>(grid_y));
            return cudaErrorInvalidConfiguration;
        }
        if (grid_z > static_cast<std::int64_t>(0xffff)) {
            std::fprintf(stderr, "%s: grid.z too large (%lld)\n", name,
                         static_cast<long long>(grid_z));
            return cudaErrorInvalidConfiguration;
        }
        return cudaSuccess;
    }
};

cudaError_t launch_prepare_wy_wu(const prepare_wy_wu_config& cfg);
cudaError_t launch_state_passing(const state_passing_config& cfg);
cudaError_t launch_output(const chunk_output_config& cfg);

} // namespace ninfer::ops::detail::gated_delta_net::chunked
