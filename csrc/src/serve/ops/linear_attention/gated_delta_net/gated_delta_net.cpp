#include "core/limits.h"
#include "api/ops/gated_delta_net.h"

#include "api/ops/l2norm.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

struct Geometry {
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t tokens;
};

void require_dtype(const Tensor& t, DType dtype, const char* name) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string("gated_delta_net: ") + name); }
}

void require_shape(const Tensor& t, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != n3) {
        throw std::invalid_argument(std::string("gated_delta_net: invalid shape for ") + name);
    }
}

void require_contiguous_nonnull(const Tensor& t, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string("gated_delta_net: ") + name +
                                    " data must be non-null");
    }
}

Geometry require_geometry(const Tensor& q, const Tensor& v) {
    const Geometry geometry{q.ne[1], v.ne[1], q.ne[2]};
    if (q.ne[0] != detail::gated_delta_net::kStateDim) {
        throw std::invalid_argument("gated_delta_net: state/head dimension must be 128");
    }
    if (!detail::gated_delta_net::are_head_counts_valid(geometry.qk_heads, geometry.value_heads)) {
        throw std::invalid_argument(
            "gated_delta_net: value heads must be at least q/k heads and divisible by them");
    }
    if (geometry.tokens <= 0) {
        throw std::invalid_argument("gated_delta_net: T must be positive");
    }
    return geometry;
}

void require_scale(float scale) {
    const float expected_scale =
        1.0f / std::sqrt(static_cast<float>(detail::gated_delta_net::kStateDim));
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - expected_scale) > 1.0e-6f) {
        throw std::invalid_argument("gated_delta_net: scale must be 1/sqrt(128)");
    }
}

Geometry validate_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& ssm_state,
                            const Tensor& out) {
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be BF16");
    require_dtype(beta, DType::FP32, "beta must be BF16");
    require_dtype(ssm_state, DType::BF16, "ssm_state must be BF16");

    const Geometry geometry = require_geometry(q, v);
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, 1,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens, 1,
                  "out");
    require_shape(g, geometry.value_heads, geometry.tokens, 1, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, 1, 1, "beta");
    require_shape(ssm_state, detail::gated_delta_net::kStateDim, detail::gated_delta_net::kStateDim,
                  geometry.value_heads, 1, "ssm_state");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_state, "ssm_state");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

Geometry validate_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& ssm_states, const Tensor& valid_columns,
                                     const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, const Tensor& out) {
    constexpr std::int32_t kMaximumBatch = kMaximumBatchColumns;
    constexpr std::int32_t kMaximumWidth = 16;
    const bool masked                    = valid_columns.data != nullptr;
    require_dtype(q, DType::BF16, "q must be BF16");
    require_dtype(k, DType::BF16, "k must be BF16");
    require_dtype(v, DType::BF16, "v must be BF16");
    require_dtype(out, DType::BF16, "out must be BF16");
    require_dtype(g, DType::FP32, "g must be BF16");
    require_dtype(beta, DType::FP32, "beta must be BF16");
    require_dtype(ssm_states, DType::BF16, "ssm_states must be BF16");
    if (masked) { require_dtype(valid_columns, DType::I32, "valid_columns must be I32"); }
    require_dtype(initial_state_slots, DType::I32, "initial_state_slots must be I32");
    require_dtype(snapshot_base_slots, DType::I32, "snapshot_base_slots must be I32");

    const Geometry geometry  = require_geometry(q, v);
    const std::int32_t batch = q.ne[3];
    if (batch <= 0 || batch > kMaximumBatch || (batch > 1 && geometry.tokens > kMaximumWidth)) {
        throw std::invalid_argument("gated_delta_net: unsupported snapshot B/W domain");
    }
    require_shape(q, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "q");
    require_shape(k, detail::gated_delta_net::kStateDim, geometry.qk_heads, geometry.tokens, batch,
                  "k");
    require_shape(v, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "v");
    require_shape(out, detail::gated_delta_net::kStateDim, geometry.value_heads, geometry.tokens,
                  batch, "out");
    require_shape(g, geometry.value_heads, geometry.tokens, batch, 1, "g");
    require_shape(beta, geometry.value_heads, geometry.tokens, batch, 1, "beta");
    if (ssm_states.ne[0] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[1] != detail::gated_delta_net::kStateDim ||
        ssm_states.ne[2] != geometry.value_heads || ssm_states.ne[3] < geometry.tokens * batch) {
        throw std::invalid_argument("gated_delta_net: invalid shape for ssm_states snapshot");
    }
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, "valid_columns"); }
    require_shape(initial_state_slots, batch, 1, 1, 1, "initial_state_slots");
    require_shape(snapshot_base_slots, batch, 1, 1, 1, "snapshot_base_slots");

    require_contiguous_nonnull(q, "q");
    require_contiguous_nonnull(k, "k");
    require_contiguous_nonnull(v, "v");
    require_contiguous_nonnull(g, "g");
    require_contiguous_nonnull(beta, "beta");
    require_contiguous_nonnull(ssm_states, "ssm_states");
    if (masked) { require_contiguous_nonnull(valid_columns, "valid_columns"); }
    require_contiguous_nonnull(initial_state_slots, "initial_state_slots");
    require_contiguous_nonnull(snapshot_base_slots, "snapshot_base_slots");
    require_contiguous_nonnull(out, "out");

    require_scale(scale);
    return geometry;
}

void validate_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, const Tensor& ssm_state_in,
                      const Tensor& ssm_state_out, const Tensor& out) {
    // ssm_state_out carries the running-state contract validated by validate_recurrent;
    // ssm_state_in is an equally-shaped read view (may alias ssm_state_out for in-place).
    const Geometry geometry = validate_recurrent(q, k, v, g, beta, scale, ssm_state_out, out);
    require_dtype(ssm_state_in, DType::BF16, "ssm_state_in must be BF16");
    require_shape(ssm_state_in, detail::gated_delta_net::kStateDim,
                  detail::gated_delta_net::kStateDim, geometry.value_heads, 1, "ssm_state_in");
    require_contiguous_nonnull(ssm_state_in, "ssm_state_in");
}

struct ChunkedWorkspace {
    Tensor normalized_q;
    Tensor normalized_k;
    Tensor padded_v;
    Tensor padded_g;
    Tensor padded_beta;
    Tensor padded_out;
    DeviceSpan stage;
};

template <class Allocator>
ChunkedWorkspace allocate_chunked_workspace(Allocator& allocator, std::int32_t qk_heads,
                                            std::int32_t value_heads, std::int32_t tokens,
                                            bool normalize_qk) {
    ChunkedWorkspace out;
    constexpr auto chunk = detail::gated_delta_net::kChunkSize;
    if (tokens < chunk) { return out; }
    const auto padded64 = ((std::int64_t(tokens) - 1) / chunk + 1) * chunk;
    if (padded64 > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("gated_delta_net: padded token count exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(padded64);
    if (normalize_qk || padded != tokens) {
        out.normalized_q =
            allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads, padded});
        out.normalized_k =
            allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads, padded});
    }
    if (padded != tokens) {
        out.padded_v = allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, value_heads, padded});
        out.padded_g = allocator.alloc(DType::FP32, {value_heads, padded});
        out.padded_beta = allocator.alloc(DType::FP32, {value_heads, padded});
        out.padded_out = allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, value_heads, padded});
    }
    out.stage =
        allocator.alloc_bytes(detail::gated_delta_net::chunked_workspace_bytes(value_heads, padded));
    return out;
}

} // namespace

std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                     std::int32_t value_heads, bool normalize_qk,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (!detail::gated_delta_net::are_head_counts_valid(qk_heads, value_heads) || min_tokens <= 0 ||
        max_tokens < min_tokens) {
        throw std::invalid_argument("gated_delta_net workspace: invalid profile or interval");
    }
    const auto bytes_for = [&](std::int32_t tokens) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_chunked_workspace(layout, qk_heads, value_heads, tokens, normalize_qk);
        return layout.peak_bytes(1);
    };
    auto bytes = bytes_for(max_tokens);
    // A partial final chunk also needs padded inputs/output; the preceding
    // partial shape can therefore need more workspace than an exact multiple.
    if (max_tokens > min_tokens && max_tokens % detail::gated_delta_net::kChunkSize == 0) {
        bytes = std::max(bytes, bytes_for(max_tokens - 1));
    }
    return bytes;
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    if (q.ne[2] != 1) {
        gated_delta_net(q, k, v, g, beta, scale, normalize_qk, ws, ssm_state, ssm_state, out,
                        stream);
        return;
    }
    validate_recurrent(q, k, v, g, beta, scale, ssm_state, out);

    (void)ws;
    detail::gated_delta_net::launch_recurrent(q, k, v, g, beta, scale, normalize_qk, ssm_state, out,
                                              stream);
}

void gated_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              Tensor& ssm_states, const Tensor& valid_columns,
                              const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                              Tensor& out, cudaStream_t stream) {
    validate_recurrent_snapshot(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                initial_state_slots, snapshot_base_slots, out);

    detail::gated_delta_net::launch_recurrent_snapshot(
        q, k, v, g, beta, scale, normalize_qk, ssm_states, valid_columns, initial_state_slots,
        snapshot_base_slots, out, stream);
}

void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     cudaStream_t stream) {
    validate_chunked(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);

    auto scratch_scope   = ws.scope();
    const std::int32_t T = q.ne[2];
    if (T < detail::gated_delta_net::kChunkSize) {
        detail::gated_delta_net::launch_recurrent_inout(q, k, v, g, beta, scale, normalize_qk,
                                                        ssm_state_in, ssm_state_out, out, stream);
        return;
    }
    ChunkedWorkspace scratch = allocate_chunked_workspace(ws, q.ne[1], v.ne[1], T, normalize_qk);
    Tensor q_compute = q, k_compute = k, v_compute = v, g_compute = g, beta_compute = beta;
    Tensor output = out;
    const bool padded = scratch.padded_out.data != nullptr;
    const auto copy_padded = [&](const Tensor& source, Tensor& destination) {
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(destination.data) + source.bytes(), 0,
                                   destination.bytes() - source.bytes(), stream));
    };
    if (normalize_qk) {
        Tensor nq = scratch.normalized_q.slice(2, 0, T);
        Tensor nk = scratch.normalized_k.slice(2, 0, T);
        l2norm(q, 1.0e-6f, nq, stream);
        l2norm(k, 1.0e-6f, nk, stream);
        if (padded) {
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(scratch.normalized_q.data) + q.bytes(), 0,
                                       scratch.normalized_q.bytes() - q.bytes(), stream));
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(scratch.normalized_k.data) + k.bytes(), 0,
                                       scratch.normalized_k.bytes() - k.bytes(), stream));
        }
        q_compute = scratch.normalized_q;
        k_compute = scratch.normalized_k;
    } else if (padded) {
        copy_padded(q, scratch.normalized_q);
        copy_padded(k, scratch.normalized_k);
        q_compute = scratch.normalized_q;
        k_compute = scratch.normalized_k;
    }
    if (padded) {
        // Keep the last partial block on the same chunked arithmetic as its
        // causal prefix in a longer prompt. Zero decay and update gates make
        // padding leave the final state unchanged.
        copy_padded(v, scratch.padded_v);
        copy_padded(g, scratch.padded_g);
        copy_padded(beta, scratch.padded_beta);
        v_compute = scratch.padded_v;
        g_compute = scratch.padded_g;
        beta_compute = scratch.padded_beta;
        output = scratch.padded_out;
    }
    detail::gated_delta_net::launch_chunked(q_compute, k_compute, v_compute, g_compute, beta_compute, scale,
                                            ssm_state_in, ssm_state_out, output,
                                            scratch.stage.data, scratch.stage.bytes, stream);
    if (padded) {
        CUDA_CHECK(cudaMemcpyAsync(out.data, output.data, out.bytes(), cudaMemcpyDeviceToDevice, stream));
    }
}

} // namespace sinfer::ops
