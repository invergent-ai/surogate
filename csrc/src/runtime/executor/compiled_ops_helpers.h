// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Helper functions for compiled operation dispatch (logging, debugging, etc).

#ifndef SUROGATE_SRC_DSL_COMPILED_OPS_HELPERS_H
#define SUROGATE_SRC_DSL_COMPILED_OPS_HELPERS_H

#include "runtime/executor/saved_tensor_cache.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/executor/graph_executor_utils.h"
#include "utilities/tensor.h"

namespace dsl {

// Global state for QKV gradient tracking (shared across split op files)
extern std::vector<std::byte*> g_qkv_dA_ptr_by_layer;
extern std::vector<int> g_qkv_dA_micro_by_layer;

// MoE compact weight information
struct MoeCompactInfo {
    std::vector<int> host_offsets;
    std::vector<int> active_experts;
    int num_active = 0;
    bool weight_is_compact = false;
};

// Build MoE compact info from expert offsets (device memory)
MoeCompactInfo build_moe_compact_info(const int* expert_offsets_dev,
                                      int num_experts,
                                      int weight_experts,
                                      cudaStream_t stream,
                                      int layer_idx,
                                      const char* tag);

// Build MoE compact info from expert offsets (host memory)
MoeCompactInfo build_moe_compact_info_from_host(const int* host_offsets,
                                                int num_experts,
                                                int weight_experts,
                                                int layer_idx,
                                                const char* tag);

int env_int(const char* name, int fallback);
float env_float(const char* name, float fallback);

// Build a Tensor wrapping a raw GPU pointer with proper Rank/Device set.
// IMPORTANT: Manual `Tensor{}` leaves Rank=0, Device=-1 which makes .bytes()
// and .nelem() return wrong values. Always use this helper instead.
inline Tensor make_raw_tensor(void* ptr, ETensorDType dtype, const std::vector<long>& shape, int device) {
    Tensor t{};
    t.Data = static_cast<std::byte*>(ptr);
    t.DType = dtype;
    t.Rank = static_cast<int>(shape.size());
    t.Device = device;
    for (int i = 0; i < t.Rank; ++i)
        t.Sizes[i] = shape[i];
    for (int i = t.Rank; i < MAX_TENSOR_DIM; ++i)
        t.Sizes[i] = 1;
    return t;
}

inline std::size_t tensor_shape_nelem(const std::vector<long>& shape) {
    std::size_t nelem = 1;
    for (long dim : shape) {
        nelem *= static_cast<std::size_t>(dim);
    }
    return nelem;
}

inline Tensor make_persistent_tensor(DslRunState& run_state,
                                     SavedTensorCache& cache,
                                     const std::string& key,
                                     ETensorDType dtype,
                                     const std::vector<long>& shape,
                                     const char* op_name = nullptr) {
    const size_t elem_sz = get_dtype_size(dtype);
    const size_t nelem = tensor_shape_nelem(shape);
    const size_t bytes = nelem * elem_sz;
    if (bytes == 0) {
        return make_raw_tensor(nullptr, dtype, shape, 0);
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool capturing = (cudaStreamIsCapturing(run_state.MainStream, &capture_status) == cudaSuccess &&
                            capture_status != cudaStreamCaptureStatusNone);

    // Routed through the cache so dispatch-PP's per-stage reset_to_pool() recycles the
    // buffer (no cudaFree churn). Plain fallback path (no arena) for op-managed keys.
    void* buf = cache.acquire(key, bytes, capturing, op_name);

    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    return make_raw_tensor(buf, dtype, shape, device);
}

inline Tensor ensure_output_tensor_or_persistent(const Tensor& candidate,
                                                 DslRunState& run_state,
                                                 SavedTensorCache& cache,
                                                 const std::string& key,
                                                 ETensorDType dtype,
                                                 const std::vector<long>& shape,
                                                 const char* op_name = nullptr) {
    // Backward ops: never reuse candidate.Data. Activation slots can carry
    // stale pointers from earlier ops (view metadata sharing, freed temps),
    // which propagate through view_backward → concat_backward → split and
    // segfault inside cudaMemcpy2DAsync. Always route backward outputs to
    // persistent buffers so they have stable lifetimes across the op chain.
    const bool is_backward = op_name != nullptr &&
                             (std::strstr(op_name, "_backward") != nullptr || std::strstr(op_name, "_grad") != nullptr);

    if (!is_backward && candidate.Data && candidate.DType == dtype &&
        static_cast<std::size_t>(candidate.nelem()) == tensor_shape_nelem(shape)) {
        return make_raw_tensor(candidate.Data, dtype, shape, candidate.Device);
    }
    return make_persistent_tensor(run_state, cache, key, dtype, shape, op_name);
}

bool refresh_moe_experts_if_needed(int layer_idx,
                                   const int* host_offsets,
                                   int num_experts,
                                   DslParamStore& weights,
                                   cudaStream_t stream);

/// Derive head_size from a QKV tensor's actual shape for hybrid models
/// where different block types have different head dimensions.
/// Falls back to config_hs if the tensor shape is ambiguous.
// Resolve the physical layer index of an op: explicit attr, tensor-ref
// layer_idx, or a "blocks[N]." prefix on any input/output name (also behind
// "saved." / "d_" prefixes for backward ops).
inline int op_layer_idx(const CompiledOp& op) {
    int layer_idx = op.attrs.layer_idx;
    for (const auto& t : op.inputs) {
        if (layer_idx >= 0) break;
        if (t.layer_idx >= 0) layer_idx = t.layer_idx;
    }
    auto try_name = [&](const std::string& name) {
        if (layer_idx >= 0 || name.empty()) return;
        std::string_view v(name);
        if (v.rfind("saved.", 0) == 0) v.remove_prefix(6);
        if (v.rfind("d_", 0) == 0) v.remove_prefix(2);
        int idx = -1;
        std::string field;
        if (parse_block_param(v, idx, field) && idx >= 0) layer_idx = idx;
    };
    for (const auto& t : op.inputs) try_name(t.name);
    for (const auto& t : op.outputs) try_name(t.name);
    return layer_idx;
}

// Resolve per-layer attention head dims for a dispatch site: override the
// global-config values with the runtime config's per-layer dims (hybrid
// models with per-layer query head counts, e.g. Laguna: 48 query heads on
// full-attention layers, 64 on sliding ones), then reconcile against the
// actual tensor shape. Rank-4 [B, T, H, D] tensors are authoritative for the
// head size (the derived per-layer head_size can be wrong when a layer
// overrides the query head count and exposes no rank-1 q_norm_weight to
// correct the out_weight-cols / global-head-count derivation) and for the
// total head count (fewer heads than expected: shared-KV / Gemma4 k_eq_v
// layers; more: Laguna-style per-layer query heads).
inline void resolve_attn_head_dims(const DslRuntimeConfig& rc,
                                   int layer_idx,
                                   const Tensor& qkv,
                                   int& Hq,
                                   int& Hkv,
                                   int& Hs) {
    const int tensor_hs = (qkv.Rank == 4) ? static_cast<int>(qkv.Sizes[3]) : 0;
    if (layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < rc.per_layer_dims.size()) {
        const BlockTypeDims& d = rc.per_layer_dims[static_cast<std::size_t>(layer_idx)];
        const long hs = (tensor_hs > 0) ? tensor_hs : d.head_size;
        if (hs > 0) {
            if (d.attn_dim > 0 && (d.attn_dim % hs) == 0) {
                Hq = static_cast<int>(d.attn_dim / hs);
            }
            if (d.kv_dim > 0 && (d.kv_dim % hs) == 0) {
                Hkv = static_cast<int>(d.kv_dim / hs);
            }
            Hs = static_cast<int>(hs);
        }
    }
    if (qkv.Rank == 4) {
        const int actual_heads = static_cast<int>(qkv.Sizes[2]);
        if (actual_heads < Hq + 2 * Hkv) {
            Hkv = (actual_heads - Hq) / 2;
            if (Hkv < 0) Hkv = 0;
        } else if (actual_heads > Hq + 2 * Hkv) {
            Hq = actual_heads - 2 * Hkv;
        }
    }
}

inline int derive_head_size(const Tensor& qkv, int Hq, int Hkv, int config_hs) {
    if (qkv.Rank == 4) {
        return static_cast<int>(qkv.Sizes[3]);
    }
    if (qkv.Rank == 3) {
        const int total_heads = Hq + 2 * Hkv;
        if (total_heads > 0) {
            return static_cast<int>(qkv.Sizes[2]) / total_heads;
        }
    }
    return config_hs;
}

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_COMPILED_OPS_HELPERS_H
